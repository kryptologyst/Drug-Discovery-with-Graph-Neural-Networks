"""Graph Neural Network models for molecular property prediction."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GINConv, GraphSAGE, MessagePassing,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm1d, LayerNorm
)
from torch_geometric.utils import softmax


class GCN(nn.Module):
    """Graph Convolutional Network for molecular property prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        pooling: str = "mean"
    ):
        """Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            pooling: Global pooling method ('mean', 'max', 'add')
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.pooling = pooling
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            
        Returns:
            Graph-level predictions
        """
        # Store input for residual connection
        if self.use_residual:
            x_input = x
        
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            
            if self.batch_norms:
                x_new = self.batch_norms[i](x_new)
            
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection
            if self.use_residual and i > 0 and x.size(-1) == x_new.size(-1):
                x_new = x_new + x
            
            x = x_new
        
        # Global pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # MLP head
        return self.mlp(x).squeeze()


class GIN(nn.Module):
    """Graph Isomorphism Network for molecular property prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        pooling: str = "mean"
    ):
        """Initialize GIN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of GIN layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            pooling: Global pooling method
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        
        # MLPs for GIN layers
        self.mlps = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlps.append(mlp)
        self.convs.append(GINConv(mlp))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.mlps.append(mlp)
            self.convs.append(GINConv(mlp))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.mlps.append(mlp)
            self.convs.append(GINConv(mlp))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply GIN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.batch_norms:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        
        # MLP head
        return self.mlp_head(x).squeeze()


class GraphSAGE(nn.Module):
    """GraphSAGE for molecular property prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        pooling: str = "mean"
    ):
        """Initialize GraphSAGE model."""
        super().__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(GraphSAGE(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.batch_norms:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        
        # MLP head
        return self.mlp_head(x).squeeze()


class MPNNLayer(MessagePassing):
    """Message Passing Neural Network layer."""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        """Initialize MPNN layer.
        
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
        """
        super().__init__(aggr='add')
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute messages."""
        # Concatenate source node features and edge features
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(message_input)
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node features."""
        # Concatenate aggregated messages with current node features
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class MPNN(nn.Module):
    """Message Passing Neural Network for molecular property prediction."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        pooling: str = "mean"
    ):
        """Initialize MPNN model."""
        super().__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        
        # MPNN layers
        self.mpnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for _ in range(num_layers):
            self.mpnn_layers.append(MPNNLayer(node_dim, edge_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(node_dim))
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply MPNN layers
        for i, mpnn_layer in enumerate(self.mpnn_layers):
            x = mpnn_layer(x, edge_index, edge_attr)
            
            if self.batch_norms:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        
        # MLP head
        return self.mlp_head(x).squeeze()


def create_model(model_name: str, **kwargs) -> nn.Module:
    """Create a GNN model by name.
    
    Args:
        model_name: Name of the model ('GCN', 'GIN', 'GraphSAGE', 'MPNN')
        **kwargs: Model parameters
        
    Returns:
        Initialized model
    """
    models = {
        'GCN': GCN,
        'GIN': GIN,
        'GraphSAGE': GraphSAGE,
        'MPNN': MPNN
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)
