"""Unit tests for molecular property prediction models."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import GCN, GIN, GraphSAGE, MPNN, create_model
from data import create_synthetic_dataset
from utils import set_seed, get_device, count_parameters


class TestModels:
    """Test GNN models."""
    
    def setup_method(self):
        """Setup test data."""
        set_seed(42)
        
        # Create synthetic data
        self.num_nodes = 10
        self.num_features = 9
        self.num_edge_features = 3
        self.batch_size = 2
        
        # Create synthetic graph data
        x = torch.randn(self.num_nodes, self.num_features)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(edge_index.size(1), self.num_edge_features)
        batch = torch.zeros(self.num_nodes, dtype=torch.long)
        y = torch.randn(1)
        
        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)
        
        # Create batch
        batch_list = [self.data, self.data]
        self.batch = DataLoader(batch_list, batch_size=self.batch_size)
    
    def test_gcn_forward(self):
        """Test GCN forward pass."""
        model = GCN(
            input_dim=self.num_features,
            hidden_dim=32,
            output_dim=1,
            num_layers=2
        )
        
        batch = next(iter(self.batch))
        output = model(batch.x, batch.edge_index, batch.batch)
        
        assert output.shape == (self.batch_size,)
        assert not torch.isnan(output).any()
    
    def test_gin_forward(self):
        """Test GIN forward pass."""
        model = GIN(
            input_dim=self.num_features,
            hidden_dim=32,
            output_dim=1,
            num_layers=2
        )
        
        batch = next(iter(self.batch))
        output = model(batch.x, batch.edge_index, batch.batch)
        
        assert output.shape == (self.batch_size,)
        assert not torch.isnan(output).any()
    
    def test_graphsage_forward(self):
        """Test GraphSAGE forward pass."""
        model = GraphSAGE(
            input_dim=self.num_features,
            hidden_dim=32,
            output_dim=1,
            num_layers=2
        )
        
        batch = next(iter(self.batch))
        output = model(batch.x, batch.edge_index, batch.batch)
        
        assert output.shape == (self.batch_size,)
        assert not torch.isnan(output).any()
    
    def test_mpnn_forward(self):
        """Test MPNN forward pass."""
        model = MPNN(
            node_dim=self.num_features,
            edge_dim=self.num_edge_features,
            hidden_dim=32,
            output_dim=1,
            num_layers=2
        )
        
        batch = next(iter(self.batch))
        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        assert output.shape == (self.batch_size,)
        assert not torch.isnan(output).any()
    
    def test_create_model(self):
        """Test model creation function."""
        # Test GCN creation
        gcn = create_model('GCN', input_dim=self.num_features, hidden_dim=32, output_dim=1)
        assert isinstance(gcn, GCN)
        
        # Test GIN creation
        gin = create_model('GIN', input_dim=self.num_features, hidden_dim=32, output_dim=1)
        assert isinstance(gin, GIN)
        
        # Test invalid model name
        with pytest.raises(ValueError):
            create_model('InvalidModel', input_dim=self.num_features)
    
    def test_model_parameters(self):
        """Test model parameter counting."""
        model = GCN(input_dim=self.num_features, hidden_dim=32, output_dim=1)
        total_params = count_parameters(model)
        assert total_params > 0
        
        # Test that all parameters are trainable by default
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params == trainable_params


class TestData:
    """Test data utilities."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(num_samples=10, num_atoms_range=(5, 15))
        
        assert len(dataset) == 10
        
        # Check first sample
        sample = dataset[0]
        assert isinstance(sample, Data)
        assert sample.x.shape[1] == 9  # Default num_features
        assert sample.y.shape == (1,)
        assert sample.edge_index.size(0) == 2
        
        # Check that all samples have reasonable sizes
        for sample in dataset:
            assert 5 <= sample.num_nodes <= 15
            assert sample.num_edges > 0


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        rand1 = torch.randn(5)
        
        set_seed(42)
        rand2 = torch.randn(5)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        num_params = count_parameters(model)
        assert num_params == 55  # 10*5 + 5 bias


class TestIntegration:
    """Integration tests."""
    
    def test_training_step(self):
        """Test a single training step."""
        set_seed(42)
        
        # Create model and data
        model = GCN(input_dim=9, hidden_dim=32, output_dim=1)
        dataset = create_synthetic_dataset(num_samples=4, num_atoms_range=(5, 10))
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Training step
        model.train()
        batch = next(iter(dataloader))
        
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
        assert loss.item() >= 0
    
    def test_evaluation_step(self):
        """Test evaluation step."""
        set_seed(42)
        
        # Create model and data
        model = GCN(input_dim=9, hidden_dim=32, output_dim=1)
        dataset = create_synthetic_dataset(num_samples=4, num_atoms_range=(5, 10))
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Evaluation step
        model.eval()
        batch = next(iter(dataloader))
        
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.batch)
        
        assert pred.shape == (batch.y.shape[0],)
        assert not torch.isnan(pred).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
