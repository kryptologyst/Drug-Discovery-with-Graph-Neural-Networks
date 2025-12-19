"""Utility functions for drug discovery with GNNs."""

import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    from omegaconf import OmegaConf
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    from omegaconf import OmegaConf
    OmegaConf.save(config, save_path)


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        
    Returns:
        Model summary string
    """
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Summary:
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Model size: {total_params * 4 / 1024 / 1024:.2f} MB
"""
    return summary


def early_stopping(
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    patience: int = 10,
    min_delta: float = 1e-4
) -> Tuple[bool, int, bool]:
    """Implement early stopping logic.
    
    Args:
        val_loss: Current validation loss
        best_val_loss: Best validation loss so far
        patience_counter: Current patience counter
        patience: Number of epochs to wait
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        Tuple of (should_stop, updated_patience_counter, is_better)
    """
    is_better = val_loss < best_val_loss - min_delta
    
    if is_better:
        patience_counter = 0
    else:
        patience_counter += 1
    
    should_stop = patience_counter >= patience
    
    return should_stop, patience_counter, is_better


def normalize_features(x: torch.Tensor, method: str = "standard") -> torch.Tensor:
    """Normalize node features.
    
    Args:
        x: Input features tensor
        method: Normalization method ('standard', 'minmax', 'none')
        
    Returns:
        Normalized features tensor
    """
    if method == "none":
        return x
    elif method == "standard":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = x.min(dim=0, keepdim=True)[0]
        max_val = x.max(dim=0, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """Compute basic molecular descriptors from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular descriptors
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        descriptors = {
            "mol_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": Descriptors.RingCount(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_hbd": Descriptors.NumHDonors(mol),
            "num_hba": Descriptors.NumHAcceptors(mol),
        }
        
        return descriptors
    except ImportError:
        print("RDKit not available. Install with: pip install rdkit")
        return {}
    except Exception as e:
        print(f"Error computing descriptors: {e}")
        return {}


class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(self, save_dir: str, monitor: str = "val_loss", mode: str = "min"):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max' for optimization direction
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, f"{self.save_dir}/checkpoint_latest.pt")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, f"{self.save_dir}/checkpoint_best.pt")
            self.best_score = metrics[self.monitor]
