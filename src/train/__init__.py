"""Training utilities for molecular property prediction."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..utils import ModelCheckpoint, early_stopping, get_device
from ..data import MolecularDataModule


class Trainer:
    """Trainer class for molecular property prediction."""
    
    def __init__(
        self,
        model: nn.Module,
        data_module: MolecularDataModule,
        config: Dict,
        save_dir: str = "./assets/models",
        use_wandb: bool = False
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            data_module: Data module
            config: Training configuration
            save_dir: Directory to save models
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = get_device()
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup checkpointing
        os.makedirs(save_dir, exist_ok=True)
        self.checkpoint = ModelCheckpoint(save_dir, monitor="val_loss", mode="min")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="drug-discovery-gnn",
                config=config,
                name=f"{config['model']['name']}_{config['data']['dataset']}"
            )
            wandb.watch(self.model)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_name = self.config['training'].get('optimizer', 'adam')
        lr = self.config['training']['lr']
        weight_decay = self.config['training'].get('weight_decay', 1e-4)
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.config['training'].get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config['training'].get('scheduler', {})
        if not scheduler_config:
            return None
        
        scheduler_name = scheduler_config.get('name', 'step')
        
        if scheduler_name == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = scheduler_config.get('T_max', self.config['training']['epochs'])
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name == 'plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.data_module.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                # MPNN model with edge features
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                # Standard GNN models
                pred = self.model(batch.x, batch.edge_index, batch.batch)
            
            loss = self.criterion(pred, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.data_module.val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    pred = self.model(batch.x, batch.edge_index, batch.batch)
                
                loss = self.criterion(pred, batch.y)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model."""
        epochs = self.config['training']['epochs']
        patience = self.config['training'].get('patience', 20)
        
        train_losses = []
        val_losses = []
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            train_losses.append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self.validate_epoch()
            val_losses.append(val_metrics['val_loss'])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Early stopping
            should_stop, self.patience_counter, is_better = early_stopping(
                val_metrics['val_loss'],
                self.best_val_loss,
                self.patience_counter,
                patience=patience
            )
            
            if is_better:
                self.best_val_loss = val_metrics['val_loss']
            
            # Logging
            metrics = {**train_metrics, **val_metrics}
            metrics['epoch'] = epoch
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:03d}: "
                  f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"LR: {metrics['lr']:.6f}")
            
            # Wandb logging
            if self.use_wandb:
                wandb.log(metrics)
            
            # Checkpointing
            self.checkpoint.save_checkpoint(
                self.model, self.optimizer, epoch, metrics, is_better
            )
            
            # Early stopping
            if should_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Final logging
        if self.use_wandb:
            wandb.finish()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.current_epoch + 1
        }
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model on test set."""
        if test_loader is None:
            test_loader = self.data_module.test_loader
        
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = batch.to(self.device)
                
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    pred = self.model(batch.x, batch.edge_index, batch.batch)
                
                loss = self.criterion(pred, batch.y)
                total_loss += loss.item()
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Compute additional metrics
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        metrics = {
            'test_loss': avg_loss,
            'test_mae': mae,
            'test_mse': mse,
            'test_rmse': rmse,
            'test_r2': r2
        }
        
        print("Test Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
