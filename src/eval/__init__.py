"""Evaluation utilities for molecular property prediction."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..utils import get_device


class Evaluator:
    """Evaluator class for molecular property prediction models."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_regression(
        self,
        data_loader: DataLoader,
        save_predictions: bool = False,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate model on regression task.
        
        Args:
            data_loader: Data loader for evaluation
            save_predictions: Whether to save predictions
            save_path: Path to save predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = []
        targets = []
        losses = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                
                # Forward pass
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    pred = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Compute loss
                loss = nn.MSELoss()(pred, batch.y)
                losses.append(loss.item())
                
                # Store predictions and targets
                predictions.extend(pred.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Compute metrics
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # Additional metrics
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        pearson_corr = np.corrcoef(targets, predictions)[0, 1]
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'pearson_corr': pearson_corr,
            'mean_loss': np.mean(losses)
        }
        
        # Save predictions if requested
        if save_predictions and save_path:
            np.savez(save_path, predictions=predictions, targets=targets)
            print(f"Predictions saved to {save_path}")
        
        return metrics
    
    def evaluate_classification(
        self,
        data_loader: DataLoader,
        threshold: float = 0.5,
        save_predictions: bool = False,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate model on classification task.
        
        Args:
            data_loader: Data loader for evaluation
            threshold: Classification threshold
            save_predictions: Whether to save predictions
            save_path: Path to save predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = []
        targets = []
        probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                
                # Forward pass
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    logits = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Apply sigmoid for binary classification
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                
                probabilities.extend(probs.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        probabilities = np.array(probabilities)
        
        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )
        
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='binary')
        recall = recall_score(targets, predictions, average='binary')
        f1 = f1_score(targets, predictions, average='binary')
        
        # ROC-AUC and PR-AUC
        try:
            roc_auc = roc_auc_score(targets, probabilities)
            precision_curve, recall_curve, _ = precision_recall_curve(targets, probabilities)
            pr_auc = auc(recall_curve, precision_curve)
        except ValueError:
            roc_auc = 0.0
            pr_auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        # Save predictions if requested
        if save_predictions and save_path:
            np.savez(save_path, 
                    predictions=predictions, 
                    targets=targets, 
                    probabilities=probabilities)
            print(f"Predictions saved to {save_path}")
        
        return metrics
    
    def plot_predictions(
        self,
        data_loader: DataLoader,
        task_type: str = "regression",
        save_path: Optional[str] = None,
        title: str = "Model Predictions"
    ) -> None:
        """Plot predictions vs targets.
        
        Args:
            data_loader: Data loader for evaluation
            task_type: Type of task ('regression' or 'classification')
            save_path: Path to save plot
            title: Plot title
        """
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating predictions"):
                batch = batch.to(self.device)
                
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    pred = self.model(batch.x, batch.edge_index, batch.batch)
                
                if task_type == "classification":
                    pred = torch.sigmoid(pred)
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if task_type == "regression":
            # Scatter plot for regression
            plt.scatter(targets, predictions, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{title} - Regression')
            
            # Add R² score
            r2 = r2_score(targets, predictions)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # ROC curve for classification
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(targets, predictions)
            roc_auc = roc_auc_score(targets, predictions)
            
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{title} - Classification')
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data_loader: DataLoader,
        task_type: str = "regression",
        save_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model names and models
            data_loader: Data loader for evaluation
            task_type: Type of task
            save_path: Path to save comparison results
            
        Returns:
            Dictionary of metrics for each model
        """
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            evaluator = Evaluator(model, self.device)
            
            if task_type == "regression":
                metrics = evaluator.evaluate_regression(data_loader)
            else:
                metrics = evaluator.evaluate_classification(data_loader)
            
            results[name] = metrics
        
        # Print comparison table
        self._print_comparison_table(results, task_type)
        
        # Save results if requested
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {save_path}")
        
        return results
    
    def _print_comparison_table(self, results: Dict[str, Dict[str, float]], task_type: str) -> None:
        """Print comparison table for multiple models."""
        print(f"\n{'='*60}")
        print(f"Model Comparison - {task_type.title()}")
        print(f"{'='*60}")
        
        # Get all metric names
        all_metrics = set()
        for metrics in results.values():
            all_metrics.update(metrics.keys())
        
        # Print header
        header = f"{'Model':<15}"
        for metric in sorted(all_metrics):
            header += f"{metric:<10}"
        print(header)
        print("-" * len(header))
        
        # Print results
        for name, metrics in results.items():
            row = f"{name:<15}"
            for metric in sorted(all_metrics):
                value = metrics.get(metric, 0.0)
                row += f"{value:<10.4f}"
            print(row)
        
        print(f"{'='*60}")


def create_leaderboard(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = "r2",
    task_type: str = "regression"
) -> None:
    """Create a leaderboard from model comparison results.
    
    Args:
        results: Dictionary of model results
        primary_metric: Primary metric for ranking
        task_type: Type of task
    """
    print(f"\n{'='*50}")
    print(f"LEADERBOARD - {task_type.upper()}")
    print(f"Primary Metric: {primary_metric.upper()}")
    print(f"{'='*50}")
    
    # Sort models by primary metric
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get(primary_metric, 0),
        reverse=True
    )
    
    print(f"{'Rank':<5}{'Model':<15}{primary_metric.upper():<10}{'MAE':<10}{'RMSE':<10}")
    print("-" * 50)
    
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        primary_value = metrics.get(primary_metric, 0.0)
        mae = metrics.get('mae', 0.0)
        rmse = metrics.get('rmse', 0.0)
        
        print(f"{rank:<5}{name:<15}{primary_value:<10.4f}{mae:<10.4f}{rmse:<10.4f}")
    
    print(f"{'='*50}")
