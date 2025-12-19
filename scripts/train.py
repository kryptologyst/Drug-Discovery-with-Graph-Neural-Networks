#!/usr/bin/env python3
"""Training script for molecular property prediction with GNNs."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import MolecularDataModule
from models import create_model
from train import Trainer
from utils import set_seed, get_device, load_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GNN for molecular property prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device != "auto":
        config.device.auto = False
        config.device.name = args.device
    
    if args.wandb:
        config.logging.use_wandb = True
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Setup data
    print("\nLoading dataset...")
    data_module = MolecularDataModule(
        dataset_name=config.data.dataset,
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        scaffold_split=config.data.scaffold_split,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.data.random_state,
        use_rdkit_features=config.data.use_rdkit_features,
        max_atoms=config.data.max_atoms
    )
    
    # Print dataset info
    dataset_info = data_module.get_dataset_info()
    print(f"\nDataset Info:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # Update model config with dataset info
    config.model.input_dim = dataset_info["num_node_features"]
    if config.model.name == "MPNN":
        config.model.node_dim = dataset_info["num_node_features"]
        config.model.edge_dim = dataset_info["num_edge_features"]
    
    # Create model
    print(f"\nCreating {config.model.name} model...")
    model = create_model(**config.model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=config,
        save_dir=f"./assets/models/{config.model.name}_{config.data.dataset}",
        use_wandb=config.logging.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=get_device())
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_score']
    
    # Train model
    print("\nStarting training...")
    training_results = trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate()
    
    # Print final results
    print(f"\nTraining completed!")
    print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    print(f"Epochs trained: {training_results['epochs_trained']}")
    print(f"\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save final results
    results_path = f"./assets/results/{config.model.name}_{config.data.dataset}_results.yaml"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    final_results = {
        'config': OmegaConf.to_container(config),
        'training_results': training_results,
        'test_metrics': test_metrics,
        'dataset_info': dataset_info
    }
    
    with open(results_path, 'w') as f:
        yaml.dump(final_results, f, default_flow_style=False)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
