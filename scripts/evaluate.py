#!/usr/bin/env python3
"""Evaluation script for molecular property prediction with GNNs."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import MolecularDataModule
from models import create_model
from eval import Evaluator, create_leaderboard
from utils import get_device, load_config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate GNN for molecular property prediction")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions")
    parser.add_argument("--plot-results", action="store_true", help="Plot results")
    parser.add_argument("--compare-models", nargs="+", default=None, help="Compare multiple models")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config.data.data_dir = args.data_dir
    config.data.batch_size = args.batch_size
    
    # Setup data
    print("Loading dataset...")
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
    
    if args.compare_models:
        # Compare multiple models
        models = {}
        for model_path in args.compare_models:
            model_name = os.path.basename(model_path).split('_')[0]
            print(f"\nLoading {model_name} from {model_path}...")
            
            model = create_model(**config.model)
            checkpoint = torch.load(model_path, map_location=get_device())
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(get_device())
            
            models[model_name] = model
        
        # Evaluate all models
        evaluator = Evaluator(models[list(models.keys())[0]], get_device())
        results = evaluator.compare_models(
            models=models,
            data_loader=data_module.test_loader,
            task_type=dataset_info["task_type"],
            save_path=f"./assets/results/model_comparison_{config.data.dataset}.json"
        )
        
        # Create leaderboard
        primary_metric = "r2" if dataset_info["task_type"] == "regression" else "roc_auc"
        create_leaderboard(results, primary_metric, dataset_info["task_type"])
        
    else:
        # Evaluate single model
        print(f"\nLoading {config.model.name} from {args.model_path}...")
        model = create_model(**config.model)
        checkpoint = torch.load(args.model_path, map_location=get_device())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(get_device())
        
        # Create evaluator
        evaluator = Evaluator(model, get_device())
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        if dataset_info["task_type"] == "regression":
            metrics = evaluator.evaluate_regression(
                data_module.test_loader,
                save_predictions=args.save_predictions,
                save_path=f"./assets/predictions/{config.model.name}_{config.data.dataset}_predictions.npz"
            )
        else:
            metrics = evaluator.evaluate_classification(
                data_module.test_loader,
                save_predictions=args.save_predictions,
                save_path=f"./assets/predictions/{config.model.name}_{config.data.dataset}_predictions.npz"
            )
        
        # Print results
        print(f"\nTest Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Plot results if requested
        if args.plot_results:
            print("\nGenerating plots...")
            os.makedirs("./assets/plots", exist_ok=True)
            evaluator.plot_predictions(
                data_module.test_loader,
                task_type=dataset_info["task_type"],
                save_path=f"./assets/plots/{config.model.name}_{config.data.dataset}_predictions.png",
                title=f"{config.model.name} - {config.data.dataset}"
            )
        
        # Save results
        results_path = f"./assets/results/{config.model.name}_{config.data.dataset}_evaluation.yaml"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        final_results = {
            'config': OmegaConf.to_container(config),
            'metrics': metrics,
            'dataset_info': dataset_info,
            'model_path': args.model_path
        }
        
        with open(results_path, 'w') as f:
            yaml.dump(final_results, f, default_flow_style=False)
        
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
