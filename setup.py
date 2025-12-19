#!/usr/bin/env python3
"""Setup script for Drug Discovery with GNNs project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def setup_environment():
    """Set up the project environment."""
    print("🚀 Setting up Drug Discovery with GNNs project...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create necessary directories
    directories = [
        "data",
        "assets/models",
        "assets/results", 
        "assets/predictions",
        "assets/plots",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("⚠️  Pre-commit installation failed, but continuing...")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but setup continues...")
    
    return True


def create_sample_config():
    """Create a sample configuration file."""
    sample_config = """# Sample configuration for quick start
model:
  name: "GCN"
  input_dim: 9
  hidden_dim: 64
  output_dim: 1
  num_layers: 3
  dropout: 0.1
  use_batch_norm: true
  use_residual: true
  pooling: "mean"

data:
  dataset: "ESOL"
  data_dir: "./data"
  batch_size: 32
  num_workers: 4
  scaffold_split: true
  test_size: 0.2
  val_size: 0.1
  random_state: 42
  use_rdkit_features: true
  max_atoms: 200

training:
  epochs: 50
  lr: 0.001
  weight_decay: 1e-4
  optimizer: "adam"
  scheduler:
    name: "plateau"
    patience: 10
    factor: 0.5
  patience: 20
  grad_clip: 1.0

evaluation:
  save_predictions: true
  plot_results: true
  compare_models: false

logging:
  use_wandb: false
  log_interval: 10

device:
  auto: true
"""
    
    config_path = Path("configs/sample_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(sample_config)
    print(f"📄 Created sample configuration: {config_path}")


def main():
    """Main setup function."""
    print("🧬 Drug Discovery with Graph Neural Networks - Setup")
    print("=" * 60)
    
    if not setup_environment():
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    
    create_sample_config()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model:")
    print("   python scripts/train.py --config configs/gcn_esol.yaml")
    print("\n2. Evaluate a model:")
    print("   python scripts/evaluate.py --model-path assets/models/GCN_ESOL/checkpoint_best.pt --config configs/gcn_esol.yaml")
    print("\n3. Launch the demo:")
    print("   streamlit run demo/app.py")
    print("\n4. Run the notebook:")
    print("   jupyter notebook notebooks/drug_discovery_demo.ipynb")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()
