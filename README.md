# Drug Discovery with Graph Neural Networks

A production-ready implementation of Graph Neural Networks for molecular property prediction in drug discovery.

## Features

- **Multiple GNN Architectures**: GCN, GIN, GraphSAGE, MPNN for molecular property prediction
- **Comprehensive Datasets**: ESOL, QM9, ZINC with RDKit featurization
- **Advanced Evaluation**: MAE, RMSE, ROC-AUC with scaffold-based splits
- **Interactive Demo**: Streamlit interface for molecular property prediction
- **Production Ready**: Type hints, comprehensive testing, CI/CD pipeline

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Drug-Discovery-with-Graph-Neural-Networks
cd Drug-Discovery-with-Graph-Neural-Networks

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### Basic Usage

```bash
# Train a GCN model on ESOL dataset
python scripts/train.py --config configs/gcn_esol.yaml

# Run evaluation
python scripts/evaluate.py --model-path assets/models/gcn_esol_best.pt

# Launch interactive demo
streamlit run demo/app.py
```

## Project Structure

```
src/
├── models/          # GNN model implementations
├── layers/          # Custom GNN layers
├── data/           # Data loading and preprocessing
├── utils/          # Utility functions
├── train/          # Training scripts
└── eval/           # Evaluation scripts

configs/            # YAML configuration files
scripts/            # Training and evaluation scripts
tests/              # Unit tests
assets/             # Model checkpoints and results
demo/               # Streamlit demo application
notebooks/          # Jupyter notebooks for analysis
```

## Models

### Supported Architectures

- **GCN**: Graph Convolutional Network (Kipf & Welling)
- **GIN**: Graph Isomorphism Network (Xu et al.)
- **GraphSAGE**: Graph Sample and Aggregate (Hamilton et al.)
- **MPNN**: Message Passing Neural Network (Gilmer et al.)

### Datasets

- **ESOL**: Aqueous solubility prediction
- **QM9**: Quantum mechanical properties
- **ZINC**: Drug-like molecules
- **Custom**: Generate synthetic molecular graphs

## Configuration

Models and experiments are configured via YAML files in `configs/`:

```yaml
model:
  name: "GCN"
  hidden_dim: 64
  num_layers: 3
  dropout: 0.1

data:
  dataset: "ESOL"
  batch_size: 32
  scaffold_split: true

training:
  epochs: 100
  lr: 0.001
  weight_decay: 1e-4
```

## Evaluation Metrics

- **Regression**: MAE, RMSE, R²
- **Classification**: ROC-AUC, Precision, Recall, F1-Score
- **Molecular**: Validity, Uniqueness, Novelty (for generative tasks)

## Demo

The interactive Streamlit demo allows you to:

- Input SMILES strings for molecular property prediction
- Visualize molecular graphs with RDKit
- Compare different GNN architectures
- Explore attention weights and feature importance

```bash
streamlit run demo/app.py
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ tests/
ruff check src/ scripts/ tests/
```

### Pre-commit Hooks

```bash
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{drug_discovery_gnn,
  title={Drug Discovery with Graph Neural Networks},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Drug-Discovery-with-Graph-Neural-Networks}
}
```

## Acknowledgments

- PyTorch Geometric team for the excellent GNN framework
- RDKit community for molecular informatics tools
- MoleculeNet team for curated molecular datasets
# Drug-Discovery-with-Graph-Neural-Networks
