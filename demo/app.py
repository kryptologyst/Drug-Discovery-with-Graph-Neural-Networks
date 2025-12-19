"""Streamlit demo for molecular property prediction with GNNs."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import MolecularDataModule
from models import create_model
from utils import get_device, compute_molecular_descriptors

# Try to import RDKit for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.warning("RDKit not available. Install with: pip install rdkit")


@st.cache_data
def load_model(model_path: str, config_path: str) -> Tuple[torch.nn.Module, Dict]:
    """Load trained model and configuration."""
    import yaml
    from omegaconf import OmegaConf
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Load dataset info to get dimensions
    data_module = MolecularDataModule(
        dataset_name=config.data.dataset,
        data_dir=config.data.data_dir,
        batch_size=1,
        scaffold_split=False,
        test_size=0.0,
        val_size=0.0
    )
    
    dataset_info = data_module.get_dataset_info()
    config.model.input_dim = dataset_info["num_node_features"]
    if config.model.name == "MPNN":
        config.model.node_dim = dataset_info["num_node_features"]
        config.model.edge_dim = dataset_info["num_edge_features"]
    
    # Create and load model
    model = create_model(**config.model)
    checkpoint = torch.load(model_path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(get_device())
    model.eval()
    
    return model, OmegaConf.to_container(config)


@st.cache_data
def smiles_to_graph(smiles: str) -> Optional[Dict]:
    """Convert SMILES to graph representation."""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Convert to PyTorch Geometric format
        from torch_geometric.data import Data
        from torch_geometric.utils import from_networkx
        import networkx as nx
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (atoms)
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), 
                      atomic_num=atom.GetAtomicNum(),
                      formal_charge=atom.GetFormalCharge(),
                      hybridization=atom.GetHybridization(),
                      is_aromatic=atom.GetIsAromatic(),
                      total_h=atom.GetTotalNumHs())
        
        # Add edges (bonds)
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                      bond_type=bond.GetBondType())
        
        # Convert to PyTorch Geometric
        data = from_networkx(G)
        
        # Add node features (simplified)
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetDegree(),
                atom.GetTotalValence(),
                atom.GetNumRadicalElectrons(),
                atom.GetIsotope()
            ]
            node_features.append(features)
        
        data.x = torch.tensor(node_features, dtype=torch.float)
        
        # Add edge features
        edge_features = []
        for bond in mol.GetBonds():
            features = [
                int(bond.GetBondType()),
                int(bond.GetIsAromatic()),
                int(bond.GetIsConjugated())
            ]
            edge_features.append(features)
            edge_features.append(features)  # Undirected graph
        
        data.edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return {
            'data': data,
            'mol': mol,
            'smiles': smiles
        }
    
    except Exception as e:
        st.error(f"Error processing SMILES: {e}")
        return None


def predict_property(model: torch.nn.Module, graph_data: Dict, config: Dict) -> float:
    """Predict molecular property."""
    data = graph_data['data']
    data = data.to(get_device())
    
    with torch.no_grad():
        if config['model']['name'] == 'MPNN' and hasattr(data, 'edge_attr'):
            pred = model(data.x, data.edge_index, data.edge_attr, torch.zeros(data.num_nodes, dtype=torch.long, device=get_device()))
        else:
            pred = model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long, device=get_device()))
    
    return pred.item()


def visualize_molecule(mol, width: int = 400, height: int = 300) -> str:
    """Visualize molecule using RDKit."""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Convert to base64 for display
        import base64
        from io import BytesIO
        
        img_data = drawer.GetDrawingText()
        img_buffer = BytesIO(img_data)
        
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    except Exception as e:
        st.error(f"Error visualizing molecule: {e}")
        return None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Drug Discovery with GNNs",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Drug Discovery with Graph Neural Networks")
    st.markdown("Predict molecular properties using state-of-the-art Graph Neural Networks")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Check if models exist
    models_dir = Path("./assets/models")
    available_models = []
    
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                checkpoint_path = model_dir / "checkpoint_best.pt"
                if checkpoint_path.exists():
                    available_models.append(model_dir.name)
    
    if not available_models:
        st.error("No trained models found. Please train a model first using the training script.")
        st.stop()
    
    selected_model = st.sidebar.selectbox("Select Model", available_models)
    
    # Load selected model
    model_path = f"./assets/models/{selected_model}/checkpoint_best.pt"
    config_path = f"./configs/{selected_model.split('_')[0]}_{selected_model.split('_')[1]}.yaml"
    
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        st.stop()
    
    with st.spinner("Loading model..."):
        model, config = load_model(model_path, config_path)
    
    st.sidebar.success(f"Loaded {config['model']['name']} model")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Molecular Input")
        
        # SMILES input
        smiles = st.text_input(
            "Enter SMILES string",
            value="CCO",  # Ethanol as default
            help="Enter a valid SMILES string representing a molecule"
        )
        
        # Example SMILES
        st.markdown("**Example SMILES:**")
        example_smiles = {
            "Ethanol": "CCO",
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Benzene": "C1=CC=CC=C1"
        }
        
        for name, example in example_smiles.items():
            if st.button(f"{name}: {example}", key=f"example_{name}"):
                st.session_state.smiles = example
                st.rerun()
        
        if 'smiles' in st.session_state:
            smiles = st.session_state.smiles
    
    with col2:
        st.header("Prediction Results")
        
        if smiles:
            # Process SMILES
            with st.spinner("Processing molecule..."):
                graph_data = smiles_to_graph(smiles)
            
            if graph_data is not None:
                # Display molecule
                st.subheader("Molecular Structure")
                
                if RDKIT_AVAILABLE:
                    img_base64 = visualize_molecule(graph_data['mol'])
                    if img_base64:
                        st.image(f"data:image/png;base64,{img_base64}", width=300)
                
                # Predict property
                with st.spinner("Making prediction..."):
                    prediction = predict_property(model, graph_data, config)
                
                # Display prediction
                st.subheader("Property Prediction")
                
                # Get dataset-specific property name
                dataset_name = config['data']['dataset']
                property_names = {
                    'ESOL': 'Aqueous Solubility (log mol/L)',
                    'QM9': 'Molecular Property',
                    'ZINC': 'Drug-likeness Score'
                }
                
                property_name = property_names.get(dataset_name, 'Molecular Property')
                
                st.metric(
                    label=property_name,
                    value=f"{prediction:.4f}",
                    delta=None
                )
                
                # Molecular descriptors
                st.subheader("Molecular Descriptors")
                descriptors = compute_molecular_descriptors(smiles)
                
                if descriptors:
                    desc_df = pd.DataFrame(list(descriptors.items()), columns=['Descriptor', 'Value'])
                    
                    # Display key descriptors
                    key_descriptors = ['mol_weight', 'logp', 'num_atoms', 'num_rings', 'tpsa']
                    key_desc_df = desc_df[desc_df['Descriptor'].isin(key_descriptors)]
                    
                    if not key_desc_df.empty:
                        st.dataframe(key_desc_df.set_index('Descriptor'), use_container_width=True)
                
                # Property interpretation
                st.subheader("Interpretation")
                if dataset_name == 'ESOL':
                    if prediction > -2:
                        st.success("High solubility - Good for drug formulation")
                    elif prediction > -4:
                        st.info("Moderate solubility - May need formulation optimization")
                    else:
                        st.warning("Low solubility - Consider structural modifications")
                
            else:
                st.error("Invalid SMILES string. Please enter a valid molecular structure.")
    
    # Additional features
    st.header("Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", config['model']['name'])
        st.metric("Hidden Dimension", config['model']['hidden_dim'])
    
    with col2:
        st.metric("Number of Layers", config['model']['num_layers'])
        st.metric("Dropout Rate", config['model']['dropout'])
    
    with col3:
        st.metric("Dataset", config['data']['dataset'])
        st.metric("Batch Size", config['data']['batch_size'])
    
    # Model architecture visualization
    st.header("Model Architecture")
    
    architecture_info = {
        'Input Layer': f"{config['model']['input_dim']} features",
        'Hidden Layers': f"{config['model']['num_layers']} x {config['model']['hidden_dim']}",
        'Output Layer': f"{config['model']['output_dim']} property",
        'Pooling': config['model']['pooling'],
        'Activation': 'ReLU',
        'Normalization': 'BatchNorm' if config['model']['use_batch_norm'] else 'None'
    }
    
    arch_df = pd.DataFrame(list(architecture_info.items()), columns=['Component', 'Specification'])
    st.dataframe(arch_df.set_index('Component'), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with PyTorch Geometric, RDKit, and Streamlit")


if __name__ == "__main__":
    main()
