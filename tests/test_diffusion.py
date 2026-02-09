"""
Unit tests for diffusion model components.
"""
import pytest
import torch
import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config, ModelConfig, DiffusionConfig, ATOM_TYPES, BOND_TYPES
from model.noise_scheduler import NoiseScheduler
from model.graph_transformer import GraphTransformer
from model.diffusion import MolecularDiffusion
from data.molecular_graph import smiles_to_graph


class TestNoiseScheduler:
    """Tests for noise scheduler."""
    
    @pytest.fixture
    def scheduler(self):
        config = DiffusionConfig(num_timesteps=100)
        return NoiseScheduler(config, device='cpu')
    
    def test_beta_schedule_shape(self, scheduler):
        """Test beta schedule has correct shape."""
        assert scheduler.betas.shape == (100,)
        assert scheduler.alphas.shape == (100,)
        assert scheduler.alphas_cumprod.shape == (100,)
    
    def test_beta_values_valid(self, scheduler):
        """Test beta values are in valid range."""
        assert (scheduler.betas >= 0).all()
        assert (scheduler.betas <= 1).all()
    
    def test_add_noise(self, scheduler):
        """Test noise addition to categorical data."""
        x = torch.tensor([0, 1, 2, 0, 1])  # Original categories
        t = torch.tensor([50])  # Timestep
        
        noisy_x = scheduler.add_noise(x, t, num_classes=len(ATOM_TYPES))
        
        assert noisy_x.shape == x.shape
        assert (noisy_x >= 0).all()
        assert (noisy_x < len(ATOM_TYPES)).all()
    
    def test_sample_timesteps(self, scheduler):
        """Test timestep sampling."""
        t = scheduler.sample_timesteps(batch_size=10)
        
        assert t.shape == (10,)
        assert (t >= 0).all()
        assert (t < scheduler.num_timesteps).all()


class TestGraphTransformer:
    """Tests for graph transformer."""
    
    @pytest.fixture
    def model(self):
        config = ModelConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )
        return GraphTransformer(config)
    
    def test_forward_pass(self, model):
        """Test forward pass produces correct output shapes."""
        # Create dummy input
        N = 10  # nodes
        E = 20  # edges
        
        node_input_dim = len(ATOM_TYPES) + 5 + 2  # atoms + charges + flags
        edge_input_dim = len(BOND_TYPES) + 2  # bonds + flags
        
        x = torch.randn(N, node_input_dim)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, edge_input_dim)
        t = torch.tensor([50])
        batch = torch.zeros(N, dtype=torch.long)
        
        output = model(x, edge_index, edge_attr, t, batch)
        
        assert 'node_logits' in output
        assert 'charge_logits' in output
        assert 'edge_logits' in output
        
        assert output['node_logits'].shape == (N, len(ATOM_TYPES))
        assert output['charge_logits'].shape == (N, 5)
        assert output['edge_logits'].shape == (E, len(BOND_TYPES))


class TestMolecularDiffusion:
    """Tests for full diffusion model."""
    
    @pytest.fixture
    def model(self):
        config = Config()
        config.model.hidden_dim = 64
        config.model.num_layers = 2
        config.model.num_heads = 4
        config.diffusion.num_timesteps = 50
        config.device = 'cpu'
        return MolecularDiffusion(config)
    
    def test_training_step(self, model):
        """Test training step produces loss."""
        # Create batch from real molecules
        graphs = [smiles_to_graph('CCO'), smiles_to_graph('c1ccccc1')]
        
        # Filter out None graphs
        graphs = [g for g in graphs if g is not None]
        
        if len(graphs) < 2:
            pytest.skip("Could not create test graphs")
        
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(graphs)
        
        metrics = model.training_step(batch)
        
        assert 'loss' in metrics
        assert metrics['loss'].requires_grad
        assert not torch.isnan(metrics['loss'])
    
    def test_sample(self, model):
        """Test molecule sampling."""
        model.eval()
        molecules = model.sample(num_molecules=2, num_atoms=10)
        
        assert len(molecules) == 2
        assert all(hasattr(m, 'node_types') for m in molecules)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
