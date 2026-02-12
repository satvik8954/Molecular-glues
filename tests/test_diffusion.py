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

from config import Config, ModelConfig, DiffusionConfig, ATOM_TYPES, BOND_TYPES, CHARGES, HYBRIDIZATIONS
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
        assert scheduler.betas.shape == (100,)
        assert scheduler.alphas.shape == (100,)
        assert scheduler.alphas_cumprod.shape == (100,)

    def test_beta_values_valid(self, scheduler):
        assert (scheduler.betas >= 0).all()
        assert (scheduler.betas <= 1).all()

    def test_add_noise(self, scheduler):
        x = torch.tensor([0, 1, 2, 0, 1])
        t = torch.tensor([50])
        noisy_x = scheduler.add_noise(x, t, num_classes=len(ATOM_TYPES))
        assert noisy_x.shape == x.shape
        assert (noisy_x >= 0).all()
        assert (noisy_x < len(ATOM_TYPES)).all()

    def test_sample_timesteps(self, scheduler):
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
        N = 10  # nodes
        E = 20  # edges

        # Updated input dims: 23 node, 8 edge
        node_input_dim = len(ATOM_TYPES) + len(CHARGES) + len(HYBRIDIZATIONS) + 4  # 23
        edge_input_dim = len(BOND_TYPES) + 3  # 8

        x = torch.randn(N, node_input_dim)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, edge_input_dim)
        t = torch.tensor([50])
        batch = torch.zeros(N, dtype=torch.long)

        output = model(x, edge_index, edge_attr, t, batch)

        assert 'node_logits' in output
        assert 'charge_logits' in output
        assert 'edge_logits' in output
        assert 'graph_features' in output

        assert output['node_logits'].shape == (N, len(ATOM_TYPES))
        assert output['charge_logits'].shape == (N, len(CHARGES))
        assert output['edge_logits'].shape == (E, len(BOND_TYPES))

    def test_forward_with_conditioning(self, model):
        """Test forward pass with property conditioning."""
        N = 10
        E = 20

        node_input_dim = len(ATOM_TYPES) + len(CHARGES) + len(HYBRIDIZATIONS) + 4
        edge_input_dim = len(BOND_TYPES) + 3

        x = torch.randn(N, node_input_dim)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, edge_input_dim)
        t = torch.tensor([50])
        batch = torch.zeros(N, dtype=torch.long)
        condition = torch.randn(1, 8)  # 8 properties

        output = model(x, edge_index, edge_attr, t, batch, condition=condition)

        assert 'node_logits' in output
        assert 'graph_features' in output
        assert output['graph_features'].shape == (1, 8)


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
        """Test training step produces loss with auxiliary losses."""
        graphs = [smiles_to_graph('CCO'), smiles_to_graph('c1ccccc1')]
        graphs = [g for g in graphs if g is not None]

        if len(graphs) < 2:
            pytest.skip("Could not create test graphs")

        from torch_geometric.data import Batch
        batch = Batch.from_data_list(graphs)

        metrics = model.training_step(batch)

        assert 'loss' in metrics
        assert 'valency_loss' in metrics
        assert 'property_loss' in metrics
        assert metrics['loss'].requires_grad
        assert not torch.isnan(metrics['loss'])
        assert metrics['valency_loss'] >= 0

    def test_sample(self, model):
        """Test molecule sampling."""
        model.eval()
        molecules = model.sample(num_molecules=2, num_atoms=10)

        assert len(molecules) == 2
        assert all(hasattr(m, 'node_types') for m in molecules)

    def test_guided_sample(self, model):
        """Test guided molecule sampling with target properties."""
        model.eval()
        target_props = torch.tensor([0.75, 0.5, 0.5, 0.4, 0.4, 0.35, 0.5, 0.3])

        molecules = model.sample(
            num_molecules=2,
            num_atoms=10,
            target_properties=target_props,
            guidance_scale=2.0,
        )

        assert len(molecules) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
