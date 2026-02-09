"""
Molecular Diffusion Model.

Combines the noise scheduler and graph transformer to create
a complete diffusion model for molecular generation.
"""
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config, ModelConfig, DiffusionConfig, ATOM_TYPES, BOND_TYPES, CHARGES
from model.noise_scheduler import NoiseScheduler
from model.graph_transformer import GraphTransformer


class MolecularDiffusion(nn.Module):
    """
    Complete diffusion model for molecular generation.
    
    Combines:
    - Discrete noise scheduler for categorical features
    - Graph transformer for denoising
    - Training and sampling procedures
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Model components
        self.transformer = GraphTransformer(config.model)
        self.noise_scheduler = NoiseScheduler(config.diffusion, config.device)
        
        # Number of classes
        self.num_atom_types = len(ATOM_TYPES)
        self.num_bond_types = len(BOND_TYPES)
        self.num_charges = len(CHARGES)
    
    def training_step(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Perform one training step.
        
        Args:
            batch: Batched molecular graphs
            
        Returns:
            Dictionary containing loss and metrics
        """
        # Sample random timesteps
        t = self.noise_scheduler.sample_timesteps(batch.num_graphs)
        
        # Get original node and edge types
        node_types_orig = batch.node_types
        edge_types_orig = batch.edge_types
        
        # Get original charges if available
        if hasattr(batch, 'node_charges'):
            charges_orig = batch.node_charges
        else:
            charges_orig = torch.zeros_like(node_types_orig) + 2  # Default to neutral (idx 2)
        
        # Expand timesteps to match nodes/edges
        node_t = t[batch.batch]
        edge_batch = batch.batch[batch.edge_index[0]]
        edge_t = t[edge_batch]
        
        # Add noise to node types and edge types
        node_types_noisy = self.noise_scheduler.add_noise(
            node_types_orig, node_t, self.num_atom_types
        )
        edge_types_noisy = self.noise_scheduler.add_noise(
            edge_types_orig, edge_t, self.num_bond_types
        )
        charges_noisy = self.noise_scheduler.add_noise(
            charges_orig, node_t, self.num_charges
        )
        
        # Create noisy input features
        x_noisy = self._create_node_features(
            node_types_noisy, charges_noisy, batch
        )
        edge_attr_noisy = self._create_edge_features(
            edge_types_noisy, batch
        )
        
        # Forward pass through transformer
        predictions = self.transformer(
            x_noisy,
            batch.edge_index,
            edge_attr_noisy,
            t,
            batch.batch
        )
        
        # Compute losses
        node_loss = F.cross_entropy(
            predictions['node_logits'],
            node_types_orig.long()
        )
        charge_loss = F.cross_entropy(
            predictions['charge_logits'],
            charges_orig.long()
        )
        edge_loss = F.cross_entropy(
            predictions['edge_logits'],
            edge_types_orig.long()
        )
        
        # Total loss (weighted)
        total_loss = node_loss + 0.5 * charge_loss + 0.5 * edge_loss
        
        # Compute accuracy metrics
        with torch.no_grad():
            node_acc = (predictions['node_logits'].argmax(dim=-1) == node_types_orig).float().mean()
            edge_acc = (predictions['edge_logits'].argmax(dim=-1) == edge_types_orig).float().mean()
        
        return {
            'loss': total_loss,
            'node_loss': node_loss,
            'charge_loss': charge_loss,
            'edge_loss': edge_loss,
            'node_acc': node_acc,
            'edge_acc': edge_acc,
        }
    
    def _create_node_features(
        self,
        node_types: torch.Tensor,
        charges: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """Create node feature tensor from categorical types."""
        N = node_types.shape[0]
        device = node_types.device
        
        # One-hot encode atom types
        node_type_onehot = torch.zeros(N, self.num_atom_types, device=device)
        node_type_onehot.scatter_(1, node_types.unsqueeze(1), 1)
        
        # One-hot encode charges
        charge_onehot = torch.zeros(N, self.num_charges, device=device)
        charge_onehot.scatter_(1, charges.unsqueeze(1), 1)
        
        # Get aromatic and ring flags from original batch
        aromatic = batch.x[:, self.num_atom_types + self.num_charges].unsqueeze(1)
        in_ring = batch.x[:, self.num_atom_types + self.num_charges + 1].unsqueeze(1)
        
        return torch.cat([node_type_onehot, charge_onehot, aromatic, in_ring], dim=1)
    
    def _create_edge_features(
        self,
        edge_types: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """Create edge feature tensor from categorical types."""
        E = edge_types.shape[0]
        device = edge_types.device
        
        # One-hot encode bond types
        edge_type_onehot = torch.zeros(E, self.num_bond_types, device=device)
        edge_type_onehot.scatter_(1, edge_types.unsqueeze(1), 1)
        
        # Get aromatic and ring flags from original batch
        aromatic = batch.edge_attr[:, self.num_bond_types].unsqueeze(1)
        in_ring = batch.edge_attr[:, self.num_bond_types + 1].unsqueeze(1)
        
        return torch.cat([edge_type_onehot, aromatic, in_ring], dim=1)
    
    @torch.no_grad()
    def sample(
        self,
        num_molecules: int,
        num_atoms: int = 20,
        temperature: float = 1.0,
    ) -> List[Data]:
        """
        Sample new molecules from the model.
        
        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Number of atoms per molecule
            temperature: Sampling temperature
            
        Returns:
            List of generated molecular graphs
        """
        self.eval()
        device = next(self.parameters()).device
        
        generated = []
        
        for _ in range(num_molecules):
            # Initialize with random noise (uniform over categories)
            node_types = torch.randint(0, self.num_atom_types, (num_atoms,), device=device)
            charges = torch.full((num_atoms,), 2, device=device)  # Neutral
            
            # Create fully connected graph for initial structure
            edge_index = self._create_full_edge_index(num_atoms, device)
            edge_types = torch.randint(0, self.num_bond_types, (edge_index.shape[1],), device=device)
            
            # Batch information
            batch_idx = torch.zeros(num_atoms, dtype=torch.long, device=device)
            
            # Reverse diffusion
            for t in reversed(range(self.noise_scheduler.num_timesteps)):
                t_tensor = torch.tensor([t], device=device)
                
                # Create features
                x = self._create_node_features_simple(node_types, charges, num_atoms, device)
                edge_attr = self._create_edge_features_simple(edge_types, device)
                
                # Predict clean data
                predictions = self.transformer(
                    x, edge_index, edge_attr, t_tensor, batch_idx
                )
                
                if t > 0:
                    # Sample from predicted distribution with temperature
                    node_probs = F.softmax(predictions['node_logits'] / temperature, dim=-1)
                    edge_probs = F.softmax(predictions['edge_logits'] / temperature, dim=-1)
                    charge_probs = F.softmax(predictions['charge_logits'] / temperature, dim=-1)
                    
                    node_types = torch.multinomial(node_probs, 1).squeeze(-1)
                    edge_types = torch.multinomial(edge_probs, 1).squeeze(-1)
                    charges = torch.multinomial(charge_probs, 1).squeeze(-1)
                else:
                    # Final step: take argmax
                    node_types = predictions['node_logits'].argmax(dim=-1)
                    edge_types = predictions['edge_logits'].argmax(dim=-1)
                    charges = predictions['charge_logits'].argmax(dim=-1)
            
            # Create output graph
            # Filter edges: only keep non-NONE bonds
            valid_edges = edge_types > 0  # NONE is index 0
            edge_index_filtered = edge_index[:, valid_edges]
            edge_types_filtered = edge_types[valid_edges]
            
            graph = Data(
                x=F.one_hot(node_types, self.num_atom_types).float(),
                edge_index=edge_index_filtered,
                edge_attr=F.one_hot(edge_types_filtered, self.num_bond_types).float(),
                node_types=node_types,
                edge_types=edge_types_filtered,
                node_charges=charges,
            )
            
            generated.append(graph)
        
        return generated
    
    def _create_full_edge_index(self, num_atoms: int, device: torch.device) -> torch.Tensor:
        """Create fully connected edge index."""
        # Create upper triangular adjacency (no self-loops)
        src = []
        dst = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                src.extend([i, j])
                dst.extend([j, i])
        
        return torch.tensor([src, dst], dtype=torch.long, device=device)
    
    def _create_node_features_simple(
        self,
        node_types: torch.Tensor,
        charges: torch.Tensor,
        num_atoms: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create simple node features without batch reference."""
        # One-hot encode
        node_type_onehot = F.one_hot(node_types, self.num_atom_types).float()
        charge_onehot = F.one_hot(charges, self.num_charges).float()
        
        # Add placeholder aromatic and ring flags
        aromatic = torch.zeros(num_atoms, 1, device=device)
        in_ring = torch.zeros(num_atoms, 1, device=device)
        
        return torch.cat([node_type_onehot, charge_onehot, aromatic, in_ring], dim=1)
    
    def _create_edge_features_simple(
        self,
        edge_types: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Create simple edge features without batch reference."""
        E = edge_types.shape[0]
        
        edge_type_onehot = F.one_hot(edge_types, self.num_bond_types).float()
        aromatic = torch.zeros(E, 1, device=device)
        in_ring = torch.zeros(E, 1, device=device)
        
        return torch.cat([edge_type_onehot, aromatic, in_ring], dim=1)
    
    def to(self, device):
        """Move model to device and update noise scheduler."""
        super().to(device)
        self.device = device
        self.noise_scheduler = NoiseScheduler(self.config.diffusion, device)
        return self
