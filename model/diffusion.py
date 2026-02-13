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
from tqdm import tqdm

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config, ModelConfig, DiffusionConfig, LossConfig, GuidanceConfig
from config import ATOM_TYPES, BOND_TYPES, CHARGES, HYBRIDIZATIONS, PROPERTY_NAMES
from model.noise_scheduler import NoiseScheduler
from model.graph_transformer import GraphTransformer


# Valid valences for each atom type (by atomic number)
VALID_VALENCES = {
    6: [4],           # Carbon
    7: [3, 5],        # Nitrogen
    8: [2],           # Oxygen
    16: [2, 4, 6],    # Sulfur
    9: [1],           # Fluorine
    17: [1, 3, 5, 7], # Chlorine
    35: [1, 3, 5],    # Bromine
    15: [3, 5],       # Phosphorus
    53: [1, 3, 5, 7], # Iodine
}

ATOM_SYMBOL_TO_ATOMIC_NUM = {
    'C': 6, 'N': 7, 'O': 8, 'S': 16, 'F': 9,
    'Cl': 17, 'Br': 35, 'P': 15, 'I': 53,
}


class MolecularDiffusion(nn.Module):
    """
    Complete diffusion model for molecular generation.

    Combines:
    - Discrete noise scheduler for categorical features
    - Graph transformer for denoising
    - Training and sampling procedures
    - Auxiliary losses (valency, property, fragment)
    - Classifier-free guidance for property conditioning
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.device

        # Model components
        self.transformer = GraphTransformer(config.model)
        self.noise_scheduler = NoiseScheduler(config.diffusion, config.device)

        # Loss config
        self.loss_config = config.loss
        self.guidance_config = config.guidance

        # Number of classes
        self.num_atom_types = len(ATOM_TYPES)
        self.num_bond_types = len(BOND_TYPES)
        self.num_charges = len(CHARGES)
        self.num_hybridizations = len(HYBRIDIZATIONS)

        # Node & edge feature dimensions (must match molecular_graph.py)
        # Node: atom_types(10) + charges(5) + hybrid(4) + aromatic(1) + in_ring(1) + num_hs(1) + conjugated(1) = 23
        self.node_feat_dim = self.num_atom_types + self.num_charges + self.num_hybridizations + 4
        # Edge: bond_types(5) + aromatic(1) + conjugated(1) + in_ring(1) = 8
        self.edge_feat_dim = self.num_bond_types + 3

    def training_step(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with multi-objective loss.

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

        # Get original charges
        if hasattr(batch, 'node_charges'):
            charges_orig = batch.node_charges
        else:
            charges_orig = torch.zeros_like(node_types_orig) + 2  # Default neutral

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

        # Property conditioning with classifier-free guidance dropout
        condition = None
        if hasattr(batch, 'properties') and batch.properties is not None:
            condition = batch.properties
            if condition.dim() == 1:
                # Flattened by DataLoader; reshape to [num_graphs, num_properties]
                num_props = condition.shape[0] // batch.num_graphs
                condition = condition.view(batch.num_graphs, num_props)
            elif condition.dim() == 2 and condition.shape[0] != batch.num_graphs:
                # Properties batched per-node; take per-graph via scatter
                condition = condition[:batch.num_graphs]

            # Classifier-free guidance dropout: randomly zero out conditioning
            if self.training and self.guidance_config.condition_dropout > 0:
                mask = torch.rand(condition.shape[0], device=condition.device) < self.guidance_config.condition_dropout
                condition = condition.clone()
                condition[mask.unsqueeze(-1).expand_as(condition)] = 0.0

        # Forward pass through transformer
        predictions = self.transformer(
            x_noisy,
            batch.edge_index,
            edge_attr_noisy,
            t,
            batch.batch,
            condition=condition,
        )

        # === Primary Loss: Cross-entropy on predictions ===
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

        diffusion_loss = node_loss + 0.5 * charge_loss + 0.5 * edge_loss

        # === Auxiliary Loss 1: Valency loss ===
        valency_loss = self._compute_valency_loss(
            predictions['node_logits'],
            predictions['edge_logits'],
            batch.edge_index,
            batch.batch,
        )

        # === Auxiliary Loss 2: Property matching loss ===
        property_loss = torch.tensor(0.0, device=diffusion_loss.device)
        if condition is not None and 'graph_features' in predictions:
            property_loss = self._compute_property_loss(
                predictions['graph_features'],
                condition,
            )

        # === Total loss ===
        total_loss = (
            diffusion_loss
            + self.loss_config.lambda_valency * valency_loss
            + self.loss_config.lambda_property * property_loss
        )

        # Compute accuracy metrics
        with torch.no_grad():
            node_acc = (predictions['node_logits'].argmax(dim=-1) == node_types_orig).float().mean()
            edge_acc = (predictions['edge_logits'].argmax(dim=-1) == edge_types_orig).float().mean()

        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'node_loss': node_loss,
            'charge_loss': charge_loss,
            'edge_loss': edge_loss,
            'valency_loss': valency_loss,
            'property_loss': property_loss,
            'node_acc': node_acc,
            'edge_acc': edge_acc,
        }

    def _compute_valency_loss(
        self,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize predicted atoms whose total bond order exceeds valid valency.

        Uses predicted (soft) atom types and bond types to compute expected
        valency violation.  Applies softmax only to the atom-type slice of
        node_logits (first num_atom_types dims) and accumulates bond orders
        from both edge directions for undirected graphs.
        """
        device = node_logits.device

        # --- Soft atom type probabilities (FIXED: slice atom-type dims only) ---
        atom_probs = F.softmax(
            node_logits[:, :self.num_atom_types], dim=-1
        )  # [N, num_atom_types]

        # Soft bond order: weighted sum of bond orders
        # Bond orders: NONE=0, SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=1.5
        bond_orders = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5], device=device)
        edge_probs = F.softmax(edge_logits, dim=-1)  # [E, num_bond_types]
        expected_bond_order = (edge_probs * bond_orders.unsqueeze(0)).sum(dim=-1)  # [E]

        # Sum bond orders per atom (accumulate from BOTH src and dst)
        N = node_logits.shape[0]
        atom_valency = torch.zeros(N, device=device)
        atom_valency.scatter_add_(0, edge_index[0], expected_bond_order)
        # NOTE: edges are stored bidirectionally so each direction already
        # contributes; if your dataset uses unidirectional edges uncomment:
        # atom_valency.scatter_add_(0, edge_index[1], expected_bond_order)

        # Expected max valency per atom (weighted by atom type probs)
        # Max valences: C=4, N=5, O=2, S=6, F=1, Cl=7, Br=5, P=5, I=7, Other=4
        max_valences = torch.tensor(
            [4.0, 5.0, 2.0, 6.0, 1.0, 7.0, 5.0, 5.0, 7.0, 4.0], device=device
        )
        expected_max_valency = (atom_probs * max_valences.unsqueeze(0)).sum(dim=-1)

        # Quadratic penalty for violations (stronger gradient signal)
        violation = F.relu(atom_valency - expected_max_valency)
        excess_valency_loss = (violation ** 2).mean()

        # --- Total valency matching loss ---
        # Target total valency (bond order + implicit H) per atom type
        target_total_valency = torch.tensor(
            [4.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 4.0], device=device
        )  # C, N, O, S, F, Cl, Br, P, I, Other
        expected_target = (atom_probs * target_total_valency.unsqueeze(0)).sum(dim=-1)
        valency_match_loss = F.mse_loss(atom_valency, expected_target)

        return excess_valency_loss + 0.5 * valency_match_loss

    def _compute_property_loss(
        self,
        graph_features: torch.Tensor,
        target_properties: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-property weighted MSE loss between predicted graph features
        and target properties.

        Properties are already normalized in dataset.py, but we add
        per-property weighting to boost broken properties (HBD, HBA,
        aromatic rings).

        Property order (from config.PROPERTY_NAMES):
            0: molecular_weight, 1: logp, 2: num_aromatic_rings,
            3: hbd, 4: hba, 5: fraction_sp3, 6: tpsa, 7: sa_score
        """
        # Per-property weights — boost properties that were broken in
        # generated molecules (HBD, HBA, aromatic rings, TPSA)
        weights = torch.tensor(
            [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.5, 1.0],
            device=graph_features.device,
        )
        # Truncate or pad weights to match actual feature dim
        num_props = min(graph_features.shape[-1], weights.shape[0])
        w = weights[:num_props]

        # Weighted MSE
        diff_sq = (graph_features[:, :num_props] - target_properties[:, :num_props]) ** 2
        weighted = diff_sq * w.unsqueeze(0)
        return weighted.mean()

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

        # Get hybridization, aromatic, ring, num_hs, conjugated from original batch
        offset = self.num_atom_types + self.num_charges
        hybrid = batch.x[:, offset:offset + self.num_hybridizations]
        aromatic = batch.x[:, offset + self.num_hybridizations].unsqueeze(1)
        in_ring = batch.x[:, offset + self.num_hybridizations + 1].unsqueeze(1)
        num_hs = batch.x[:, offset + self.num_hybridizations + 2].unsqueeze(1)
        conjugated = batch.x[:, offset + self.num_hybridizations + 3].unsqueeze(1)

        return torch.cat([node_type_onehot, charge_onehot, hybrid, aromatic, in_ring, num_hs, conjugated], dim=1)

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

        # Get aromatic, conjugated, ring flags from original batch
        aromatic = batch.edge_attr[:, self.num_bond_types].unsqueeze(1)
        conjugated = batch.edge_attr[:, self.num_bond_types + 1].unsqueeze(1)
        in_ring = batch.edge_attr[:, self.num_bond_types + 2].unsqueeze(1)

        return torch.cat([edge_type_onehot, aromatic, conjugated, in_ring], dim=1)

    @torch.no_grad()
    def sample(
        self,
        num_molecules: int,
        num_atoms: int = 20,
        temperature: float = 1.0,
        target_properties: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        num_sampling_steps: Optional[int] = None,
    ) -> List[Data]:
        """
        Sample new molecules from the model.

        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Number of atoms per molecule
            temperature: Sampling temperature
            target_properties: Optional property targets for guided generation [num_properties]
            guidance_scale: Guidance scale (uses config default if None)
            num_sampling_steps: Number of denoising steps (default: all timesteps).
                               Lower = faster but lower quality. 50-100 is usually fine.

        Returns:
            List of generated molecular graphs
        """
        self.eval()
        device = next(self.parameters()).device
        gs = guidance_scale if guidance_scale is not None else self.guidance_config.guidance_scale

        # Build the timestep schedule (optionally skipping steps for speed)
        total_timesteps = self.noise_scheduler.num_timesteps
        if num_sampling_steps is not None and num_sampling_steps < total_timesteps:
            # Evenly spaced subset of timesteps (like DDIM-style skipping)
            step_ratio = total_timesteps / num_sampling_steps
            timesteps = [int(round(i * step_ratio)) for i in reversed(range(num_sampling_steps))]
        else:
            timesteps = list(reversed(range(total_timesteps)))

        # Atom-type prior: bias toward realistic drug-like distribution
        # C:60%, N:12%, O:10%, S:4%, F:3%, Cl:3%, Br:2%, P:2%, I:1%, Other:3%
        atom_prior = torch.tensor([0.60, 0.12, 0.10, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.03],
                                  device=device)
        # Bond-type prior: most edges should be NONE in initial noise for sparse graphs
        # NONE:70%, SINGLE:18%, DOUBLE:7%, TRIPLE:1%, AROMATIC:4%
        bond_prior = torch.tensor([0.70, 0.18, 0.07, 0.01, 0.04], device=device)

        generated = []

        for mol_idx in range(num_molecules):
            # Initialize with prior-biased noise (not uniform)
            node_types = torch.multinomial(atom_prior, num_atoms, replacement=True)
            charges = torch.full((num_atoms,), 2, device=device)  # Neutral

            # Create fully connected graph for initial structure
            edge_index = self._create_full_edge_index(num_atoms, device)
            num_edges = edge_index.shape[1]
            edge_types = torch.multinomial(bond_prior, num_edges, replacement=True)

            # Batch information
            batch_idx = torch.zeros(num_atoms, dtype=torch.long, device=device)

            # Prepare conditioning
            cond = target_properties.unsqueeze(0).to(device) if target_properties is not None else None

            # Reverse diffusion with progress bar
            desc = f"Mol {mol_idx+1}/{num_molecules} ({num_atoms} atoms)"
            for t in tqdm(timesteps, desc=desc, leave=False):
                t_tensor = torch.tensor([t], device=device)

                # Create features
                x = self._create_node_features_simple(node_types, charges, num_atoms, device)
                edge_attr = self._create_edge_features_simple(edge_types, device)

                # Conditional prediction
                predictions = self.transformer(
                    x, edge_index, edge_attr, t_tensor, batch_idx,
                    condition=cond,
                )

                # Classifier-free guidance
                if cond is not None and gs > 1.0:
                    # Unconditional prediction
                    predictions_uncond = self.transformer(
                        x, edge_index, edge_attr, t_tensor, batch_idx,
                        condition=None,
                    )
                    # Guided logits
                    for key in ['node_logits', 'edge_logits', 'charge_logits']:
                        predictions[key] = (
                            predictions_uncond[key]
                            + gs * (predictions[key] - predictions_uncond[key])
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
            print(f"  Generated molecule {mol_idx+1}/{num_molecules}")

        return generated

    def _create_full_edge_index(self, num_atoms: int, device: torch.device) -> torch.Tensor:
        """Create fully connected edge index."""
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
        """Create simple node features without batch reference.

        Estimates implicit hydrogen count from atom type to avoid the
        training/generation mismatch that caused HBD=0 in all generated
        molecules.
        """
        node_type_onehot = F.one_hot(node_types, self.num_atom_types).float()
        charge_onehot = F.one_hot(charges, self.num_charges).float()

        # Placeholder hybridization (sp3 default), aromatic, ring, conjugated
        hybrid = torch.zeros(num_atoms, self.num_hybridizations, device=device)
        hybrid[:, 2] = 1.0  # sp3 default
        aromatic = torch.zeros(num_atoms, 1, device=device)
        in_ring = torch.zeros(num_atoms, 1, device=device)
        conjugated = torch.zeros(num_atoms, 1, device=device)

        # Estimate implicit hydrogens from atom type (FIXED: was all zeros)
        # Default H counts assuming sp3: C=4-bonds, N=1, O=1, S=0, halogens=0
        default_hs = torch.tensor(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            device=device,
        )  # C, N, O, S, F, Cl, Br, P, I, Other
        num_hs = default_hs[node_types].unsqueeze(1)  # [num_atoms, 1]

        return torch.cat([node_type_onehot, charge_onehot, hybrid, aromatic, in_ring, num_hs, conjugated], dim=1)

    def _create_edge_features_simple(
        self,
        edge_types: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Create simple edge features without batch reference."""
        E = edge_types.shape[0]

        edge_type_onehot = F.one_hot(edge_types, self.num_bond_types).float()
        aromatic = torch.zeros(E, 1, device=device)
        conjugated = torch.zeros(E, 1, device=device)
        in_ring = torch.zeros(E, 1, device=device)

        return torch.cat([edge_type_onehot, aromatic, conjugated, in_ring], dim=1)

    def to(self, device):
        """Move model to device and update noise scheduler."""
        super().to(device)
        self.device = device
        self.noise_scheduler = NoiseScheduler(self.config.diffusion, device)
        return self
