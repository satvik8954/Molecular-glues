# model/classifier.py
"""
Binary classifier for molecular glue detection.

Built from existing Graph Transformer components:
- GraphAttentionLayer (multi-head attention with edge updates)
- RingAttentionLayer (ring-centric attention)
- GlobalGraphPool (gated global pooling)

New: Classification head (3 linear layers → 1 logit)

Anti-overfitting features:
- DropEdge: randomly removes edges during training
- Reduced FFN expansion (2× instead of 4×)
- Higher dropout throughout (configurable, default 0.3)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Reuse existing components
from model.graph_transformer import (
    GraphAttentionLayer,
    RingAttentionLayer,
    GlobalGraphPool
)
from config import ATOM_TYPES, BOND_TYPES, CHARGES, HYBRIDIZATIONS


def drop_edge(edge_index, edge_attr, drop_rate=0.15, training=True):
    """
    DropEdge: randomly remove edges during training to prevent
    over-reliance on specific graph topology.
    
    Args:
        edge_index: [2, E]
        edge_attr: [E, D]
        drop_rate: fraction of edges to drop
        training: only drop during training
    
    Returns:
        edge_index, edge_attr with some edges removed
    """
    if not training or drop_rate <= 0:
        return edge_index, edge_attr
    
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
    
    # Ensure at least 1 edge per graph
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    
    return edge_index[:, keep_mask], edge_attr[keep_mask]


class ClassifierTransformerBlock(nn.Module):
    """
    Simplified version of MultiScaleBlock for classification.
    
    Removed vs MultiScaleBlock:
    - No FiLM conditioning (no property targets)
    - No time embedding (no diffusion timesteps)
    
    Kept:
    - GraphAttentionLayer (local attention)
    - RingAttentionLayer (ring-centric attention)
    - GlobalGraphPool (global context)
    - Feed-forward network
    - Pre-norm + residual connections
    
    Anti-overfitting changes:
    - FFN expansion reduced from 4× to 2× hidden_dim
    - Dropout applied throughout
    """
    
    def __init__(self, hidden_dim, num_heads, dropout, edge_dim=None):
        super().__init__()
        
        actual_edge_dim = edge_dim or hidden_dim
        
        # Reuse existing layers
        self.local_attn = GraphAttentionLayer(
            hidden_dim, num_heads, dropout, edge_dim=actual_edge_dim
        )
        self.ring_attn = RingAttentionLayer(
            hidden_dim, num_heads=max(1, num_heads // 2), dropout=dropout
        )
        self.global_pool = GlobalGraphPool(hidden_dim, dropout)
        
        # Feed-forward network — REDUCED from 4× to 2× expansion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Pre-norm layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(actual_edge_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
    
    def forward(self, h, edge_index, edge_attr, batch, in_ring_flags=None):
        """
        Args:
            h: Node embeddings [N, hidden_dim]
            edge_index: [2, E]
            edge_attr: Edge embeddings [E, edge_dim]
            batch: Batch assignment [N]
            in_ring_flags: Pre-extracted in_ring flags [N] or full node features [N, F]
            
        Returns:
            h: Updated node embeddings [N, hidden_dim]
            edge_attr: Updated edge embeddings [E, edge_dim]
        """
        # Pre-norm + local attention + residual
        h_norm = self.norm1(h)
        e_norm = self.edge_norm(edge_attr)
        h_new, e_new = self.local_attn(h_norm, edge_index, e_norm)
        h = h + h_new
        edge_attr = edge_attr + e_new
        
        # Pre-norm + ring attention + residual
        h = h + self.ring_attn(self.norm2(h), batch, in_ring_flags)
        
        # Pre-norm + global context + residual
        h = h + self.global_pool(self.norm3(h), batch)
        
        # Pre-norm + FFN + residual
        h = h + self.ffn(self.norm4(h))
        
        return h, edge_attr


class MolecularGlueClassifier(nn.Module):
    """
    Binary classifier for molecular glue detection.
    
    Architecture:
    1. Input encoding (node + edge encoders) — same structure as diffusion model
    2. N × ClassifierTransformerBlock — reuses GraphAttention/Ring/GlobalPool
    3. Global mean+max pooling → graph-level representation
    4. Classification head → single logit (use BCEWithLogitsLoss)
    
    Anti-overfitting features:
    - DropEdge during training (configurable rate)
    - Reduced FFN expansion (2× instead of 4×)
    - Higher dropout in classifier head (dropout + 0.1 above backbone)
    """
    
    def __init__(self, hidden_dim=128, num_layers=3, num_heads=8, dropout=0.3,
                 drop_edge_rate=0.15):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.drop_edge_rate = drop_edge_rate
        
        # Input dimensions (from config.py vocabularies)
        # Node: atom_types(10) + charges(5) + hybridizations(4) + aromatic(1) + in_ring(1) + num_hs(1) + conjugated(1) = 23
        node_input_dim = len(ATOM_TYPES) + len(CHARGES) + len(HYBRIDIZATIONS) + 4
        # Edge: bond_types(5) + aromatic(1) + conjugated(1) + in_ring(1) = 8
        edge_input_dim = len(BOND_TYPES) + 3
        
        # Input encoders (same structure as GraphTransformer)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Edge encoder projects to hidden_dim (matches GraphAttentionLayer's default edge_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer blocks (reuse existing attention layers)
        self.layers = nn.ModuleList([
            ClassifierTransformerBlock(
                hidden_dim, num_heads, dropout, edge_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head (the only truly new component)
        # mean+max pooling gives 2*hidden_dim features
        # Uses higher dropout (dropout + 0.1) since this is where most memorization happens
        head_dropout = min(dropout + 0.1, 0.5)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            
            nn.Linear(hidden_dim // 2, 1)  # Single logit for binary classification
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass for classification.
        
        Args:
            x: Node features [N, 23]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 8]
            batch: Batch assignment [N]
            
        Returns:
            logits: Classification logits [B, 1]
        """
        # Extract in_ring flags before encoding (avoids cloning full tensor)
        in_ring_flags = x[:, 20]  # in_ring feature at index 20
        
        # Encode inputs
        h = self.node_encoder(x)        # [N, hidden_dim]
        e = self.edge_encoder(edge_attr) # [E, hidden_dim]
        
        # DropEdge: randomly remove edges during training
        edge_index_used, e_used = drop_edge(
            edge_index, e, self.drop_edge_rate, self.training
        )
        
        # Transformer blocks (edge features updated through layers)
        for layer in self.layers:
            h, e_used = layer(h, edge_index_used, e_used, batch, in_ring_flags)
        
        # Final normalization
        h = self.final_norm(h)
        
        # Global pooling: mean + max for richer graph representation
        h_mean = global_mean_pool(h, batch)  # [B, hidden_dim]
        h_max = global_max_pool(h, batch)    # [B, hidden_dim]
        h_graph = torch.cat([h_mean, h_max], dim=-1)  # [B, 2*hidden_dim]
        
        # Classify
        logits = self.classifier(h_graph)  # [B, 1]
        
        return logits