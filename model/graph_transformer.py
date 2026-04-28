"""
IMPROVED Graph Transformer for molecular denoising in diffusion models.

Key improvements over original:
1. Fixed edge feature propagation in attention layers
2. Added gradient checkpointing for memory efficiency
4. Better ring attention with actual ring detection
5. More stable attention mechanism with proper normalization
6. Added hybrid features to all output heads
7. Spectral normalization option for training stability
8. Better edge prediction using both node features
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from torch.utils.checkpoint import checkpoint
import math

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import ModelConfig, ATOM_TYPES, BOND_TYPES, CHARGES, HYBRIDIZATIONS, PROPERTY_NAMES




class GraphAttentionLayer(MessagePassing):
    """
    IMPROVED: Multi-head graph attention with better edge handling.
    
    Changes:
    - Edge features properly integrated into attention scores
    - Attention normalization per destination node (not global)
    - Optional spectral normalization for stability
    - Edge feature update mechanism
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, 
                 edge_dim: int = None, spectral_norm: bool = False):
        super().__init__(aggr='add', node_dim=0)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim or hidden_dim

        # Node projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge projection - IMPROVED: properly sized
        self.W_e = nn.Linear(self.edge_dim, num_heads)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature update (NEW) — with dropout + BatchNorm to prevent memorization
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim*3, self.edge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.edge_dim, self.edge_dim)
        )
        self.edge_batch_norm = nn.BatchNorm1d(self.edge_dim)

        self.dropout = nn.Dropout(dropout)
        
        # Optional spectral normalization for stability
        if spectral_norm:
            self.W_q = nn.utils.spectral_norm(self.W_q)
            self.W_k = nn.utils.spectral_norm(self.W_k)
            self.W_v = nn.utils.spectral_norm(self.W_v)

    def forward(self, x, edge_index, edge_attr):
        """
        Returns:
            x_out: Updated node features
            edge_attr_out: Updated edge features (NEW)
        """
        q = self.W_q(x).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(x).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(x).view(-1, self.num_heads, self.head_dim)

        edge_weight = self.W_e(edge_attr)  # [E, num_heads]

        # Message passing
        out = self.propagate(edge_index, q=q, k=k, v=v, 
                            edge_weight=edge_weight, x=x, edge_attr=edge_attr)
        out = out.view(-1, self.hidden_dim)
        x_out = self.W_o(out)
        
        # Update edge features (NEW) — dropout + BatchNorm for regularization
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_update = self.edge_update_mlp(edge_input)
        edge_update = self.dropout(edge_update)
        edge_attr_out = edge_attr + self.edge_batch_norm(edge_update)  # Residual + norm
        
        return x_out, edge_attr_out

    def message(self, q_i, k_j, v_j, edge_weight, index, ptr, size_i):
        """
        IMPROVED: Better attention computation with proper normalization.
        """
        # Attention scores: q·k/√d + edge_bias
        attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, num_heads]
        attn = attn + edge_weight  # Add edge bias
        
        # Softmax per destination node (proper normalization)
        attn = softmax(attn, index, ptr, size_i)
        attn = self.dropout(attn)
        
        # Apply attention to values
        return v_j * attn.unsqueeze(-1)  # [E, num_heads, head_dim]


class RingAttentionLayer(nn.Module):
    """
    IMPROVED: Ring-centric attention with actual ring detection.
    
    Changes:
    - Uses in_ring feature from node features (index 20)
    - Separate attention for ring vs non-ring atoms
    - Ring pooling and broadcasting
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, 
                node_features=None) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [N, hidden_dim]
            batch: Batch assignment [N]
            node_features: Either original node features [N, feature_dim] for ring
                          detection (index 20), or pre-extracted in_ring flags [N].
        
        Returns:
            Updated node features with ring context
        """
        N = x.shape[0]
        device = x.device

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        # Detect ring atoms — accept pre-extracted flags or full feature matrix
        if node_features is not None:
            if node_features.dim() == 1:
                # Pre-extracted in_ring flags [N]
                in_ring_mask = node_features > 0.5
            else:
                # Full feature matrix — in_ring is at index 20
                in_ring_mask = node_features[:, 20] > 0.5
        else:
            in_ring_mask = torch.ones(N, dtype=torch.bool, device=device)
        
        num_graphs = batch.max().item() + 1

        # ---- Vectorized ring pooling (replaces Python for-loop) ----
        ring_k = torch.zeros(num_graphs, self.num_heads, self.head_dim, device=device, dtype=k.dtype)
        ring_v = torch.zeros(num_graphs, self.num_heads, self.head_dim, device=device, dtype=v.dtype)

        if in_ring_mask.any():
            ring_batch = batch[in_ring_mask]          # [R] graph IDs of ring atoms
            ring_k_sel = k[in_ring_mask]              # [R, heads, head_dim]
            ring_v_sel = v[in_ring_mask]              # [R, heads, head_dim]

            # Expand index to match [R, heads, head_dim] shape
            idx = ring_batch.view(-1, 1, 1).expand_as(ring_k_sel)

            # scatter_mean via scatter_reduce
            ring_k.scatter_reduce_(0, idx, ring_k_sel, reduce='mean', include_self=False)
            ring_v.scatter_reduce_(0, idx, ring_v_sel, reduce='mean', include_self=False)

        # Broadcast ring context to all atoms
        node_ring_k = ring_k[batch]  # [N, heads, head_dim]
        node_ring_v = ring_v[batch]

        # Attention scores
        attn = (q * node_ring_k).sum(dim=-1) / math.sqrt(self.head_dim)  # [N, heads]
        
        # Higher attention for ring atoms
        attn = attn + in_ring_mask.float().unsqueeze(-1) * 0.5

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted values
        out = node_ring_v * attn.unsqueeze(-1)  # [N, heads, head_dim]
        out = out.reshape(N, self.hidden_dim)

        return self.out_proj(out)


class GlobalGraphPool(nn.Module):
    """
    IMPROVED: Global pooling with gating mechanism.
    
    Changes:
    - Gated update (learn when to use global context)
    - Separate MLP for global features
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Global feature MLP — with dropout for regularization
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Gating mechanism (NEW) — with dropout before sigmoid
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Global mean pool per graph
        graph_emb = global_mean_pool(x, batch)  # [B, hidden_dim]
        
        # Process global features
        graph_emb = self.global_mlp(graph_emb)  # [B, hidden_dim]
        
        # Broadcast back to nodes
        node_graph = graph_emb[batch]  # [N, hidden_dim]
        
        # Gated fusion (NEW)
        combined = torch.cat([x, node_graph], dim=-1)  # [N, 2*hidden_dim]
        gate = self.gate(combined)  # [N, hidden_dim]
        
        return gate * node_graph  # Gated global context



