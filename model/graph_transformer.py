"""
Graph Transformer for molecular denoising.

A transformer-based graph neural network that predicts clean
molecular graphs from noisy inputs at various diffusion timesteps.
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import ModelConfig, ATOM_TYPES, BOND_TYPES, CHARGES


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GraphAttentionLayer(MessagePassing):
    """
    Multi-head graph attention layer with edge features.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        edge_dim: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
        """
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        
        # Edge bias
        edge_bias = self.edge_proj(edge_attr)  # [E, num_heads]
        
        # Message passing
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_bias=edge_bias)
        
        # Reshape and project
        out = out.view(-1, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out
    
    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_bias: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        # Compute attention scores
        attn = (q_i * k_j).sum(dim=-1) * self.scale  # [E, num_heads]
        attn = attn + edge_bias
        
        # Softmax over neighbors
        attn = softmax(attn, index, dim=0)
        attn = self.dropout(attn)
        
        # Weighted values
        out = attn.unsqueeze(-1) * v_j  # [E, num_heads, head_dim]
        
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with graph attention and feed-forward network.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        edge_dim: int = 7,
        dropout: float = 0.1,
        time_dim: int = 256,
    ):
        super().__init__()
        
        # Graph attention
        self.attention = GraphAttentionLayer(
            hidden_dim, num_heads, edge_dim, dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        time_emb: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # Time conditioning (scale and shift)
        time_out = self.time_mlp(time_emb)  # [B, hidden_dim * 2]
        scale, shift = time_out.chunk(2, dim=-1)  # [B, hidden_dim] each
        
        # Expand to node level
        scale = scale[batch]  # [N, hidden_dim]
        shift = shift[batch]  # [N, hidden_dim]
        
        # Attention with residual
        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = self.attention(h, edge_index, edge_attr)
        x = x + h
        
        # FFN with residual
        h = self.norm2(x)
        h = h * (1 + scale) + shift
        h = self.ffn(h)
        x = x + h
        
        return x


class GraphTransformer(nn.Module):
    """
    Graph Transformer for molecular denoising.
    
    Takes noisy molecular graphs and timesteps as input,
    predicts the clean node types and edge types.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input dimensions
        self.node_input_dim = len(ATOM_TYPES) + len(CHARGES) + 2  # +2 for aromatic, ring
        self.edge_input_dim = len(BOND_TYPES) + 2  # +2 for aromatic, ring
        
        # Embeddings
        self.node_embed = nn.Linear(self.node_input_dim, config.hidden_dim)
        self.edge_embed = nn.Linear(self.edge_input_dim, config.hidden_dim // 4)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim // 4,
                config.dropout,
                config.hidden_dim,
            )
            for _ in range(config.num_layers)
        ])
        
        # Output heads
        self.node_out = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(ATOM_TYPES)),
        )
        
        self.charge_out = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, len(CHARGES)),
        )
        
        # Edge prediction through node pairs
        self.edge_out = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(BOND_TYPES)),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            x: Node features [N, node_input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_input_dim]
            t: Timesteps [B]
            batch: Batch assignment [N]
            
        Returns:
            Dictionary with predicted node types, charges, and edge types
        """
        # Embed inputs
        h = self.node_embed(x)  # [N, hidden_dim]
        edge_h = self.edge_embed(edge_attr)  # [E, hidden_dim // 4]
        
        # Time embedding
        time_emb = self.time_embed(t.float())  # [B, hidden_dim]
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, edge_index, edge_h, time_emb, batch)
        
        # Predict node types
        node_logits = self.node_out(h)  # [N, num_atom_types]
        charge_logits = self.charge_out(h)  # [N, num_charges]
        
        # Predict edge types from node pairs
        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=-1)  # [E, hidden_dim * 2]
        edge_logits = self.edge_out(edge_features)  # [E, num_bond_types]
        
        return {
            'node_logits': node_logits,
            'charge_logits': charge_logits,
            'edge_logits': edge_logits,
        }
