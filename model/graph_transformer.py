"""
IMPROVED Graph Transformer for molecular denoising in diffusion models.

Key improvements over original:
1. Fixed edge feature propagation in attention layers
2. Added gradient checkpointing for memory efficiency
3. Improved FiLM conditioning with conditional dropout
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


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timestamp conditioning."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.float().unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class PropertyEmbedding(nn.Module):
    """
    IMPROVED: MLP with conditional dropout for classifier-free guidance.
    
    Changes:
    - Added dropout_prob parameter for classifier-free guidance
    - Separate MLPs for scale and shift (more expressive)
    - Layer normalization for stability
    """

    def __init__(self, num_properties: int, hidden_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        
        # Shared base MLP
        self.base_mlp = nn.Sequential(
            nn.Linear(num_properties, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added for stability
            nn.GELU(),
            nn.Dropout(0.1),  # Internal dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Separate projections for scale and shift (more expressive)
        self.scale_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bound scale to [-1, 1] then shift to [0, 2]
        )
        
        self.shift_mlp = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, properties: torch.Tensor):
        """
        Args:
            properties: [B, num_properties]
        Returns:
            scale: [B, hidden_dim], shift: [B, hidden_dim]
        """
        # Classifier-free guidance: randomly zero out properties during training
        if self.training and torch.rand(1).item() < self.dropout_prob:
            properties = torch.zeros_like(properties)
        
        h = self.base_mlp(properties)
        
        # Scale bounded to [0, 2] range (1 ± 1)
        scale = self.scale_mlp(h) + 1.0  # Map [-1, 1] -> [0, 2]
        shift = self.shift_mlp(h)
        
        return scale, shift


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
        
        # Edge feature update (NEW)
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.edge_dim, self.edge_dim),
            nn.GELU(),
            nn.Linear(self.edge_dim, self.edge_dim)
        )

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
        
        # Update edge features (NEW)
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_attr_out = edge_attr + self.edge_update_mlp(edge_input)  # Residual
        
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
                node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node embeddings [N, hidden_dim]
            batch: Batch assignment [N]
            node_features: Original node features for ring detection [N, feature_dim]
        
        Returns:
            Updated node features with ring context
        """
        N = x.shape[0]
        device = x.device

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        # Detect ring atoms (NEW - uses actual ring information)
        if node_features is not None:
            # Assuming in_ring is at index 20 in node features
            in_ring_mask = node_features[:, 20] > 0.5  # Boolean mask
        else:
            # Fallback: use all atoms
            in_ring_mask = torch.ones(N, dtype=torch.bool, device=device)
        
        num_graphs = batch.max().item() + 1

        # Pool ring atoms per graph
        ring_k = torch.zeros(num_graphs, self.num_heads, self.head_dim, device=device)
        ring_v = torch.zeros(num_graphs, self.num_heads, self.head_dim, device=device)
        ring_count = torch.zeros(num_graphs, device=device)

        # Only aggregate ring atoms
        for graph_id in range(num_graphs):
            graph_mask = (batch == graph_id) & in_ring_mask
            if graph_mask.sum() > 0:
                ring_k[graph_id] = k[graph_mask].mean(dim=0)
                ring_v[graph_id] = v[graph_mask].mean(dim=0)
                ring_count[graph_id] = graph_mask.sum()
        
        # Broadcast ring context to all atoms
        node_ring_k = ring_k[batch]  # [N, heads, head_dim]
        node_ring_v = ring_v[batch]

        # Attention scores
        attn = (q * node_ring_k).sum(dim=-1) / math.sqrt(self.head_dim)  # [N, heads]
        
        # Higher attention for ring atoms
        if node_features is not None:
            attn = attn + in_ring_mask.float().unsqueeze(-1) * 0.5  # Bias for ring atoms
        
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
        
        # Global feature MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Gating mechanism (NEW)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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


class MultiScaleBlock(nn.Module):
    """
    IMPROVED: Multi-scale transformer block with gradient checkpointing.
    
    Changes:
    - Edge features properly updated through layers
    - Gradient checkpointing option for memory efficiency
    - Pre-norm instead of post-norm (more stable)
    - Original node features passed for ring detection
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 edge_dim: int = None, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Local attention (message passing on graph edges)
        self.local_attn = GraphAttentionLayer(hidden_dim, num_heads, dropout, edge_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(edge_dim or hidden_dim)  # NEW

        # Ring attention
        self.ring_attn = RingAttentionLayer(hidden_dim, num_heads=max(1, num_heads // 2), dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Global pooling update
        self.global_pool = GlobalGraphPool(hidden_dim, dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch, node_features=None):
        """
        Returns:
            x: Updated node features
            edge_attr: Updated edge features
        """
        def _forward(x, edge_attr):
            # Pre-norm + local attention + residual
            x_norm = self.norm1(x)
            edge_norm = self.edge_norm(edge_attr)
            x_new, edge_new = self.local_attn(x_norm, edge_index, edge_norm)
            x = x + x_new
            edge_attr = edge_attr + edge_new

            # Pre-norm + ring attention + residual
            x = x + self.ring_attn(self.norm2(x), batch, node_features)

            # Pre-norm + global context + residual
            x = x + self.global_pool(self.norm3(x), batch)

            # Pre-norm + FFN + residual
            x = x + self.ffn(self.norm4(x))

            return x, edge_attr
        
        # Use gradient checkpointing if enabled (saves memory)
        if self.use_checkpoint and self.training:
            x, edge_attr = checkpoint(_forward, x, edge_attr)
        else:
            x, edge_attr = _forward(x, edge_attr)

        return x, edge_attr


class GraphTransformer(nn.Module):
    """
    IMPROVED: Graph Transformer for molecular denoising.

    Key improvements:
    1. Edge features propagated through all layers
    2. Gradient checkpointing option
    3. Better property conditioning with dropout
    4. Hybrid prediction heads (separate atom type, charge, hybridization)
    5. Improved ring attention with actual ring detection
    6. Pre-norm architecture for stability
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # Input dimensions
        node_input_dim = len(ATOM_TYPES) + len(CHARGES) + len(HYBRIDIZATIONS) + 4
        edge_input_dim = len(BOND_TYPES) + 3

        # Store for later use
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim

        # Input projections with layer norm
        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Property conditioning (FiLM) with classifier-free guidance
        self.property_embed = PropertyEmbedding(
            len(PROPERTY_NAMES), 
            hidden_dim, 
            dropout_prob=0.1  # 10% unconditional during training
        )

        # Multi-scale transformer blocks
        self.blocks = nn.ModuleList([
            MultiScaleBlock(
                hidden_dim, 
                config.num_heads, 
                config.dropout,
                edge_dim=hidden_dim,
                use_checkpoint=getattr(config, 'gradient_checkpointing', False)
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # IMPROVED: Separate output heads for different features
        # Atom type prediction
        self.atom_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, len(ATOM_TYPES)),
        )

        # Charge prediction
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, len(CHARGES)),
        )

        # Hybridization prediction (NEW - separate head)
        self.hybrid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, len(HYBRIDIZATIONS)),
        )

        # Binary features prediction (aromatic, in_ring, conjugated)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 4, 3),  # 3 binary features
            nn.Sigmoid()
        )

        # NumHs prediction (0-3)
        self.numhs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 4, 4),  # 0, 1, 2, 3 hydrogens
        )

        # Bond type prediction (uses both node features)
        self.bond_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, len(BOND_TYPES)),
        )

        # Bond features prediction (aromatic, conjugated, in_ring)
        self.bond_binary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

        # Graph-level output for property prediction
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, len(PROPERTY_NAMES)),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [N, node_input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_input_dim]
            t: Timestep [B]
            batch: Batch assignment [N]
            condition: Optional property targets [B, num_properties]

        Returns:
            Dictionary with all predictions
        """
        # Store original features for ring detection
        original_x = x.clone()
        
        # Project inputs
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)

        # Add time embedding (broadcast to nodes)
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        h = h + t_emb[batch]

        # Apply property conditioning via FiLM
        if condition is not None:
            scale, shift = self.property_embed(condition)  # [B, hidden_dim] each
            # Broadcast to nodes
            node_scale = scale[batch]  # [N, hidden_dim]
            node_shift = shift[batch]  # [N, hidden_dim]
            h = h * node_scale + node_shift

        # Transformer blocks (edge features updated through layers)
        for block in self.blocks:
            h, e = block(h, edge_index, e, batch, original_x)

        # Final normalization
        h = self.final_norm(h)

        # Node predictions (all separate heads)
        atom_type_logits = self.atom_type_head(h)
        charge_logits = self.charge_head(h)
        hybrid_logits = self.hybrid_head(h)
        binary_features = self.binary_head(h)  # [N, 3] (aromatic, in_ring, conjugated)
        numhs_logits = self.numhs_head(h)

        # Edge predictions: concatenate source and target node features
        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=-1)  # [E, 2*hidden_dim]
        bond_type_logits = self.bond_type_head(edge_features)
        bond_binary = self.bond_binary_head(edge_features)  # [E, 3]

        # Graph-level features
        graph_features = global_mean_pool(h, batch)  # [B, hidden_dim]
        graph_properties = self.graph_head(graph_features)  # [B, num_properties]

        return {
            # Node predictions
            'atom_type_logits': atom_type_logits,      # [N, 10]
            'charge_logits': charge_logits,            # [N, 5]
            'hybrid_logits': hybrid_logits,            # [N, 4]
            'binary_features': binary_features,        # [N, 3] (sigmoid)
            'numhs_logits': numhs_logits,             # [N, 4]
            
            # Edge predictions
            'bond_type_logits': bond_type_logits,      # [E, 5]
            'bond_binary': bond_binary,                # [E, 3] (sigmoid)
            
            # Graph-level
            'graph_properties': graph_properties,      # [B, 7]
            
            # Legacy compatibility (concatenated)
            'node_logits': torch.cat([
                atom_type_logits,
                charge_logits,
                hybrid_logits,
                binary_features,
                numhs_logits
            ], dim=-1),  # [N, 26]
            
            'edge_logits': torch.cat([
                bond_type_logits,
                bond_binary
            ], dim=-1),  # [E, 8]
            
            'graph_features': graph_properties,
        }

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training."""
        for block in self.blocks:
            block.use_checkpoint = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing (faster but uses more memory)."""
        for block in self.blocks:
            block.use_checkpoint = False


# Utility function for model initialization
def create_graph_transformer(config: ModelConfig) -> GraphTransformer:
    """
    Factory function to create and initialize GraphTransformer.
    
    Applies:
    - Xavier uniform initialization for linear layers
    - Small constant initialization for biases
    - Proper initialization for layer norms
    """
    model = GraphTransformer(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    model.apply(init_weights)
    
    return model