# Classifier Changelog — All Modifications

> **Date**: 2026-02-26  
> **Scope**: Performance optimization + interpretability fixes for the molecular glue classifier

---

## Files Modified (Summary)

| File | Change Type | Description |
|------|:-----------:|-------------|
| `model/graph_transformer.py` | **Perf + Bugfix** | Vectorized `RingAttentionLayer`, AMP dtype fix |
| `model/classifier.py` | **Perf + Refactor** | Removed `x.clone()`, streamlined ring flag passing |
| `train/classifier_trainer.py` | **Perf** | GPU-side metric accumulation |
| `config_classifier.py` | **Config** | Documented `num_workers=0` rationale |
| `interpret_classifier.py` | **Bugfix** | Complete rewrite — 3 critical bugs fixed |
| `CLASSIFIER_ARCHITECTURE.md` | **New** | Architecture & pipeline documentation |

---

## 1. `model/graph_transformer.py` — RingAttentionLayer

### Change 1a: Vectorized Ring Pooling (Critical Performance Fix)

**Problem**: The original `RingAttentionLayer.forward()` had a **Python for-loop** iterating over every graph in the batch to pool ring atoms:

```python
# BEFORE — lines 248-253 (REMOVED)
for graph_id in range(num_graphs):           # 128 iterations with batch_size=128!
    graph_mask = (batch == graph_id) & in_ring_mask
    if graph_mask.sum() > 0:
        ring_k[graph_id] = k[graph_mask].mean(dim=0)
        ring_v[graph_id] = v[graph_mask].mean(dim=0)
        ring_count[graph_id] = graph_mask.sum()
```

With batch_size=128 and 4 transformer layers, this caused **~1024 serial GPU kernel launches per training step** (128 graphs × 4 layers × 2 for forward+backward). Each `.mean()` with a boolean mask forces a CUDA synchronization. This was the **primary cause of the 5s/it bottleneck** on A100.

**Fix**: Replaced with fully vectorized `scatter_reduce_` — a single GPU kernel:

```python
# AFTER — vectorized scatter_reduce (no Python loop)
if in_ring_mask.any():
    ring_batch = batch[in_ring_mask]          # [R] graph IDs of ring atoms
    ring_k_sel = k[in_ring_mask]              # [R, heads, head_dim]
    ring_v_sel = v[in_ring_mask]              # [R, heads, head_dim]

    idx = ring_batch.view(-1, 1, 1).expand_as(ring_k_sel)

    ring_k.scatter_reduce_(0, idx, ring_k_sel, reduce='mean', include_self=False)
    ring_v.scatter_reduce_(0, idx, ring_v_sel, reduce='mean', include_self=False)
```

**Impact**: Eliminates ~1024 serial GPU calls per step → expected **5-10× speedup** for this layer alone.

### Change 1b: Flexible Input Format

**Before**: Only accepted full 2D node feature matrix `[N, feature_dim]`.

**After**: Accepts both:
- 1D tensor `[N]` — pre-extracted `in_ring` flags (used by classifier)
- 2D tensor `[N, feature_dim]` — full feature matrix (used by diffusion model)

```python
# NEW — dimension-based dispatch
if node_features.dim() == 1:
    in_ring_mask = node_features > 0.5       # Pre-extracted flags
else:
    in_ring_mask = node_features[:, 20] > 0.5  # Full feature matrix
```

**Why**: This maintains **backward compatibility** with the diffusion model's `GraphTransformer` (which still passes `original_x` as 2D) while allowing the classifier to skip the full tensor clone.

### Change 1c: AMP-Compatible dtype

**Problem**: Under mixed precision (AMP), the Q/K/V projections output `float16`, but `torch.zeros()` defaults to `float32`. `scatter_reduce_` requires matching dtypes.

```python
# BEFORE
ring_k = torch.zeros(num_graphs, self.num_heads, self.head_dim, device=device)

# AFTER — inherits dtype from the projected tensors
ring_k = torch.zeros(num_graphs, self.num_heads, self.head_dim, device=device, dtype=k.dtype)
ring_v = torch.zeros(num_graphs, self.num_heads, self.head_dim, device=device, dtype=v.dtype)
```

### Change 1d: Simplified Ring Attention Bias

**Before**: Conditionally applied ring atom bias only when `node_features is not None`.

**After**: Always applies the +0.5 attention bias for ring atoms (the mask defaults to all-True if no features provided, so this is safe).

```python
# BEFORE
if node_features is not None:
    attn = attn + in_ring_mask.float().unsqueeze(-1) * 0.5

# AFTER — always applied
attn = attn + in_ring_mask.float().unsqueeze(-1) * 0.5
```

---

## 2. `model/classifier.py` — MolecularGlueClassifier

### Change 2a: Eliminated `x.clone()` in Forward Pass

**Problem**: Every forward pass cloned the entire node feature tensor `[N, 23]` just so `RingAttentionLayer` could later access `x[:, 20]` (the `in_ring` column).

```python
# BEFORE
original_x = x.clone()                                    # Unnecessary full copy
for layer in self.layers:
    h, e = layer(h, edge_index, e, batch, original_x)     # Passing full [N, 23] tensor

# AFTER
in_ring_flags = x[:, 20]                                  # Extract just the column needed
for layer in self.layers:
    h, e = layer(h, edge_index, e, batch, in_ring_flags)  # Pass 1D [N] tensor
```

**Impact**: Eliminates one full tensor allocation and copy per forward pass.

### Change 2b: Updated `ClassifierTransformerBlock.forward` Signature

Renamed parameter `node_features` → `in_ring_flags` for clarity:

```python
# BEFORE
def forward(self, h, edge_index, edge_attr, batch, node_features=None):
    ...
    h = h + self.ring_attn(self.norm2(h), batch, node_features)

# AFTER
def forward(self, h, edge_index, edge_attr, batch, in_ring_flags=None):
    ...
    h = h + self.ring_attn(self.norm2(h), batch, in_ring_flags)
```

---

## 3. `train/classifier_trainer.py` — ClassifierTrainer

### Change 3a: GPU-Side Metric Accumulation

**Problem**: Every training iteration transferred predictions to CPU and converted to numpy:

```python
# BEFORE — per-iteration CPU transfer (forces CUDA sync every step)
preds = torch.sigmoid(logits).detach().cpu() > 0.5
all_preds.extend(preds.numpy())
all_labels.extend(batch.y.cpu().numpy())
```

**After**: Predictions stay on GPU as tensors, single transfer at epoch end:

```python
# AFTER — accumulate on GPU
all_preds.append((torch.sigmoid(logits).detach() > 0.5).float())
all_labels.append(batch.y.detach())

# Single CPU transfer at epoch end
all_preds_np = torch.cat(all_preds).cpu().numpy()
all_labels_np = torch.cat(all_labels).cpu().numpy()
accuracy = accuracy_score(all_labels_np, all_preds_np)
```

**Impact**: Eliminates per-iteration GPU→CPU synchronization, letting the GPU pipeline stay full.

### Change 3b: torch.compile() (Added then Removed)

Initially added `torch.compile()` for kernel fusion on A100, but **removed** because:
- The DGX environment lacked a C compiler (`gcc`) required by Triton's JIT
- `torch.compile` also caused graph breaks on `.item()` calls in `RingAttentionLayer`

---

## 4. `config_classifier.py` — ClassifierConfig

### Change 4: num_workers Documentation

**Attempted**: Changing `num_workers` from 0 → 4 to enable parallel collation.

**Reverted**: With 46K+ pre-cached `Data` objects stored as `self.graphs` on the dataset, each DataLoader worker must pickle the **entire dataset** into its own process, exhausting `/dev/shm` shared memory.

```python
# Final state — stays at 0, but with clear documentation
num_workers: int = 0  # Must be 0: pre-cached graphs (46K+ Data objects) cause shared memory OOM in workers
```

---

## 5. `interpret_classifier.py` — Interpretability Script

### Complete Rewrite — 3 Critical Bugs Fixed

**Bug 1**: Used nonexistent `converter.smiles_to_graph()` method.

```python
# BEFORE (broken)
graph = converter.smiles_to_graph(smiles)

# AFTER
from data.molecular_graph import smiles_to_graph
graph = smiles_to_graph(smiles)
```

**Bug 2**: Captum's `IntegratedGradients` corrupted graph structure via `additional_forward_args`. When Captum interpolates inputs, it expands **all** tensor arguments along dim 0 — turning `edge_index` from `[2, E]` to `[2*n_steps, E]`.

```python
# BEFORE (broken) — Captum expands edge_index
ig = IntegratedGradients(wrapper_module)
attributions = ig.attribute(
    graph.x,
    additional_forward_args=(graph.edge_index, graph.edge_attr, batch_idx),  # Gets corrupted!
)

# AFTER — closure captures graph structure, Captum only sees x
edge_index = graph.edge_index
edge_attr = graph.edge_attr
def forward_fn(x):
    return model(x, edge_index, edge_attr, batch_idx).squeeze(-1)
ig = IntegratedGradients(forward_fn)
attributions = ig.attribute(graph.x, baselines=baseline, n_steps=50, internal_batch_size=1)
```

**Bug 3**: Even with the closure fix, Captum batches all 50 interpolation steps into one call, expanding `x` from `[N, 23]` to `[50*N, 23]`. This breaks `RingAttentionLayer` because `batch_idx` is still `[N]`.

```python
# Fix: process one interpolation step at a time
internal_batch_size=1
```

**Bug 4**: Single graphs from `smiles_to_graph()` don't have a `.batch` attribute. Added manual creation:

```python
batch_idx = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
```

---

## 6. `CLASSIFIER_ARCHITECTURE.md` — New File

Created comprehensive architecture & pipeline documentation covering:
- Full pipeline flow (data creation → dataset → training → evaluation)
- Model architecture diagram with all 4 transformer block layers
- Node features (23-dim) and edge features (8-dim) tables
- Training pipeline features (DDP, AMP, cuDNN, etc.)
- Hyperparameter table
- File map
- Quick start commands

---

## Backward Compatibility

All changes are **fully backward compatible** with the diffusion model:

| Component | Diffusion Model | Classifier |
|-----------|----------------|-----------|
| `RingAttentionLayer` | Receives 2D `original_x [N, 23]` → uses `[:, 20]` | Receives 1D `in_ring_flags [N]` → uses directly |
| `MultiScaleBlock` | Still passes `node_features` → unchanged | `ClassifierTransformerBlock` passes `in_ring_flags` |
| `GraphTransformer` | Still does `original_x = x.clone()` → unchanged | `MolecularGlueClassifier` does `x[:, 20]` → lighter |

The `RingAttentionLayer` dispatches based on `node_features.dim()`:
- `dim() == 1` → pre-extracted flags (classifier path)
- `dim() == 2` → full feature matrix (diffusion path)
