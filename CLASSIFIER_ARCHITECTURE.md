   # Molecular Glue Classifier — Architecture & Pipeline Guide

> **Goal**: Binary classification — is a given molecule a molecular glue (1) or not (0)?

---

## Pipeline Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Creation   │────▶│  Dataset/Loader   │────▶│    Training      │────▶│   Evaluation     │────▶│  Interpretation  │
│                  │     │                   │     │                  │     │                  │     │                  │
│ create_          │     │ classifier_       │     │ classifier_      │     │ evaluate_        │     │ interpret_       │
│ classifier_data  │     │ dataset.py        │     │ trainer.py       │     │ classifier.py    │     │ classifier.py    │
│ .py              │     │                   │     │                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

**Entry points**:
```bash
python train_classifier.py --epochs 50          # Single-seed training
python run_multi_seed.py --num_seeds 10         # Multi-seed training + averaged metrics
```

---

## 1. Data Preparation (`create_classifier_data.py`)

Creates a **balanced** binary dataset with MW-matched negatives and scaffold-based splitting.

### 1.1 Data Sources

| Source | Label | File |
|--------|-------|------|
| Known molecular glue chemotypes | 1 (glue) | `data/glue_chemotypes.smi` |
| ZINC15 drug-like molecules | 0 (non-glue) | `data/zinc15_druglike.smi` |
| ChEMBL drug-like molecules | 0 (non-glue) | `data/chembl_druglike.csv` |

Negative sources are **pooled** together for better MW coverage.

### 1.2 MW-Stratified Sampling (Anti-Shortcut)

> **Problem**: Glues tend to have higher MW (300–600 Da) vs random drug-like molecules (150–350 Da). Without correction, a simple logistic regression on MW alone achieves ~99% accuracy — the model learns a trivial shortcut.

**Solution** — `mw_stratified_sample()`:

1. Compute the MW distribution of all glues
2. Divide the MW range into **20 histogram bins** (2nd–98th percentile)
3. Sample negatives **proportionally** to match the glue MW distribution per bin
4. Result: negatives and glues have overlapping MW distributions → MW alone can't separate classes

**Verification**: A MW-only logistic regression is run after sampling. Target: accuracy ≤ 60%.

### 1.3 Scaffold Split

All molecules sharing the same **Murcko scaffold** (core ring system) go to the **same** split — prevents data leakage from structurally similar molecules.

- **Stratified**: Glue and non-glue scaffolds are distributed independently to maintain ~50/50 class balance in each split
- **Split ratio**: 80% train / 10% val / 10% test
- Deduplication: negatives overlapping with glues are removed
- All SMILES validated with RDKit

### 1.4 Outputs

```
data/classifier_train.csv   (80% — ~50/50 glue/non-glue)
data/classifier_val.csv     (10%)
data/classifier_test.csv    (10%)
```

---

## 2. Dataset & Graph Conversion (`data/classifier_dataset.py`)

`MolecularGlueClassifierDataset` converts SMILES → PyG graphs **at init time** (pre-cached):

```
SMILES string ──▶ RDKit Mol ──▶ MolecularGraph ──▶ PyG Data object
                                                     ├── x: [N, 23]  (node features)
                                                     ├── edge_index: [2, E]
                                                     ├── edge_attr: [E, 8]  (edge features)
                                                     ├── y: [1]  (label)
                                                     └── smiles: str
```

### Node Features (23-dim per atom)

| Feature | Dims | Encoding |
|---------|------|----------|
| Atom type (C, N, O, S, F, Cl, Br, P, I, Other) | 10 | One-hot |
| Formal charge (-2 to +2) | 5 | One-hot |
| Hybridization (SP, SP2, SP3, Other) | 4 | One-hot |
| Is aromatic | 1 | Binary |
| Is in ring | 1 | Binary |
| Number of Hs | 1 | Integer |
| Is conjugated | 1 | Binary |

### Edge Features (8-dim per bond)

| Feature | Dims | Encoding |
|---------|------|----------|
| Bond type (Single, Double, Triple, Aromatic, None) | 5 | One-hot |
| Is aromatic | 1 | Binary |
| Is conjugated | 1 | Binary |
| Is in ring | 1 | Binary |

### Graph Augmentation (Training Only)

When `augment=True`, each sample has a 50% chance of augmentation:

| Augmentation | Description | Rate |
|-------------|-------------|------|
| Node feature masking | Randomly zeros out atom features | 15% of atoms |
| Edge dropping | Randomly removes bonds | 10% of edges |

All graphs are **pre-cached in memory** — no SMILES parsing during training.

---

## 3. Model Architecture (`model/classifier.py`)

**`MolecularGlueClassifier`** — a 3-layer Graph Transformer with ~1.8M parameters.

```
                         ┌───── Node features [N, 23] ─────┐
                         │                                  │
                    Node Encoder                      Edge Encoder
                  Linear → LN → GELU              Linear → LN → GELU
                   [N, 23] → [N, 128]              [E, 8] → [E, 128]
                         │                                  │
                         │     ┌────────────────────────┐   │
                         │     │ DropEdge: drop 15% of  │   │
                         │     │ edges (training only)   │   │
                         │     └────────────────────────┘   │
                         ▼                                  ▼
              ┌─────────────────────────────────────────────────┐
              │       ClassifierTransformerBlock (×3)            │
              │                                                 │
              │  1. GraphAttentionLayer (8-head attention)       │
              │     - Multi-head attention with edge bias        │
              │     - Edge feature update via MLP                │
              │                                                 │
              │  2. RingAttentionLayer (4-head)                  │
              │     - Vectorized scatter_reduce ring pooling     │
              │     - Ring atoms get +0.5 attention bias         │
              │       │
              │                                                 │
              │  3. GlobalGraphPool (gated)                      │
              │     - Global mean pool → MLP → gated broadcast  │
              │                                                 │
              │  4. Feed-Forward Network                         │
              │     - Linear(128→256) → GELU → Linear(256→128)  │
              │     - 2× expansion ratio                        │
              │                                                 │
              │  All with Pre-LayerNorm + Residual connections   │
              │  + Dropout(0.3)                                  │
              └─────────────────────────────────────────────────┘
                         │
                    Final LayerNorm
                         │
               ┌─────────┴──────────┐
          global_mean_pool      global_max_pool
            [B, 128]              [B, 128]
               └─────────┬──────────┘
                    cat → [B, 256]
                         │
                  Classification Head
              Linear(256→128) → LN → GELU → Dropout(0.4)
              Linear(128→64)  → LN → GELU → Dropout(0.4)
              Linear(64→1)    → logit
                         │
                    BCEWithLogitsLoss
```

### Why This Architecture?

- **Ring attention** — molecular glues are defined by their ring scaffolds (e.g. glutarimide/IMiD cores). Ring-centric attention provides an inductive bias for detecting these chemotypes.
- **Gated global pooling** — lets the model learn *when* global molecular context matters vs. local structure.
- **Mean + Max pooling** — captures both average atom behavior and the most extreme/distinguishing features.
- **DropEdge** — randomly removes 15% of edges during training, preventing over-reliance on specific bond connectivity.
- **Reduced capacity** — 128 hidden dim / 3 layers (vs 256/4 in diffusion model) to prevent overfitting on a small dataset.

### Components Reused from Diffusion Model

| Component | Source | What It Does |
|-----------|--------|-------------|
| `GraphAttentionLayer` | `model/graph_transformer.py` | Multi-head message passing with edge features |
| `RingAttentionLayer` | `model/graph_transformer.py` | Ring-centric attention — vectorized with `scatter_reduce_` |
| `GlobalGraphPool` | `model/graph_transformer.py` | Gated global context pooling |
| `smiles_to_graph()` | `data/molecular_graph.py` | SMILES → PyG Data conversion |

The classifier removes diffusion-specific concepts (timestep embedding, FiLM conditioning, noise prediction heads) and adds a classification head.

---

## 4. Training Pipeline (`train/classifier_trainer.py`)

### Parallelism Strategy (auto-detected)

| Launch Method | Strategy | Speed |
|---------------|----------|-------|
| `torchrun --nproc_per_node=N` | DistributedDataParallel (DDP) | Fastest |
| `python` (multi-GPU) | DataParallel | Fallback |
| `python` (1 GPU) | Single GPU + AMP | Default |

### Anti-Overfitting Features (6 complementary techniques)

| Technique | Config | Effect |
|-----------|--------|--------|
| Dropout | 0.3 | Randomly zeros neurons |
| DropEdge | 0.15 | Randomly removes bonds during training |
| Label smoothing | 0.05 | Softens targets: `y' = y × 0.95 + 0.025` |
| Weight decay | 1e-3 | L2 penalty (100× stronger than diffusion model) |
| Reduced capacity | 128 dim, 3 layers | ~1.8M params (vs ~7M in diffusion model) |
| Graph augmentation | 50% prob | Node masking (15% atoms) + edge dropping (10% edges) |

### Training Features

- **Mixed precision (AMP)**: `torch.amp.autocast` + `GradScaler` for ~2× speedup
- **cuDNN auto-tuner**: `torch.backends.cudnn.benchmark = True`
- **Efficient gradient ops**: `zero_grad(set_to_none=True)`, gradient clipping at 1.0
- **GPU-side metrics**: Predictions accumulated on GPU, single CPU transfer at epoch end
- **Gradient accumulation**: 2 steps → effective batch = 128

### Optimizer & Scheduler

- **Optimizer**: AdamW (Adam with decoupled weight decay)
- **Scheduler**: OneCycleLR (cosine annealing with 10% warmup)
  - LR starts low → ramps up → decays via cosine
  - Steps per **batch**, not per epoch

### Early Stopping

- Monitors **validation loss**
- Stops if val loss doesn't improve for **10 consecutive epochs**
- Best model (by **val AUC**) saved as `best_classifier.pt`

### Hyperparameters (`config_classifier.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_dim | 128 | Reduced from diffusion model's 256 |
| num_layers | 3 | Reduced from 4 |
| num_heads | 8 | Attention heads |
| dropout | 0.3 | Stronger than usual |
| batch_size | 64 | Per-GPU |
| learning_rate | 1e-4 | AdamW |
| weight_decay | 1e-3 | Strong L2 regularization |
| label_smoothing | 0.05 | Soft targets |
| drop_edge_rate | 0.15 | DropEdge regularization |
| early_stop_patience | 10 | Epochs without improvement |
| gradient_accumulation | 2 | Effective batch = 128 |
| use_cosine_schedule | True | OneCycleLR with warmup |
| num_workers | 0 | Must be 0: pre-cached graphs cause shared memory OOM |

### Checkpointing

Saved at `checkpoints/classifier/best_classifier.pt`:
```python
{
    'model_state_dict': ...,
    'config': {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 8,
        'dropout': 0.3,
        'drop_edge_rate': 0.15
    },
    'epoch': ...,
    'best_val_auc': ...,
    'optimizer_state_dict': ...,
}
```

Config is embedded in the checkpoint so the model can be reconstructed without external config files.

---

## 5. Evaluation (`evaluate_classifier.py`)

Comprehensive evaluation on the scaffold-split test set:

| Output | Description |
|--------|-------------|
| `confusion_matrix.png` | TP/FP/TN/FN heatmap |
| `roc_curve.png` | ROC curve with AUC score |
| `error_cases.csv` | All misclassified molecules with probabilities |
| Console | Classification report + MCC, Spearman, Pearson correlations |

### Metrics

| Metric | What It Measures |
|--------|-----------------|
| **MCC** (Matthews Correlation Coefficient) | Best single metric for binary classification quality — balanced even with class imbalance |
| **ROC-AUC** | Separability across all thresholds |
| **F1** | Harmonic mean of precision & recall |
| **Precision / Recall** | Per-class performance |
| **SPCC** (Spearman) | Monotonic relationship between predicted prob and true label |
| **PCC** (Pearson) | Linear relationship between predicted prob and true label |

---

## 6. Multi-Seed Evaluation (`run_multi_seed.py`)

Trains and evaluates the classifier across **multiple random seeds** (default: 10) to report averaged metrics with standard deviations.

```bash
python run_multi_seed.py                          # 10 seeds (0-9)
python run_multi_seed.py --seeds 42 123 456       # specific seeds
python run_multi_seed.py --num_seeds 5 --epochs 5 # 5 seeds, 5 epochs each
python run_multi_seed.py --eval_only              # skip training, evaluate existing checkpoints
```

**What changes per seed**: Model weight initialisation, dropout masks, augmentation randomness.
**What stays the same**: The data (same train/val/test split).

### Output

- Per-seed metrics table (MCC, AUC, F1, Accuracy, Precision, Recall, SPCC, PCC)
- Aggregated table: **mean / std / min / max** across all seeds
- CSV saved to `checkpoints/classifier/multi_seed_results.csv`

Checkpoints saved separately per seed: `checkpoints/classifier/seed_{N}/best_classifier.pt`

This is the standard way to report results in ML papers:
> *MCC: 0.72 ± 0.03 (mean ± std over 10 seeds)*

---

## 7. Interpretability (`interpret_classifier.py`)

Post-hoc explainability using **Gradient × Input** attribution:

1. Forward pass → compute logits for a molecule
2. Backward pass → gradients of logit w.r.t. input node features
3. Attribution: `importance = |gradient × input|` → summed across feature dims → one score per atom
4. Visualisation via RDKit atom highlighting (green = important, red = unimportant)

**Expected result**: For correctly classified molecular glues, the glutarimide/phthalimide ring nitrogen and carbonyl atoms should be highlighted.

---

## File Map

```
├── config_classifier.py           # All hyperparameters (dataclass)
├── create_classifier_data.py      # Data prep: MW-stratified sampling + scaffold split
├── train_classifier.py            # Training entry point (single seed, handles DDP)
├── run_multi_seed.py              # Multi-seed train + eval with averaged metrics
├── evaluate_classifier.py         # Post-training evaluation (metrics + plots)
├── interpret_classifier.py        # Gradient×Input atom importance maps
│
├── data/
│   ├── classifier_dataset.py      # Dataset class (pre-caching + graph augmentation)
│   ├── molecular_graph.py         # SMILES → PyG graph conversion
│   ├── glue_chemotypes.smi        # Positive data (known molecular glues)
│   ├── zinc15_druglike.smi        # Negative pool (ZINC15)
│   ├── chembl_druglike.csv        # Negative pool (ChEMBL)
│   ├── classifier_train.csv       # Training split (80%, scaffold-split)
│   ├── classifier_val.csv         # Validation split (10%)
│   └── classifier_test.csv        # Test split (10%)
│
├── model/
│   ├── classifier.py              # MolecularGlueClassifier (3-layer Graph Transformer)
│   └── graph_transformer.py       # Shared layers (GraphAttn, RingAttn, GlobalPool)
│
├── train/
│   └── classifier_trainer.py      # Training engine (DDP, AMP, early stopping, metrics)
│
├── utils/
│   ├── glue_scoring.py            # Rule-based glue-likeness score (0–100)
│   ├── chemistry.py               # Molecular property calculations
│   └── filters.py                 # PAINS, reactive group filters
│
└── checkpoints/
    └── classifier/
        ├── best_classifier.pt         # Best single-seed checkpoint
        ├── seed_0/best_classifier.pt  # Multi-seed checkpoints
        ├── seed_1/best_classifier.pt
        └── multi_seed_results.csv     # Aggregated results table
```

---

## Quick Start

```bash
# 1. Create dataset (one-time — MW-stratified sampling + scaffold split)
python create_classifier_data.py

# 2. Train (single seed)
python train_classifier.py --epochs 50 --batch_size 64

# 3. Multi-GPU training
torchrun --nproc_per_node=4 train_classifier.py --epochs 50

# 4. Evaluate
python evaluate_classifier.py

# 5. Multi-seed training + averaged metrics (recommended for reporting)
python run_multi_seed.py --num_seeds 10 --epochs 50

# 6. Interpret a prediction
python interpret_classifier.py
```
