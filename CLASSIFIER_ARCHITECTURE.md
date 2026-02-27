# Molecular Glue Classifier — Architecture & Pipeline Guide

> **Goal**: Binary classification — is a given molecule a molecular glue (1) or not (0)?

---

## Pipeline Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Creation   │────▶│  Dataset/Loader   │────▶│    Training      │────▶│   Evaluation     │
│                  │     │                   │     │                  │     │                  │
│ create_          │     │ classifier_       │     │ classifier_      │     │ evaluate_        │
│ classifier_data  │     │ dataset.py        │     │ trainer.py       │     │ classifier.py    │
│ .py              │     │                   │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

**Entry point**: `python train_classifier.py --epochs 50 --batch_size 128`

---

## 1. Data Preparation (`create_classifier_data.py`)

Creates a balanced binary dataset from two sources:

| Source | Label | Description |
|--------|-------|-------------|
| `data/glue_chemotypes.smi` + `data/glue_chemotypes.csv` | 1 (glue) | Known molecular glue chemotypes (~28K molecules) |
| `data/chembl_druglike.csv` | 0 (non-glue) | ChEMBL druglike molecules (sampled to match glue count) |

**Output**: `data/classifier_dataset.csv` — ~57K molecules with columns `SMILES, label`

---

## 2. Dataset & Graph Conversion (`data/classifier_dataset.py`)

`MolecularGlueClassifierDataset` converts SMILES → PyG graphs **at init time** (one-time cost):

```
SMILES string ──▶ RDKit Mol ──▶ MolecularGraph ──▶ PyG Data object
                                                     ├── x: [N, 23]  (node features)
                                                     ├── edge_index: [2, E]
                                                     ├── edge_attr: [E, 8]  (edge features)
                                                     └── y: [1]  (label)
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

All 46K+ graphs are **pre-cached in memory** — epochs are fast since no SMILES parsing happens during training.

---

## 3. Model Architecture (`model/classifier.py`)

**`MolecularGlueClassifier`** — a 4-layer Graph Transformer with 7M parameters.

```
                         ┌───── Node features [N, 23] ─────┐
                         │                                  │
                    Node Encoder                      Edge Encoder
                  Linear → LN → GELU              Linear → LN → GELU
                   [N, 23] → [N, 256]              [E, 8] → [E, 256]
                         │                                  │
                         ▼                                  ▼
              ┌─────────────────────────────────────────────────┐
              │         ClassifierTransformerBlock (×4)          │
              │                                                 │
              │  1. GraphAttentionLayer (multi-head attention)   │
              │     - 8-head attention with edge bias            │
              │     - Edge feature update via MLP                │
              │                                                 │
              │  2. RingAttentionLayer                           │
              │     - Pools ring atoms per graph (vectorized)    │
              │     - Broadcasts ring context to all atoms       │
              │     - Ring atoms get attention bias (+0.5)       │
              │                                                 │
              │  3. GlobalGraphPool (gated)                      │
              │     - Global mean pool → MLP → gated broadcast   │
              │                                                 │
              │  4. Feed-Forward Network                         │
              │     - Linear(256→1024) → GELU → Linear(1024→256)│
              │                                                 │
              │  All with Pre-LayerNorm + Residual connections   │
              └─────────────────────────────────────────────────┘
                         │
                    Final LayerNorm
                         │
               ┌─────────┴──────────┐
          global_mean_pool      global_max_pool
            [B, 256]              [B, 256]
               └─────────┬──────────┘
                    cat → [B, 512]
                         │
                  Classification Head
              Linear(512→256) → LN → GELU
              Linear(256→128) → LN → GELU
              Linear(128→1)  → logit
                         │
                    BCEWithLogitsLoss
```

### Why This Architecture?

- **Ring attention** is key — molecular glues are defined by their ring scaffolds (e.g. phthalimide/IMiD cores). Ring-centric attention gives the model an inductive bias for detecting these chemotypes.
- **Gated global pooling** lets the model learn *when* global molecular context matters vs. local structure.
- **Mean + Max pooling** captures both average atom behavior and the most extreme/distinguishing features.

### Key Components Reused from Diffusion Model

| Component | Source | What It Does |
|-----------|--------|-------------|
| `GraphAttentionLayer` | `model/graph_transformer.py` | Multi-head message passing with edge features |
| `RingAttentionLayer` | `model/graph_transformer.py` | Ring-centric attention with actual ring detection |
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
| `python` (1 GPU) | Single GPU | Default |

### Training Features

- **Mixed precision (AMP)**: `torch.amp.autocast` + `GradScaler` for ~2× speedup
- **cuDNN auto-tuner**: `torch.backends.cudnn.benchmark = True`
- **Efficient gradient ops**: `zero_grad(set_to_none=True)`, gradient clipping at 1.0
- **GPU-side metrics**: Predictions accumulated on GPU, single CPU transfer at epoch end
- **LR scheduler**: `ReduceLROnPlateau` (monitors val accuracy, halves LR after 5 stagnant epochs)

### Hyperparameters (`config_classifier.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_dim | 256 | Embedding dimension |
| num_layers | 4 | Transformer blocks |
| num_heads | 8 | Attention heads |
| dropout | 0.1 | |
| batch_size | 64 | Use 128 on A100 |
| learning_rate | 1e-4 | Adam optimizer |
| weight_decay | 1e-5 | L2 regularization |
| num_epochs | 50 | |
| val_split | 0.2 | 80/20 train/val split |

---

## 5. Evaluation (`evaluate_classifier.py`)

Produces comprehensive evaluation artifacts:

| Output | Description |
|--------|-------------|
| `confusion_matrix.png` | TP/FP/TN/FN heatmap |
| `roc_curve.png` | ROC curve with AUC score |
| `error_cases.csv` | All misclassified molecules with probabilities |
| Console | Full classification report + MCC, Spearman, Pearson correlations |

---

## File Map

```
├── config_classifier.py           # All hyperparameters
├── create_classifier_data.py      # Data preparation script
├── train_classifier.py            # Entry point (handles DDP setup)
│
├── data/
│   ├── classifier_dataset.py      # Dataset class (SMILES → graphs, pre-caching)
│   ├── classifier_dataset.csv     # The actual dataset (SMILES + labels)
│   └── molecular_graph.py         # Graph conversion utilities
│
├── model/
│   ├── classifier.py              # MolecularGlueClassifier model
│   └── graph_transformer.py       # Shared attention layers (GraphAttn, RingAttn, GlobalPool)
│
├── train/
│   └── classifier_trainer.py      # Training loop (DDP, AMP, metrics, checkpointing)
│
├── evaluate_classifier.py         # Post-training evaluation
└── checkpoints/classifier/        # Saved model weights
```

---

## Quick Start

```bash
# 1. Create dataset (one-time)
python create_classifier_data.py

# 2. Train
python train_classifier.py --epochs 50 --batch_size 128

# 3. Multi-GPU training
torchrun --nproc_per_node=4 train_classifier.py --epochs 50 --batch_size 128

# 4. Evaluate
python evaluate_classifier.py
```
