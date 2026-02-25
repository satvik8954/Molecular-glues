# Molecular Glue Diffusion Model & Classifier

A graph-based deep learning system for molecular glue discovery, featuring both a **generative diffusion model** and a **binary classifier** built on shared Graph Transformer components.

## Features

- **Discrete Diffusion**: D3PM-style categorical noise for atom/bond types
- **Graph Transformer**: Multi-head attention with edge features, ring-centric attention, and global pooling
- **Binary Classifier**: Predicts whether a molecule is a molecular glue (reuses ~80% of diffusion model code)
- **Chemical Validity**: RDKit-based validation and filtering
- **Drug-Likeness**: Lipinski Rule of Five and glue-specific property filters
- **Interpretability**: Integrated gradients for atom-level importance visualization

## Installation

```bash
pip install -r requirements.txt
```

Requires:
- Python 3.8+
- PyTorch 2.0+
- RDKit 2023.03+
- PyTorch Geometric 2.3+
- scikit-learn 1.0+

---

## Quick Start

### Diffusion Model

```bash
# Quick test (5 epochs, small model)
python train_model.py --quick_test

# Full training
python train_model.py --epochs 100 --batch_size 64

# Generate molecules
python generate_molecules.py \
    --checkpoint checkpoints/best_model.pt \
    --n_molecules 1000 \
    --output generated.csv

# Analyze outputs
python eval/analyze.py \
    --input generated.csv \
    --training_data data/glue_chemotypes.csv
```

### Classifier

```bash
# Full pipeline (data prep → train → evaluate)
bash run_classifier_pipeline.sh

# Quick test (10 epochs, skip data prep if CSV exists)
bash run_classifier_pipeline.sh --skip-data --epochs 10

# Custom settings
bash run_classifier_pipeline.sh --epochs 100 --batch-size 64 --lr 5e-5

# Or run training directly
python train_classifier.py --epochs 50 --data_path data/classifier_dataset.csv

# Evaluate trained model
python evaluate_classifier.py

# Interpret predictions (atom-level importance)
python interpret_classifier.py
```

#### Classifier CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--data_path` | `data/classifier_dataset.csv` | Path to CSV with `SMILES` and `label` columns |

#### Pipeline Script Options

| Flag | Description |
|------|-------------|
| `--skip-data` | Skip data preparation (use existing CSV) |
| `--epochs N` | Override number of epochs |
| `--batch-size N` | Override batch size |
| `--lr F` | Override learning rate |
| `--data-path PATH` | Override dataset path |

---

## Project Structure

```
├── config.py                    # Diffusion model configuration
├── config_classifier.py         # Classifier configuration
├── train_model.py               # Diffusion training script
├── train_classifier.py          # Classifier training script
├── run_classifier_pipeline.sh   # Full classifier pipeline (data → train → eval)
├── generate_molecules.py        # Molecule generation script
├── evaluate_classifier.py       # Classifier evaluation & metrics
├── interpret_classifier.py      # Atom-level prediction interpretation
│
├── data/
│   ├── molecular_graph.py       # SMILES ↔ graph conversion (shared)
│   ├── dataset.py               # Diffusion dataset
│   ├── classifier_dataset.py    # Classifier dataset
│   ├── glue_chemotypes.csv      # Known molecular glues
│   └── chembl_druglike.csv      # Drug-like non-glues
│
├── data_prep/
│   └── collect_classifier_data.py  # Balanced dataset collection
│
├── model/
│   ├── graph_transformer.py     # Shared transformer layers
│   └── classifier.py            # Classifier model
│
├── train/
│   ├── trainer.py               # Diffusion trainer
│   └── classifier_trainer.py    # Classifier trainer
│
├── generate/                    # Generation & post-processing
├── eval/                        # Metrics & analysis
└── tests/                       # Unit tests
```

---

## Architecture

### Shared Components (in `model/graph_transformer.py`)

Both models share the same core transformer layers:

| Component | Description |
|-----------|-------------|
| `GraphAttentionLayer` | Multi-head attention with edge feature updates |
| `RingAttentionLayer` | Ring-centric attention using molecular ring information |
| `GlobalGraphPool` | Gated global mean pooling for graph-level context |

### Diffusion Model

```
SMILES → Graph → Node/Edge Encoders → Time Embed + FiLM Conditioning
                                       ↓
                              MultiScaleBlock × 8
                                       ↓
                              7 Denoising Heads (atom type, charge, bond type, ...)
```

- **Purpose**: Generate novel molecules from noise
- **Parameters**: ~15M
- **Loss**: Multi-objective (atom, bond, property, valency)

### Classifier

```
SMILES → Graph → Node/Edge Encoders → ClassifierTransformerBlock × 4
                                       ↓
                              Mean + Max Pooling
                                       ↓
                              Classification Head → Glue / Non-Glue
```

- **Purpose**: Predict if a molecule is a molecular glue
- **Parameters**: ~7M
- **Loss**: Binary cross-entropy
- **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC

### What the Classifier Removes vs. Diffusion

| Diffusion-only Component | Why Removed |
|--------------------------|-------------|
| `SinusoidalPositionEmbeddings` | No diffusion timestep in classification |
| `PropertyEmbedding` (FiLM) | No property conditioning needed |
| `NoiseScheduler` | No denoising process |
| 7 output heads | Replaced by single binary classification head |

---

## Molecular Glue Properties

Target molecules with:
- MW: 200-500 Da
- Aromatic/heterocyclic scaffolds
- Mixed polarity (TPSA 20-120)
- Multiple H-bond interaction points
- Low rotatable bonds (<8)

### Dataset Format

The classifier expects a CSV file with two columns:

```csv
SMILES,label
O=C1CCC(=O)N1c1cccc2[nH]ccc12,1
CC(=O)Oc1ccccc1C(=O)O,0
```

- `label=1`: Molecular glue
- `label=0`: Non-glue

---

## License

MIT
