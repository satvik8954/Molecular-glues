# Molecular Glue Diffusion Model — Improvements & Metrics

## Project Overview

A graph-based discrete diffusion model (D3PM-style) for generating small molecules with **molecular glue-like** properties. The model learns to denoise molecular graphs using a Graph Transformer architecture, conditioned on molecular properties, and generates novel drug-like candidates via reverse diffusion sampling.

---

## Architecture & Design Improvements

### 1. Discrete Diffusion Framework (D3PM)

- Implemented **D3PM-style categorical noise** for atom types, bond types, and formal charges — unlike continuous Gaussian diffusion, this respects the discrete nature of molecular data.
- **Cosine beta schedule** adapted from "Improved Denoising Diffusion Probabilistic Models" for smoother noise injection.
- Precomputed **transition matrices** for efficient forward/reverse process computation.
- Posterior sampling via `q(x_{t-1} | x_t, x_0)` with uniform mixing for numerical stability.

### 2. Graph Transformer with Multi-Scale Message Passing

- **Multi-head graph attention** (8 heads, 256 hidden dim, 8 layers) with edge feature integration.
- **Ring-centric attention layer**: per-graph pooling mechanism that provides explicit ring-structure awareness — critical for molecular glue scaffolds which are aromatic/heterocyclic.
- **Global graph pooling → broadcast**: a GlobalGraphPool module that gives every node access to full molecular context.
- **FiLM conditioning** (Feature-wise Linear Modulation): property vectors modulate node representations via learned scale and shift parameters at every transformer block.
- **Sinusoidal time embeddings** for continuous timestep conditioning.

### 3. Rich Molecular Graph Representation

| Feature Type    | Components                                                                                                          | Dimension |
| --------------- | ------------------------------------------------------------------------------------------------------------------- | --------- |
| **Node (atom)** | Atom type (one-hot, 10) + Charge (one-hot, 5) + Hybridization (one-hot, 4) + Aromatic + InRing + NumHs + Conjugated | **23**    |
| **Edge (bond)** | Bond type (one-hot, 5) + Aromatic + Conjugated + InRing                                                             | **8**     |

- Atom types: `C, N, O, S, F, Cl, Br, P, I, Other`
- Bond types: `None, Single, Double, Triple, Aromatic`
- Charges: `-2, -1, 0, +1, +2`
- Hybridizations: `SP, SP2, SP3, Other`

### 4. Multi-Objective Training Loss

| Loss Component                                       | Weight (λ) | Purpose                                                    |
| ---------------------------------------------------- | ---------- | ---------------------------------------------------------- |
| **Diffusion loss** (node + 0.5·charge + 0.5·edge CE) | 1.0        | Primary denoising objective                                |
| **Valency loss**                                     | 0.5        | Penalizes predicted atoms exceeding valid chemical valency |
| **Property matching loss**                           | 0.3        | Enforces property conditioning fidelity                    |
| **Fragment frequency loss**                          | 0.1        | Encourages realistic BRICS fragment usage                  |

- Valency loss uses soft atom-type probabilities × expected bond orders to compute differentiable valency violations.
- Property loss uses MSE between predicted graph features and target property vectors.

### 5. Classifier-Free Guidance

- **Condition dropout** (p=0.1) during training: randomly zeros out property conditioning vectors so the model learns both conditional and unconditional generation.
- At inference, **guidance scale** (default 2.0) interpolates between conditional and unconditional predictions for steerable generation.
- Predefined **glue-target property vectors** for guided sampling toward molecular glue chemical space.

### 6. Training Infrastructure

- **Exponential Moving Average (EMA)** of model parameters (decay=0.999) used for validation and generation.
- **Warmup + Cosine Annealing** learning rate schedule (1000 warmup steps, cosine decay to 1% of base LR).
- **Gradient clipping** (max norm 1.0) for stable training.
- **AdamW optimizer** with weight decay (1e-6).
- Periodic checkpointing every 10 epochs + best model tracking via validation loss.
- Train/validation split (90/10) with EMA-weighted validation.

### 7. Molecular Filtering Pipeline

A comprehensive multi-stage filtering pipeline for generated molecules:

| Filter                      | Description                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Lipinski Rule of Five**   | MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10 (allows 1 violation)                                               |
| **Glue-likeness**           | MW 200–500, LogP 0–4, ≥1 aromatic ring, ≤8 rotatable bonds, TPSA 20–140, ≤35 heavy atoms                 |
| **Target range**            | MW 300–450, LogP 1.5–3.5, 2–3 aromatic rings, HBD+HBA 4–8, Fsp3 > 0.2                                    |
| **PAINS filter**            | Pan-Assay INterference compoundS removal via RDKit FilterCatalog                                         |
| **Reactive groups**         | Reject aldehydes, epoxides, Michael acceptors, acyl halides, anhydrides, isocyanates, sulfonyl fluorides |
| **Synthetic accessibility** | SA score < 4.5 (reasonably synthesizable)                                                                |
| **Deduplication**           | Canonical SMILES dedup + optional training set exclusion for novelty                                     |

### 8. Diverse Sampling Strategy

- **Variable molecule sizes** (min 5 to max 50 atoms) for structural diversity.
- **Multi-temperature sampling** (0.7, 0.8, 0.9, 1.0, 1.1) to explore the generation landscape.
- **Oversampling** with configurable ratio (default 3×) to compensate for post-filtering rejection.
- **Minimum heavy atom threshold** (default 10) to reject trivially small molecules.

### 9. Fragment Vocabulary (BRICS)

- Automated **BRICS decomposition** of training molecules to build a fragment frequency dictionary.
- Generated molecules scored by how common their fragments are vs. training data.
- Provides a learned prior on realistic substructure usage.

### 10. Comprehensive Evaluation Suite

Implemented the following metrics:

| Metric                     | Description                                                                     |
| -------------------------- | ------------------------------------------------------------------------------- |
| **Validity rate**          | Fraction of chemically valid molecules (RDKit sanitization)                     |
| **Uniqueness rate**        | Fraction of unique canonical SMILES among valid outputs                         |
| **Novelty rate**           | Fraction of generated molecules not in training set                             |
| **Internal diversity**     | 1 − mean pairwise Tanimoto similarity (Morgan FP, radius 2)                     |
| **Scaffold diversity**     | Fraction of unique Murcko scaffolds                                             |
| **QED statistics**         | Mean, std, and fraction with QED > 0.4                                          |
| **Glue similarity**        | Mean nearest-neighbor Tanimoto to known glues, max similarity, fraction > 0.4   |
| **Wasserstein distance**   | Earth mover's distance between generated and training property distributions    |
| **Property distributions** | Per-property mean ± std for MW, LogP, TPSA, HBD, HBA, aromatic rings, Fsp3, QED |

### 11. Visualization & Analysis

- **Property distribution plots** (histograms + KDE) comparing generated vs. training data.
- **Scatter matrix** of key properties (MW, LogP, TPSA, aromatic rings, QED).
- **Scaffold distribution** bar chart (top-15 Murcko scaffolds).
- Summary statistics table with mean ± std and [min, max] ranges.

### 12. Output Formats

- **CSV** with all computed molecular properties (MW, LogP, HBD, HBA, TPSA, rotatable bonds, rings, heavy atoms, Fsp3, QED).
- **SDF** (Structure Data File) with 3D coordinates for visualization in molecular modeling software.

---

## Training Configuration

| Parameter            | Value           |
| -------------------- | --------------- |
| Hidden dimension     | 256             |
| Transformer layers   | 8               |
| Attention heads      | 8               |
| Dropout              | 0.1             |
| Activation           | GELU            |
| Diffusion timesteps  | 500             |
| Beta schedule        | Cosine          |
| Batch size           | 64              |
| Learning rate        | 1e-4            |
| Weight decay         | 1e-6            |
| EMA decay            | 0.999           |
| Warmup steps         | 1,000           |
| Gradient clip        | 1.0             |
| Training epochs      | 100             |
| Validation frequency | Every 5 epochs  |
| Checkpoint frequency | Every 10 epochs |

---

## Training Data

- **154 curated molecular glue chemotypes** across 8 chemical classes:
  - Imide scaffolds (glutarimide, phthalimide derivatives)
  - Kinase scaffolds (quinazoline-aniline derivatives)
  - Aromatic heterocycles (carbazole, indole, benzimidazole, etc.)
  - Drug-like molecules (aspirin, ibuprofen, caffeine, etc.)
  - PPI modulators
  - Bifunctional degraders (PROTAC-inspired)
  - Linker fragments
  - Natural product-derived scaffolds

---

## Generated Molecule Metrics

From the generated output (`generated.csv`, 18 molecules):

### Property Statistics

| Property              | Mean   | Range           |
| --------------------- | ------ | --------------- |
| Molecular Weight (Da) | 441.76 | 319.90 – 559.82 |
| LogP                  | 1.21   | -2.33 – 5.43    |
| HBD                   | 0      | 0 – 0           |
| HBA                   | 26.6   | 20 – 35         |
| TPSA (Å²)             | 295.68 | 184.60 – 459.90 |
| Rotatable bonds       | 0      | 0 – 0           |
| Rings                 | 1.22   | 1 – 3           |
| Aromatic rings        | 0.44   | 0 – 1           |
| Heavy atoms           | 27.5   | 20 – 35         |
| Fraction sp3          | 0.11   | 0.0 – 1.0       |
| QED                   | 0.418  | 0.289 – 0.613   |

### Core Generation Metrics

| Metric              | Value |
| ------------------- | ----- |
| Molecules generated | 18    |
| QED mean            | 0.418 |
| QED > 0.4 fraction  | ~50%  |

> **Note**: The current generated set reflects an early-stage model (trained on 154 molecules). With more training data and longer training, validity, diversity, and property alignment are expected to improve significantly.

---

## Model Checkpoints

| Checkpoint                                 | Description                              |
| ------------------------------------------ | ---------------------------------------- |
| `checkpoints/best_model.pt`                | Best validation loss model (EMA weights) |
| `checkpoints/final_model.pt`               | Final epoch model                        |
| `checkpoints/checkpoint_epoch_{10-100}.pt` | Periodic snapshots every 10 epochs       |

Each checkpoint stores: model weights, optimizer state, EMA state, LR scheduler state, training config, and full training history (loss curves, accuracy curves).

---

## Testing

Unit tests covering:

- `test_chemistry.py` — Molecular property calculations, SMILES validation, fingerprints, scaffolds
- `test_dataset.py` — Dataset loading, graph conversion, property extraction, filtering
- `test_diffusion.py` — Noise addition, posterior computation, training step, sampling
- `test_metrics.py` — Validity/uniqueness/novelty/diversity metrics, QED stats, glue similarity

---

## Project Structure

```
├── config.py                     # All hyperparameters and vocabularies
├── train_model.py                # Training entry point (CLI)
├── generate_molecules.py         # Generation entry point (CLI)
├── data/
│   ├── dataset.py                # MolecularGlueDataset with property conditioning
│   ├── molecular_graph.py        # SMILES ↔ PyG graph conversion (23-dim nodes, 8-dim edges)
│   └── glue_chemotypes.csv       # 154 curated training molecules
├── model/
│   ├── diffusion.py              # MolecularDiffusion: full diffusion model with multi-loss
│   ├── graph_transformer.py      # GraphTransformer with ring attention + FiLM + global pool
│   ├── noise_scheduler.py        # D3PM categorical noise scheduler
│   └── fragment_vocab.py         # BRICS fragment frequency scoring
├── train/
│   └── trainer.py                # Trainer with EMA, warmup+cosine LR, checkpointing
├── generate/
│   ├── sampler.py                # MoleculeSampler: basic, guided, and diverse sampling
│   └── postprocess.py            # Filtering, deduplication, CSV/SDF export
├── eval/
│   ├── metrics.py                # Full evaluation suite (validity, diversity, QED, Wasserstein)
│   └── analyze.py                # Visualization and summary statistics
├── utils/
│   ├── chemistry.py              # RDKit property calculation, validation, fingerprints
│   └── filters.py                # Drug-like, glue-like, PAINS, reactive group, SA filters
├── tests/                        # Unit tests
├── checkpoints/                  # Saved model checkpoints
└── requirements.txt              # Dependencies
```

---

## Key Dependencies

| Package              | Version   | Purpose                                             |
| -------------------- | --------- | --------------------------------------------------- |
| PyTorch              | ≥ 2.0     | Deep learning framework                             |
| PyTorch Geometric    | ≥ 2.3     | Graph neural networks                               |
| RDKit                | ≥ 2023.03 | Cheminformatics (validation, properties, scaffolds) |
| NumPy                | ≥ 1.24    | Numerical computing                                 |
| Pandas               | ≥ 2.0     | Data handling                                       |
| Matplotlib + Seaborn | —         | Visualization                                       |
| SciPy                | —         | Wasserstein distance metric                         |
| tqdm                 | —         | Progress bars                                       |
| pytest               | ≥ 7.3     | Testing                                             |

---

## Summary of All Improvements

1. **D3PM discrete diffusion** — proper categorical noise for molecular features
2. **Graph Transformer with ring attention** — multi-scale message passing aware of ring structures
3. **FiLM property conditioning** — feature-wise modulation for targeted generation
4. **Classifier-free guidance** — steerable generation with condition dropout
5. **Multi-objective loss** — diffusion + valency + property + fragment losses
6. **Rich graph representation** — 23-dim node features, 8-dim edge features
7. **EMA + warmup-cosine LR** — stable training with moving average evaluation
8. **Comprehensive filtering** — Lipinski, glue-likeness, PAINS, reactivity, SA checks
9. **Diverse sampling** — variable sizes, temperatures, and oversampling
10. **BRICS fragment vocabulary** — learned prior on realistic substructures
11. **Full evaluation suite** — validity, uniqueness, novelty, diversity, QED, Wasserstein, glue similarity
12. **Visualization pipeline** — property distributions, scatter matrices, scaffold analysis
13. **Dual output formats** — CSV with properties + SDF with 3D coordinates
14. **Robust checkpointing** — periodic saves with full training state recovery
15. **Curated training data** — 154 molecular glue chemotypes across 8 chemical classes
16. **Unit test coverage** — chemistry, dataset, diffusion, and metrics tests
