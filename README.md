# Molecular Glue Diffusion Model

A graph-based discrete diffusion model for generating small molecules with molecular glue-like properties.

## Features

- **Discrete Diffusion**: D3PM-style categorical noise for atom/bond types
- **Graph Transformer**: Multi-head attention with edge features and time conditioning
- **Chemical Validity**: RDKit-based validation and filtering
- **Drug-Likeness**: Lipinski Rule of Five and glue-specific property filters

## Installation

```bash
pip install -r requirements.txt
```

Requires:
- Python 3.8+
- PyTorch 2.0+
- RDKit 2023.03+
- PyTorch Geometric 2.3+

## Quick Start

### Training
```bash
# Quick test (5 epochs, small model)
python train_model.py --quick_test

# Full training
python train_model.py --epochs 100 --batch_size 64
```

### Generation
```bash
python generate_molecules.py \
    --checkpoint checkpoints/best_model.pt \
    --n_molecules 1000 \
    --output generated.csv
```

### Analysis
```bash
python eval/analyze.py \
    --input generated.csv \
    --training_data data/glue_chemotypes.csv
```

## Project Structure

```
├── config.py              # Configuration and hyperparameters
├── train_model.py         # Main training script
├── generate_molecules.py  # Molecule generation script
├── data/                  # Dataset and graph conversion
├── model/                 # Diffusion model components
├── train/                 # Training utilities
├── generate/              # Generation and post-processing
├── eval/                  # Metrics and analysis
└── tests/                 # Unit tests
```

## Architecture

The model uses:
1. **Molecular Graph Representation**: Atoms as nodes, bonds as edges
2. **Discrete Noise Scheduler**: Categorical corruption towards uniform distribution
3. **Graph Transformer**: Denoises graphs conditioned on timestep
4. **Reverse Sampling**: Iteratively denoise from random noise to valid molecules

## Molecular Glue Properties

Target molecules with:
- MW: 200-500 Da
- Aromatic/heterocyclic scaffolds
- Mixed polarity (TPSA 20-120)
- Multiple H-bond interaction points
- Low rotatable bonds (<8)

## License

MIT
