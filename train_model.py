"""
Main script for training the molecular diffusion model.
"""
import argparse
import torch
from torch.utils.data import DataLoader

import sys
import os

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config, ModelConfig, DiffusionConfig, TrainingConfig
from data.dataset import MolecularGlueDataset, create_train_val_split
from model.diffusion import MolecularDiffusion
from train.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Molecular Diffusion Model')

    # Data
    parser.add_argument('--data_path', type=str, default='data/glue_chemotypes.csv',
                        help='Path to training data CSV')
    parser.add_argument('--smiles_column', type=str, default='smiles',
                        help='SMILES column name')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=500)
    parser.add_argument('--beta_schedule', type=str, default='cosine')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.1)

    # Loss weights
    parser.add_argument('--lambda_valency', type=float, default=0.5)
    parser.add_argument('--lambda_property', type=float, default=0.3)
    parser.add_argument('--lambda_fragment', type=float, default=0.1)

    # Guidance
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    parser.add_argument('--condition_dropout', type=float, default=0.1)

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')

    # Quick test
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with small model/data')

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Quick test overrides
    if args.quick_test:
        args.hidden_dim = 64
        args.num_layers = 2
        args.num_heads = 4
        args.num_timesteps = 50
        args.epochs = 5
        args.batch_size = 16
        args.warmup_steps = 10

    # Build config
    config = Config()
    config.model.hidden_dim = args.hidden_dim
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.model.dropout = args.dropout
    config.diffusion.num_timesteps = args.num_timesteps
    config.diffusion.beta_schedule = args.beta_schedule
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.warmup_steps = args.warmup_steps
    config.training.ema_decay = args.ema_decay
    config.training.gradient_clip = args.gradient_clip
    config.training.val_split = args.val_split
    config.loss.lambda_valency = args.lambda_valency
    config.loss.lambda_property = args.lambda_property
    config.loss.lambda_fragment = args.lambda_fragment
    config.guidance.guidance_scale = args.guidance_scale
    config.guidance.condition_dropout = args.condition_dropout
    config.device = device
    config.seed = args.seed

    print(f"Device: {device}")
    print(f"Config: hidden={args.hidden_dim}, layers={args.num_layers}, "
          f"heads={args.num_heads}, timesteps={args.num_timesteps}")
    print(f"EMA decay: {args.ema_decay}, Warmup steps: {args.warmup_steps}")

    # Load dataset
    print(f"\nLoading data from {args.data_path}...")
    dataset = MolecularGlueDataset(
        data_path=args.data_path,
        smiles_column=args.smiles_column,
        filter_drug_like=False,  # Don't over-filter small datasets
        filter_glue_like=False,
        max_atoms=50,
    )

    # Split
    train_dataset, val_dataset = create_train_val_split(
        dataset, val_ratio=args.val_split, seed=args.seed
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=MolecularGlueDataset.collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MolecularGlueDataset.collate_fn,
        num_workers=0,
    )

    print(f"Train: {len(train_dataset)} molecules, Val: {len(val_dataset)} molecules")

    # Create model
    model = MolecularDiffusion(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader)

    # Resume from checkpoint
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        print(f"Resumed from {args.checkpoint}")

    # Train
    history = trainer.train(num_epochs=args.epochs)

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
