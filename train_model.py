"""
Main training script for molecular glue diffusion model.

Usage:
    python train_model.py --epochs 100 --batch_size 64
    python train_model.py --quick_test  # Fast test run
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from config import Config
from data.dataset import MolecularGlueDataset, create_train_val_split
from model.diffusion import MolecularDiffusion
from train.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train molecular glue diffusion model')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data/glue_chemotypes.csv',
                        help='Path to training data')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--diffusion_steps', type=int, default=500, help='Number of diffusion steps')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, or auto)')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test (5 epochs, small model)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create config
    config = Config()
    config.device = device
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.checkpoint_dir = args.checkpoint_dir
    config.model.hidden_dim = args.hidden_dim
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.diffusion.num_timesteps = args.diffusion_steps
    
    # Quick test mode
    if args.quick_test:
        print("\n=== QUICK TEST MODE ===")
        config.training.epochs = 5
        config.model.hidden_dim = 64
        config.model.num_layers = 2
        config.model.num_heads = 4
        config.diffusion.num_timesteps = 100
        config.training.checkpoint_dir = 'checkpoints/quick_test'
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_path}")
    dataset = MolecularGlueDataset(
        data_path=args.data_path,
        filter_drug_like=True,
        filter_glue_like=False,
        cache_graphs=True,
        max_atoms=50,
    )
    
    # Split into train/val
    train_dataset, val_dataset = create_train_val_split(
        dataset, 
        val_ratio=config.training.val_split,
        seed=config.seed
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    print("\nCreating model...")
    model = MolecularDiffusion(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Quick generation test
    print("\nTesting generation...")
    model.eval()
    test_mols = model.sample(num_molecules=5, num_atoms=15)
    print(f"Generated {len(test_mols)} test molecules")
    
    from data.molecular_graph import graph_to_smiles
    for i, mol in enumerate(test_mols):
        smiles = graph_to_smiles(mol)
        print(f"  {i+1}: {smiles}")


if __name__ == '__main__':
    main()
