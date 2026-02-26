# train_classifier.py
"""
Main script to train the molecular glue classifier.

Usage:
  Single GPU:   python train_classifier.py --epochs 50
  Multi-GPU:    torchrun --nproc_per_node=NUM_GPUS train_classifier.py --epochs 50
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from config_classifier import ClassifierConfig
from data.classifier_dataset import MolecularGlueClassifierDataset
from model.classifier import MolecularGlueClassifier
from train.classifier_trainer import ClassifierTrainer


def setup_distributed():
    """Initialize DDP process group if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        return rank
    return 0


def main(args):
    # Initialize distributed (no-op if not using torchrun)
    rank = setup_distributed()
    is_main = (rank == 0)
    
    # Configuration
    config = ClassifierConfig()
    
    # Override with command line args
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.lr is not None:
        config.learning_rate = args.lr
    
    if is_main:
        print("=" * 80)
        print("MOLECULAR GLUE CLASSIFIER - TRAINING")
        print("=" * 80)
        print(f"\nConfiguration:")
        for key, value in vars(config).items():
            print(f"  {key}: {value}")
        print()
    
    # Create checkpoint directory
    if is_main:
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Datasets
    if is_main:
        print("Loading datasets...")
    train_dataset = MolecularGlueClassifierDataset(
        csv_file=config.data_path,
        split='train',
        val_split=config.val_split
    )
    
    val_dataset = MolecularGlueClassifierDataset(
        csv_file=config.data_path,
        split='val',
        val_split=config.val_split
    )
    
    # Model
    if is_main:
        print("\nInitializing model...")
    model = MolecularGlueClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    )
    if is_main:
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer (handles DDP/DataParallel automatically)
    trainer = ClassifierTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    trainer.train(num_epochs=config.num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train molecular glue classifier')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--data_path', type=str, default=None, help='Path to CSV dataset')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    args = parser.parse_args()
    main(args)