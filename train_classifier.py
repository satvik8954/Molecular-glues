# train_classifier.py
"""
Main script to train the molecular glue classifier.
"""

import argparse
from pathlib import Path

from config_classifier import ClassifierConfig
from data.classifier_dataset import MolecularGlueClassifierDataset
from model.classifier import MolecularGlueClassifier
from train.classifier_trainer import ClassifierTrainer


def main(args):
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
    
    print("=" * 80)
    print("MOLECULAR GLUE CLASSIFIER - TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print()
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Datasets
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
    print("\nInitializing model...")
    model = MolecularGlueClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
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
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--data_path', type=str, default=None, help='Path to CSV dataset')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    args = parser.parse_args()
    main(args)