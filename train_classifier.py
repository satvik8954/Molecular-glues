# train_classifier.py

import torch
from config_classifier import ClassifierConfig
from data.classifier_dataset import MolecularGlueClassifierDataset
from model.classifier import MolecularGlueClassifier
from train.classifier_trainer import ClassifierTrainer

def main():
    # Configuration
    config = ClassifierConfig()
    
    print("="*80)
    print("MOLECULAR GLUE CLASSIFIER - TRAINING")
    print("="*80)
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = MolecularGlueClassifierDataset(
        csv_file=config.data_path,
        split='train',
        val_split=0.2
    )
    
    val_dataset = MolecularGlueClassifierDataset(
        csv_file=config.data_path,
        split='val',
        val_split=0.2
    )
    
    # Model
    print("\nInitializing model...")
    model = MolecularGlueClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    )
    
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
    main()