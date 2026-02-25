# config_classifier.py

from dataclasses import dataclass

@dataclass
class ClassifierConfig:
    """Configuration for classifier training."""
    
    # Model
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    
    # Data
    num_workers: int = 4
    
    # Logging
    use_wandb: bool = True
    
    # Paths
    data_path: str = 'data/classifier_dataset_augmented.csv'
    checkpoint_dir: str = 'checkpoints'