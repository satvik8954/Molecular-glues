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
    batch_size: int = 64  # Use 64-128 on GPU, 32 on CPU
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    
    # Data
    num_workers: int = 4  # Set to 0 on Windows, 4+ on Linux/DGX
    val_split: float = 0.2
    data_path: str = 'data/classifier_dataset.csv'
    
    # Logging
    use_wandb: bool = False
    project_name: str = 'molecular-glue-classifier'
    
    # Paths
    checkpoint_dir: str = 'checkpoints/classifier'