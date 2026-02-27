# config_classifier.py

from dataclasses import dataclass

@dataclass
class ClassifierConfig:
    """Configuration for classifier training."""
    
    # Model — reduced capacity to prevent overfitting
    hidden_dim: int = 128          # Was 256 — cuts params ~4× (7M → 1.8M)
    num_layers: int = 3            # Was 4 — slightly fewer layers
    num_heads: int = 8
    dropout: float = 0.3           # Was 0.1 — stronger regularization
    
    # Training
    batch_size: int = 64           # Use 64-128 on GPU, 32 on CPU
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3     # Was 1e-5 — 100× stronger L2 regularization
    num_epochs: int = 50
    
    # Regularization
    label_smoothing: float = 0.05  # Smooth labels: y' = y * 0.95 + 0.025
    drop_edge_rate: float = 0.15   # Randomly drop 15% of edges during training
    early_stop_patience: int = 10  # Stop if val loss doesn't improve for 10 epochs
    
    # Augmentation
    use_augmentation: bool = True  # Graph augmentation (node masking + edge dropping)
    augment_prob: float = 0.5      # Probability of augmenting each sample
    
    # Loss
    use_focal_loss: bool = False   # Enable for hard-example focusing
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Training enhancements
    gradient_accumulation_steps: int = 2  # Effective batch = batch_size * 2
    use_cosine_schedule: bool = True      # OneCycleLR with warmup
    
    # Data (scaffold-split CSV files from create_classifier_data.py)
    train_path: str = 'data/classifier_train.csv'
    val_path: str = 'data/classifier_val.csv'
    test_path: str = 'data/classifier_test.csv'
    num_workers: int = 0  # Must be 0: pre-cached graphs cause shared memory OOM in workers
    
    # Logging
    use_wandb: bool = False
    project_name: str = 'molecular-glue-classifier'
    
    # Paths
    checkpoint_dir: str = 'checkpoints/classifier'