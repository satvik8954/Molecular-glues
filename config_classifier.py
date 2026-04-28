# config_classifier.py

from dataclasses import dataclass

@dataclass
class ClassifierConfig:
    """Configuration for classifier training."""
    
    
    hidden_dim: int = 128       
    num_layers: int = 3            
    num_heads: int = 8
    dropout: float = 0.3       
    
    # Training
    batch_size: int = 128       
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3    
    num_epochs: int = 50
    
    # Regularization
    label_smoothing: float = 0.05  # Smooth labels: y' = y * 0.95 + 0.025
    drop_edge_rate: float = 0.15   # Randomly drop 15% of edges during training
    early_stop_patience: int = 5  # Stop if val loss doesn't improve for 5 epochs
    
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
# Atom type vocabulary
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I', 'Other']
ATOM_TO_IDX = {atom: idx for idx, atom in enumerate(ATOM_TYPES)}

# Bond type vocabulary
BOND_TYPES = ['NONE', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
BOND_TO_IDX = {bond: idx for idx, bond in enumerate(BOND_TYPES)}

# Charge vocabulary
CHARGES = [-2, -1, 0, 1, 2]
CHARGE_TO_IDX = {charge: idx for idx, charge in enumerate(CHARGES)}

# Hybridization vocabulary
HYBRIDIZATIONS = ['SP', 'SP2', 'SP3', 'OTHER']
HYBRID_TO_IDX = {h: idx for idx, h in enumerate(HYBRIDIZATIONS)}
