"""
Configuration for Molecular Glue Diffusion Model
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ModelConfig:
    """Graph transformer architecture settings."""
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"  # "gelu" or "silu"

    # Node features: atom types
    num_atom_types: int = 10  # C, N, O, S, F, Cl, Br, P, I, other
    num_charges: int = 5      # -2, -1, 0, +1, +2
    num_hybridizations: int = 4  # sp, sp2, sp3, other

    # Edge features: bond types
    num_bond_types: int = 5   # none, single, double, triple, aromatic

    # Property conditioning
    num_properties: int = 8   # MW, logP, aromatic_rings, HBD, HBA, Fsp3, TPSA, SA


@dataclass
class DiffusionConfig:
    """Diffusion process settings."""
    num_timesteps: int = 500
    beta_schedule: str = "cosine"  # "linear" or "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02


@dataclass
class LossConfig:
    """Auxiliary loss weights."""
    lambda_valency: float = 0.5
    lambda_property: float = 0.3
    lambda_fragment: float = 0.1


@dataclass
class GuidanceConfig:
    """Classifier-free guidance settings."""
    guidance_scale: float = 2.0
    condition_dropout: float = 0.1  # Probability of dropping condition during training

    # Target property ranges for glue-like generation
    target_mw: Tuple[float, float] = (300.0, 450.0)
    target_logp: Tuple[float, float] = (1.5, 3.5)
    target_aromatic_rings: Tuple[int, int] = (2, 3)
    target_hbd_hba: Tuple[int, int] = (4, 8)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # EMA
    ema_decay: float = 0.999

    # Validation
    val_split: float = 0.1
    val_frequency: int = 5  # validate every N epochs

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 10


@dataclass
class GenerationConfig:
    """Molecule generation settings."""
    num_molecules: int = 1000
    max_atoms: int = 50
    min_atoms: int = 5
    batch_size: int = 100

    # Sampling
    temperature: float = 1.0


@dataclass
class MolecularConstraints:
    """Drug-like and glue-like property constraints."""
    # Molecular weight
    mw_min: float = 200.0
    mw_max: float = 500.0

    # Lipophilicity (tightened per spec)
    logp_min: float = 0.0
    logp_max: float = 4.0

    # Lipinski's Rule of Five
    hbd_max: int = 5   # H-bond donors
    hba_max: int = 10  # H-bond acceptors

    # Additional filters
    rotatable_bonds_max: int = 8
    tpsa_min: float = 20.0
    tpsa_max: float = 140.0

    # Synthetic accessibility (tightened per spec)
    sa_score_max: float = 4.5

    # Ring constraints
    min_rings: int = 1
    max_rings: int = 5
    min_aromatic_rings: int = 1

    # Rejection sampling
    max_heavy_atoms: int = 35
    min_fsp3: float = 0.2


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    constraints: MolecularConstraints = field(default_factory=MolecularConstraints)
    loss: LossConfig = field(default_factory=LossConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)

    # Paths
    data_dir: str = "data"
    output_dir: str = "output"

    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42


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

# Property names for conditioning
PROPERTY_NAMES = [
    'molecular_weight', 'logp', 'num_aromatic_rings',
    'hbd', 'hba', 'fraction_sp3', 'tpsa', 'sa_score'
]
