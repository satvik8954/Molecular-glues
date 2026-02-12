"""
Molecule sampling from trained diffusion model.
"""
from typing import Optional, List, Dict, Tuple
import torch
from torch_geometric.data import Data

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config, GuidanceConfig, PROPERTY_NAMES
from model.diffusion import MolecularDiffusion


class MoleculeSampler:
    """
    High-level sampler for generating molecules from trained diffusion model.

    Supports:
    - Basic sampling with temperature control
    - Guided sampling with target properties
    - Diverse sampling with varied sizes and temperatures
    """

    def __init__(
        self,
        model: MolecularDiffusion,
        config: Config,
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

    def sample(
        self,
        num_molecules: int = 100,
        num_atoms: int = 20,
        temperature: float = 1.0,
    ) -> List[Data]:
        """
        Generate molecules using basic sampling.

        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Atoms per molecule
            temperature: Sampling temperature

        Returns:
            List of generated molecular graphs
        """
        batch_size = self.config.generation.batch_size
        all_molecules = []

        for i in range(0, num_molecules, batch_size):
            n = min(batch_size, num_molecules - i)
            molecules = self.model.sample(
                num_molecules=n,
                num_atoms=num_atoms,
                temperature=temperature,
            )
            all_molecules.extend(molecules)

        return all_molecules

    def guided_sample(
        self,
        num_molecules: int = 100,
        num_atoms: int = 20,
        temperature: float = 1.0,
        target_properties: Optional[Dict[str, float]] = None,
        guidance_scale: Optional[float] = None,
    ) -> List[Data]:
        """
        Generate molecules with property guidance.

        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Atoms per molecule
            temperature: Sampling temperature
            target_properties: Dict of {property_name: target_value}
            guidance_scale: Override default guidance scale

        Returns:
            List of generated molecular graphs
        """
        gs = guidance_scale or self.config.guidance.guidance_scale

        # Build target property tensor
        prop_tensor = None
        if target_properties:
            values = []
            for name in PROPERTY_NAMES:
                if name in target_properties:
                    values.append(target_properties[name])
                else:
                    values.append(0.0)  # Zero = no guidance for this property
            prop_tensor = torch.tensor(values, dtype=torch.float32)

        batch_size = self.config.generation.batch_size
        all_molecules = []

        for i in range(0, num_molecules, batch_size):
            n = min(batch_size, num_molecules - i)
            molecules = self.model.sample(
                num_molecules=n,
                num_atoms=num_atoms,
                temperature=temperature,
                target_properties=prop_tensor,
                guidance_scale=gs,
            )
            all_molecules.extend(molecules)

        return all_molecules

    def sample_diverse(
        self,
        num_molecules: int = 100,
        min_atoms: Optional[int] = None,
        max_atoms: Optional[int] = None,
        temperatures: Optional[List[float]] = None,
        target_properties: Optional[Dict[str, float]] = None,
        guidance_scale: Optional[float] = None,
    ) -> List[Data]:
        """
        Generate diverse molecules by varying sizes and temperatures.

        Args:
            num_molecules: Total molecules to generate
            min_atoms: Minimum atoms per molecule
            max_atoms: Maximum atoms per molecule
            temperatures: List of temperatures to try
            target_properties: Optional property targets for guidance
            guidance_scale: Override guidance scale

        Returns:
            List of generated molecular graphs
        """
        min_atoms = min_atoms or self.config.generation.min_atoms
        max_atoms = max_atoms or self.config.generation.max_atoms
        temperatures = temperatures or [0.7, 0.8, 0.9, 1.0, 1.1]

        gs = guidance_scale or self.config.guidance.guidance_scale
        prop_tensor = None
        if target_properties:
            values = [target_properties.get(name, 0.0) for name in PROPERTY_NAMES]
            prop_tensor = torch.tensor(values, dtype=torch.float32)

        all_molecules = []
        num_settings = len(temperatures) * (max_atoms - min_atoms + 1)
        per_setting = max(1, num_molecules // num_settings)

        import random
        atom_sizes = list(range(min_atoms, max_atoms + 1))

        while len(all_molecules) < num_molecules:
            num_atoms = random.choice(atom_sizes)
            temp = random.choice(temperatures)
            n = min(per_setting, num_molecules - len(all_molecules))

            molecules = self.model.sample(
                num_molecules=n,
                num_atoms=num_atoms,
                temperature=temp,
                target_properties=prop_tensor,
                guidance_scale=gs,
            )
            all_molecules.extend(molecules)

        return all_molecules[:num_molecules]

    @staticmethod
    def get_glue_targets() -> Dict[str, float]:
        """
        Return normalized target properties for glue-like molecule generation.
        Uses midpoints of the spec-defined target ranges.
        """
        return {
            'molecular_weight': 375.0 / 500.0,     # MW 300-450, mid=375
            'logp': (2.5 + 2.0) / 8.0,             # LogP 1.5-3.5, mid=2.5
            'num_aromatic_rings': 2.5 / 5.0,        # 2-3 aromatic rings, mid=2.5
            'hbd': 2.0 / 5.0,                       # HBD ~2
            'hba': 4.0 / 10.0,                      # HBA ~4
            'fraction_sp3': 0.35,                    # Fsp3 ~0.35
            'tpsa': 70.0 / 140.0,                   # TPSA ~70
            'sa_score': 3.0 / 10.0,                 # SA ~3
        }
