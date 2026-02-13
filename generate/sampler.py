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
        num_sampling_steps: Optional[int] = None,
    ) -> List[Data]:
        """
        Generate molecules using basic sampling.

        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Atoms per molecule
            temperature: Sampling temperature
            num_sampling_steps: Fewer steps = faster (default: all timesteps)

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
                num_sampling_steps=num_sampling_steps,
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
        num_sampling_steps: Optional[int] = None,
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
                num_sampling_steps=num_sampling_steps,
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
        num_sampling_steps: Optional[int] = None,
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
                num_sampling_steps=num_sampling_steps,
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

    # ------------------------------------------------------------------
    # Rejection sampling
    # ------------------------------------------------------------------

    def generate_with_rejection(
        self,
        n_molecules: int,
        max_attempts_per_molecule: int = 10,
        num_atoms: int = 20,
        temperature: float = 1.0,
        target_properties: Optional[Dict[str, float]] = None,
        guidance_scale: Optional[float] = None,
        num_sampling_steps: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, int]]:
        """Generate molecules with hard chemical validity checks.

        Repeats generation until *n_molecules* valid SMILES are collected,
        or until a maximum number of total attempts is exceeded.

        Returns:
            (valid_smiles_list, rejection_stats_dict)
        """
        from data.molecular_graph import graph_to_smiles
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        valid_molecules: List[str] = []
        stats = {
            'attempts': 0,
            'invalid_graph': 0,
            'sanitization_fail': 0,
            'property_reject': 0,
            'success': 0,
        }

        max_total_attempts = n_molecules * max_attempts_per_molecule

        while len(valid_molecules) < n_molecules and stats['attempts'] < max_total_attempts:
            # Generate a batch
            remaining = n_molecules - len(valid_molecules)
            batch_size = min(remaining * 2, self.config.generation.batch_size)

            if target_properties is not None:
                graphs = self.guided_sample(
                    num_molecules=batch_size,
                    num_atoms=num_atoms,
                    temperature=temperature,
                    target_properties=target_properties,
                    guidance_scale=guidance_scale,
                    num_sampling_steps=num_sampling_steps,
                )
            else:
                graphs = self.sample(
                    num_molecules=batch_size,
                    num_atoms=num_atoms,
                    temperature=temperature,
                    num_sampling_steps=num_sampling_steps,
                )

            for graph in graphs:
                stats['attempts'] += 1
                if len(valid_molecules) >= n_molecules:
                    break

                smiles = graph_to_smiles(graph)
                if smiles is None:
                    stats['invalid_graph'] += 1
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    stats['invalid_graph'] += 1
                    continue

                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    stats['sanitization_fail'] += 1
                    continue

                if not self.check_property_bounds(mol):
                    stats['property_reject'] += 1
                    continue

                canonical = Chem.MolToSmiles(mol, canonical=True)
                valid_molecules.append(canonical)
                stats['success'] += 1

        # Report
        total = max(stats['attempts'], 1)
        print("\n=== Rejection Sampling Statistics ===")
        print(f"Total attempts:        {stats['attempts']}")
        print(f"Success rate:          {stats['success']/total*100:.1f}%")
        print(f"Invalid graph/SMILES:  {stats['invalid_graph']/total*100:.1f}%")
        print(f"Sanitization failures: {stats['sanitization_fail']/total*100:.1f}%")
        print(f"Property rejections:   {stats['property_reject']/total*100:.1f}%")
        print(f"Valid molecules:       {len(valid_molecules)}/{n_molecules}")

        return valid_molecules, stats

    @staticmethod
    def check_property_bounds(mol) -> bool:
        """Hard bounds for molecular properties. Returns True if acceptable."""
        from rdkit.Chem import Descriptors

        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        # Reject chemically impossible / non-drug-like
        if hbd == 0 and hba > 15:
            return False
        if hbd > 10:
            return False
        if hba > 15:
            return False
        if tpsa > 200:
            return False
        if aromatic_rings == 0:
            return False
        if heavy_atoms < 10 or heavy_atoms > 40:
            return False
        if not (200 <= mw <= 550):
            return False
        if not (-1 <= logp <= 6):
            return False

        return True
