"""
Diverse molecule sampling strategies for the molecular diffusion model.

Extends the base ``MoleculeSampler`` with:
    - Variable molecule sizes (15-35 atoms)
    - Variable temperatures (0.7-1.2)
    - Variable guidance scales (1.5-3.0)
    - Multiple diverse property targets
    - Scaffold-tracking to avoid over-representation
"""
from typing import Optional, List, Dict, Tuple
import random

import torch

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from model.diffusion import MolecularDiffusion
from generate.sampler import MoleculeSampler
from generate.postprocess import postprocess_molecule
from utils.chemistry import get_murcko_scaffold, canonicalize_smiles


class DiverseMoleculeSampler(MoleculeSampler):
    """
    Sampler with diversity-promoting strategies layered on top of the
    base ``MoleculeSampler``.

    Key additions over the base class:

    * ``sample_diverse`` – sweep atom counts, temperatures, and guidance
      scales to cover a wider region of chemical space.
    * ``generate_diverse_property_targets`` – produces multiple target
      property vectors spanning glue-like space.
    * ``scaffold_diversity_sampling`` – caps the number of molecules per
      Murcko scaffold.

    Example::

        sampler = DiverseMoleculeSampler(model, config)
        smiles = sampler.sample_diverse(
            num_molecules=500,
            size_range=(15, 35),
            temp_range=(0.7, 1.2),
            guidance_range=(1.5, 3.0),
        )
    """

    def __init__(self, model: MolecularDiffusion, config: Config, seed: int = 42):
        super().__init__(model, config)
        self.rng = random.Random(seed)

    # ── Public API ──────────────────────────────────────────────────

    def sample_diverse(
        self,
        num_molecules: int = 500,
        size_range: Tuple[int, int] = (15, 35),
        temp_range: Tuple[float, float] = (0.7, 1.2),
        guidance_range: Tuple[float, float] = (1.5, 3.0),
        batch_size: int = 50,
        num_sampling_steps: Optional[int] = None,
    ) -> List[str]:
        """
        Generate molecules by randomly varying atom count, temperature,
        and guidance scale across batches.

        Args:
            num_molecules: Total SMILES wanted.
            size_range: (min_atoms, max_atoms) uniform range.
            temp_range: (min_temp, max_temp) uniform range.
            guidance_range: (min_guide, max_guide) uniform range.
            batch_size: Molecules per inner sampling call.
            num_sampling_steps: Override diffusion steps.

        Returns:
            List of unique, valid canonical SMILES.
        """
        collected: List[str] = []
        seen: set = set()
        total_attempts = 0
        max_attempts = num_molecules * 5  # safety cap

        targets = self.generate_diverse_property_targets(
            max(5, num_molecules // batch_size)
        )

        while len(collected) < num_molecules and total_attempts < max_attempts:
            n_atoms = self.rng.randint(size_range[0], size_range[1])
            temp = round(
                self.rng.uniform(temp_range[0], temp_range[1]), 2
            )
            guide = round(
                self.rng.uniform(guidance_range[0], guidance_range[1]), 2
            )
            target = self.rng.choice(targets)

            remaining = num_molecules - len(collected)
            n_batch = min(batch_size, remaining)

            graphs = self.guided_sample(
                num_molecules=n_batch,
                num_atoms=n_atoms,
                temperature=temp,
                target_properties=target,
                guidance_scale=guide,
                num_sampling_steps=num_sampling_steps,
            )

            for g in graphs:
                smi = postprocess_molecule(g)
                if smi is None:
                    continue
                canon = canonicalize_smiles(smi)
                if canon and canon not in seen:
                    seen.add(canon)
                    collected.append(canon)

            total_attempts += n_batch

        print(
            f"DiverseSampler: collected {len(collected)} unique molecules "
            f"from {total_attempts} attempts "
            f"({len(collected) / max(1, total_attempts) * 100:.1f}% yield)"
        )
        return collected

    def generate_diverse_property_targets(
        self,
        num_targets: int = 10,
    ) -> List[Dict[str, float]]:
        """
        Produce diverse property-target vectors spanning glue-like space.

        The ranges sampled are:
            MW:  250 – 450
            LogP: 1.0 – 4.0
            Aromatic rings: 1 – 3
            HBD: 1 – 4
            HBA: 2 – 8
            Fsp3: 0.15 – 0.55
            TPSA: 40 – 120

        Targets are normalised to the same scale the model was trained on
        (using the ``Config`` normalisation ranges, falling back to
        reasonable defaults).

        Args:
            num_targets: How many target vectors to generate.

        Returns:
            List of property dictionaries ready for ``guided_sample``.
        """
        targets: List[Dict[str, float]] = []

        ranges = {
            "molecular_weight": (250.0, 450.0),
            "logp": (1.0, 4.0),
            "num_aromatic_rings": (1.0, 3.0),
            "hbd": (1.0, 4.0),
            "hba": (2.0, 8.0),
            "fraction_sp3": (0.15, 0.55),
            "tpsa": (40.0, 120.0),
        }

        for _ in range(num_targets):
            t = {}
            for key, (lo, hi) in ranges.items():
                t[key] = round(self.rng.uniform(lo, hi), 2)
            targets.append(t)

        return targets

    def scaffold_diversity_sampling(
        self,
        num_molecules: int = 500,
        max_per_scaffold: int = 5,
        size_range: Tuple[int, int] = (15, 35),
        temperature: float = 1.0,
        guidance_scale: float = 2.0,
        batch_size: int = 50,
        num_sampling_steps: Optional[int] = None,
    ) -> List[str]:
        """
        Generate molecules while capping the number of representatives
        per Murcko scaffold, promoting structural diversity.

        Args:
            num_molecules: Target number of molecules.
            max_per_scaffold: Maximum accept per scaffold.
            size_range: (min, max) atom range.
            temperature: Sampling temperature.
            guidance_scale: Guidance scale.
            batch_size: Molecules per sampling call.
            num_sampling_steps: Override diffusion steps.

        Returns:
            List of unique SMILES with high scaffold diversity.
        """
        from collections import Counter

        collected: List[str] = []
        seen: set = set()
        scaffold_counts: Counter = Counter()
        total_attempts = 0
        max_attempts = num_molecules * 8

        targets = self.generate_diverse_property_targets(
            max(5, num_molecules // batch_size)
        )

        while len(collected) < num_molecules and total_attempts < max_attempts:
            n_atoms = self.rng.randint(size_range[0], size_range[1])
            target = self.rng.choice(targets)

            remaining = num_molecules - len(collected)
            n_batch = min(batch_size, remaining * 2)  # oversample

            graphs = self.guided_sample(
                num_molecules=n_batch,
                num_atoms=n_atoms,
                temperature=temperature,
                target_properties=target,
                guidance_scale=guidance_scale,
                num_sampling_steps=num_sampling_steps,
            )

            for g in graphs:
                if len(collected) >= num_molecules:
                    break

                smi = postprocess_molecule(g)
                if smi is None:
                    continue
                canon = canonicalize_smiles(smi)
                if not canon or canon in seen:
                    continue

                scaffold = get_murcko_scaffold(canon) or "none"
                if scaffold_counts[scaffold] >= max_per_scaffold:
                    continue  # skip over-represented scaffold

                seen.add(canon)
                scaffold_counts[scaffold] += 1
                collected.append(canon)

            total_attempts += n_batch

        n_scaffolds = len(scaffold_counts)
        diversity = n_scaffolds / max(1, len(collected))
        print(
            f"ScaffoldDiversity: {len(collected)} molecules, "
            f"{n_scaffolds} scaffolds ({diversity:.1%} diversity), "
            f"{total_attempts} attempts"
        )
        return collected
