"""
Fragment vocabulary for fragment frequency loss.

Pre-computes fragment frequencies from training data using BRICS decomposition
and provides scoring functions for generated molecules.
"""
import json
import os
from typing import Dict, List, Optional
from collections import Counter
import math

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem


class FragmentVocab:
    """
    Fragment frequency vocabulary from training data.

    Uses BRICS decomposition to extract common drug-like fragments
    and scores generated molecules based on fragment frequencies.
    """

    def __init__(self, fragment_counts: Optional[Dict[str, int]] = None):
        """
        Args:
            fragment_counts: Pre-computed fragment counts {SMILES: count}
        """
        self.fragment_counts = fragment_counts or {}
        self.total_fragments = sum(self.fragment_counts.values()) if self.fragment_counts else 0

    @classmethod
    def from_smiles_list(cls, smiles_list: List[str], min_count: int = 2) -> 'FragmentVocab':
        """
        Build fragment vocabulary from a list of SMILES.

        Args:
            smiles_list: List of training SMILES
            min_count: Minimum fragment occurrences to include

        Returns:
            FragmentVocab instance
        """
        fragment_counter = Counter()

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # BRICS decomposition
                fragments = BRICS.BRICSDecompose(mol)
                for frag in fragments:
                    # Clean fragment SMILES
                    frag_mol = Chem.MolFromSmiles(frag)
                    if frag_mol is not None:
                        clean_frag = Chem.MolToSmiles(frag_mol, canonical=True)
                        fragment_counter[clean_frag] += 1
            except Exception:
                continue

        # Filter by min_count
        filtered = {
            frag: count for frag, count in fragment_counter.items()
            if count >= min_count
        }

        return cls(fragment_counts=filtered)

    def score_molecule(self, mol_or_smiles) -> float:
        """
        Score a molecule based on how common its fragments are.

        Higher score = more common/realistic fragments.
        Returns negative log probability (lower is better).

        Args:
            mol_or_smiles: RDKit mol or SMILES string

        Returns:
            Fragment score (lower = more common fragments)
        """
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None or self.total_fragments == 0:
            return 10.0  # High penalty for invalid

        try:
            fragments = BRICS.BRICSDecompose(mol)
            if not fragments:
                return 5.0  # Medium penalty for no fragments

            scores = []
            for frag in fragments:
                frag_mol = Chem.MolFromSmiles(frag)
                if frag_mol is None:
                    continue
                clean_frag = Chem.MolToSmiles(frag_mol, canonical=True)

                count = self.fragment_counts.get(clean_frag, 0)
                if count > 0:
                    prob = count / self.total_fragments
                    scores.append(-math.log(prob))
                else:
                    scores.append(10.0)  # Unseen fragment penalty

            if not scores:
                return 5.0

            return sum(scores) / len(scores)

        except Exception:
            return 10.0

    def save(self, path: str):
        """Save fragment vocabulary to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'fragment_counts': self.fragment_counts,
                'total_fragments': self.total_fragments,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FragmentVocab':
        """Load fragment vocabulary from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(fragment_counts=data['fragment_counts'])

    def __len__(self) -> int:
        return len(self.fragment_counts)

    def __repr__(self) -> str:
        return f"FragmentVocab(n_fragments={len(self)}, total_count={self.total_fragments})"
