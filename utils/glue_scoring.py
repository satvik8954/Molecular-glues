"""
Glue-likeness scoring for molecular glue candidates.

Scores molecules from 0 to 100 based on structural and physicochemical
features characteristic of known molecular glues (IMiDs, PROTACs warheads,
aryl-sulfonamides, etc.).
"""
from typing import Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.chemistry import get_molecular_properties, is_valid_molecule
from utils.filters import (
    has_reactive_groups,
    passes_pains_filter,
    calculate_sa_score,
)

# ── SMARTS definitions ──────────────────────────────────────────────

# Warhead motifs found in clinically relevant molecular glues
WARHEAD_SMARTS = {
    "glutarimide": "[#7]1C(=O)[#6][#6]C(=O)1",
    "phthalimide": "[#7]1C(=O)c2ccccc2C1=O",
    "succinimide": "[#7]1C(=O)[#6]C(=O)1",
    "sulfonamide": "[#7]S(=O)(=O)c",
    "hydantoin": "O=C1NC(=O)NC1",
    "barbiturate": "O=C1NC(=O)NC(=O)C1",
}

# Privileged scaffolds from known glue chemotypes
PRIVILEGED_SCAFFOLD_SMARTS = {
    "isoindolinone": "O=C1NCc2ccccc12",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "benzothiazole": "c1ccc2ncsc2c1",
    "benzoxazole": "c1ccc2ncoc2c1",
    "quinazolinone": "O=c1[nH]cnc2ccccc12",
    "indole": "c1ccc2[nH]ccc2c1",
    "dihydroquinazolinone": "O=C1NC(c2ccccc2N1)c1ccccc1",
    "phenyl_piperazine": "c1ccc(N2CCNCC2)cc1",
    "phenyl_morpholine": "c1ccc(N2CCOCC2)cc1",
    "coumarin": "O=c1ccc2ccccc2o1",
}


def has_glue_warhead(mol_or_smiles) -> Tuple[bool, List[str]]:
    """
    Check if a molecule contains known molecular-glue warhead motifs.

    Args:
        mol_or_smiles: RDKit Mol object or SMILES string.

    Returns:
        Tuple of (has_warhead, list_of_matched_warhead_names).
    """
    mol = _to_mol(mol_or_smiles)
    if mol is None:
        return False, []

    matched = []
    for name, smarts in WARHEAD_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            matched.append(name)
    return len(matched) > 0, matched


def check_privileged_scaffolds(mol_or_smiles) -> Tuple[bool, List[str]]:
    """
    Check if a molecule contains privileged scaffolds found in molecular glues.

    Args:
        mol_or_smiles: RDKit Mol object or SMILES string.

    Returns:
        Tuple of (has_scaffold, list_of_matched_scaffold_names).
    """
    mol = _to_mol(mol_or_smiles)
    if mol is None:
        return False, []

    matched = []
    for name, smarts in PRIVILEGED_SCAFFOLD_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            matched.append(name)
    return len(matched) > 0, matched


def compute_glue_likeness_score(mol_or_smiles) -> int:
    """
    Score a molecule for molecular-glue-likeness on a 0–100 scale.

    Scoring breakdown (positive contributions):
        - Has warhead motif (glutarimide/phthalimide/sulfonamide …)  : +30
        - Aromatic rings 2–3                                         : +20
        - Rigidity (≤3 rotatable bonds)                              : +15
        - Balanced polarity (HBD 1–4, HBA 3–8)                      : +15
        - Proper size (MW 250–450)                                   : +10
        - Proper lipophilicity (LogP 1.5–3.5)                        : +10

    Penalties:
        - Reactive groups              : −30
        - PAINS alerts                 : −20
        - Poor permeability (TPSA>140) : −15

    Args:
        mol_or_smiles: RDKit Mol object or SMILES string.

    Returns:
        Integer score clamped to [0, 100].

    Example::

        >>> from utils.glue_scoring import compute_glue_likeness_score
        >>> compute_glue_likeness_score("O=C1NC(=O)c2ccccc12")  # phthalimide
        45
    """
    mol = _to_mol(mol_or_smiles)
    if mol is None:
        return 0

    props = get_molecular_properties(mol)
    if props is None:
        return 0

    score = 0

    # ── Positive contributions ──────────────────────────────────────
    # Warhead motif (+30)
    has_wh, _ = has_glue_warhead(mol)
    if has_wh:
        score += 30

    # Aromatic rings 2-3 (+20), partial for 1 or 4 (+10)
    arom = props.get("num_aromatic_rings", 0)
    if 2 <= arom <= 3:
        score += 20
    elif arom == 1 or arom == 4:
        score += 10

    # Rigidity: ≤3 rotatable bonds (+15), 4-5 (+8)
    rot = props.get("rotatable_bonds", 99)
    if rot <= 3:
        score += 15
    elif rot <= 5:
        score += 8

    # Balanced polarity: HBD 1-4 and HBA 3-8 (+15)
    hbd = props.get("hbd", 0)
    hba = props.get("hba", 0)
    if 1 <= hbd <= 4 and 3 <= hba <= 8:
        score += 15
    elif 0 <= hbd <= 5 and 1 <= hba <= 10:
        score += 7  # partial credit

    # Proper size: MW 250-450 (+10)
    mw = props.get("molecular_weight", 0)
    if 250 <= mw <= 450:
        score += 10
    elif 200 <= mw <= 500:
        score += 5

    # Proper LogP: 1.5-3.5 (+10)
    logp = props.get("logp", -999)
    if 1.5 <= logp <= 3.5:
        score += 10
    elif 0.0 <= logp <= 5.0:
        score += 5

    # ── Penalties ───────────────────────────────────────────────────
    if has_reactive_groups(mol):
        score -= 30

    if not passes_pains_filter(mol):
        score -= 20

    tpsa = props.get("tpsa", 0)
    if tpsa > 140:
        score -= 15

    return max(0, min(100, score))


def rank_molecules_by_glue_score(smiles_list: List[str]) -> List[Tuple[str, int]]:
    """
    Rank a list of SMILES by glue-likeness score (descending).

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        List of (smiles, score) tuples sorted by score descending.
    """
    scored = []
    for smi in smiles_list:
        s = compute_glue_likeness_score(smi)
        scored.append((smi, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Private helpers ─────────────────────────────────────────────────

def _to_mol(mol_or_smiles):
    """Convert SMILES string or Mol to RDKit Mol."""
    if mol_or_smiles is None:
        return None
    if isinstance(mol_or_smiles, str):
        return Chem.MolFromSmiles(mol_or_smiles)
    if hasattr(mol_or_smiles, "GetNumAtoms"):
        return mol_or_smiles
    return None
