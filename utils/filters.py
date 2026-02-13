"""
Molecular filters for drug-likeness, glue-likeness, PAINS, and synthetic accessibility.
"""
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
import math


def is_drug_like(mol_or_smiles) -> bool:
    """
    Check if molecule passes Lipinski's Rule of Five.

    Args:
        mol_or_smiles: RDKit Mol or SMILES

    Returns:
        True if drug-like
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return False

        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        return violations <= 1
    except Exception:
        return False


def is_glue_like(mol_or_smiles) -> bool:
    """
    Check if molecule has molecular glue-like properties.

    Criteria: MW 200-500, LogP 0-4, ≥1 aromatic ring, ≤8 rotatable bonds,
    TPSA 20-140, heavy atoms ≤35.

    Args:
        mol_or_smiles: RDKit Mol or SMILES

    Returns:
        True if glue-like
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return False

        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        if mw < 200 or mw > 500:
            return False
        if logp < 0.0 or logp > 4.0:
            return False
        if num_aromatic_rings < 1:
            return False
        if rotatable_bonds > 8:
            return False
        if tpsa < 20.0 or tpsa > 140.0:
            return False
        if heavy_atoms > 35:
            return False

        return True
    except Exception:
        return False


def is_glue_target_range(mol_or_smiles) -> bool:
    """
    Check if molecule falls within the guided target ranges for glue-like generation.

    Tighter criteria: MW 300-450, LogP 1.5-3.5, 2-3 aromatic rings,
    HBD+HBA 4-8, Fsp3 > 0.2.
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return False

        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        # Fsp3
        num_sp3 = sum(1 for a in mol.GetAtoms()
                      if a.GetAtomicNum() == 6 and
                      a.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
        num_carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        fsp3 = num_sp3 / max(1, num_carbons)

        if mw < 300 or mw > 450:
            return False
        if logp < 1.5 or logp > 3.5:
            return False
        if num_aromatic_rings < 2 or num_aromatic_rings > 3:
            return False
        if (hbd + hba) < 4 or (hbd + hba) > 8:
            return False
        if fsp3 < 0.2:
            return False

        return True
    except Exception:
        return False


def has_reactive_groups(mol_or_smiles) -> bool:
    """
    Check if molecule contains reactive functional groups that should be excluded.

    Checks for: aldehydes, epoxides, Michael acceptors, acyl halides,
    anhydrides, isocyanates.

    Returns:
        True if reactive groups are found (molecule should be rejected)
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return True

        reactive_smarts = [
            '[CH]=O',                       # Aldehyde
            'C1OC1',                        # Epoxide
            '[C]=[C][C]=O',                 # Michael acceptor (enone)
            'C(=O)[F,Cl,Br,I]',            # Acyl halide
            'C(=O)OC(=O)',                  # Anhydride
            'N=C=O',                        # Isocyanate
            'N=C=S',                        # Isothiocyanate
            'S(=O)(=O)F',                   # Sulfonyl fluoride
        ]

        for smarts in reactive_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True

        return False
    except Exception:
        return True


def passes_pains_filter(mol_or_smiles) -> bool:
    """
    Check if molecule passes PAINS (Pan Assay INterference compoundS) filter.

    Args:
        mol_or_smiles: RDKit Mol or SMILES

    Returns:
        True if passes (no PAINS alerts)
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return False

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)

        return not catalog.HasMatch(mol)
    except Exception:
        return False


def calculate_sa_score(mol_or_smiles) -> Optional[float]:
    """
    Calculate Synthetic Accessibility (SA) score.

    Uses a simplified approach based on fragment contributions.
    Score ranges from 1 (easy) to 10 (hard).

    Args:
        mol_or_smiles: RDKit Mol or SMILES

    Returns:
        SA score (1-10) or None if invalid
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return None

        # Complexity metrics
        num_rings = mol.GetRingInfo().NumRings()
        num_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        num_heavy = mol.GetNumHeavyAtoms()
        num_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [6, 1])

        # Heuristic SA score
        score = 1.0
        score += 0.2 * num_rings
        score += 0.5 * num_stereo
        score += 0.02 * num_heavy
        score += 0.1 * num_heteroatoms
        score += 0.3 * len([b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE])

        # Bridged rings are harder
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() > 1:
            atom_rings = ring_info.AtomRings()
            for i, ring_a in enumerate(atom_rings):
                for ring_b in atom_rings[i + 1:]:
                    shared = set(ring_a) & set(ring_b)
                    if len(shared) > 2:
                        score += 1.0

        return min(10.0, max(1.0, score))
    except Exception:
        return None


def is_synthetically_accessible(mol_or_smiles, max_score: float = 4.5) -> bool:
    """
    Check if molecule is synthetically accessible (SA score ≤ threshold).

    Args:
        mol_or_smiles: RDKit Mol or SMILES
        max_score: Maximum acceptable SA score

    Returns:
        True if synthetically accessible
    """
    score = calculate_sa_score(mol_or_smiles)
    if score is None:
        return False
    return score <= max_score


def passes_all_filters(mol_or_smiles, check_glue: bool = True) -> bool:
    """
    Check if molecule passes all quality filters.

    Args:
        mol_or_smiles: RDKit Mol or SMILES
        check_glue: Whether to check glue-likeness

    Returns:
        True if passes all filters
    """
    if not is_drug_like(mol_or_smiles):
        return False
    if check_glue and not is_glue_like(mol_or_smiles):
        return False
    if not passes_pains_filter(mol_or_smiles):
        return False
    if not is_synthetically_accessible(mol_or_smiles):
        return False
    if has_reactive_groups(mol_or_smiles):
        return False
    return True
