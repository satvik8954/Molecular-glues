"""
Molecular filters for drug-likeness and glue-like properties.
"""
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, FilterCatalog

from .chemistry import get_molecular_properties


def is_drug_like(mol_or_smiles, strict: bool = False) -> bool:
    """
    Check if molecule passes Lipinski's Rule of Five.
    
    Args:
        mol_or_smiles: RDKit molecule or SMILES string
        strict: If True, require all rules; if False, allow 1 violation
        
    Returns:
        True if molecule is drug-like
    """
    props = get_molecular_properties(mol_or_smiles)
    if props is None:
        return False
    
    violations = 0
    
    # Lipinski's Rule of Five
    if props['molecular_weight'] > 500:
        violations += 1
    if props['logp'] > 5:
        violations += 1
    if props['hbd'] > 5:
        violations += 1
    if props['hba'] > 10:
        violations += 1
    
    if strict:
        return violations == 0
    else:
        return violations <= 1


def is_glue_like(mol_or_smiles) -> bool:
    """
    Check if molecule has molecular glue-like properties.
    
    Criteria:
    - MW 200-500 Da (compact size)
    - At least 1 aromatic ring (scaffold rigidity)
    - Mixed polarity (has both hydrophobic and polar groups)
    - Multiple H-bond interaction points
    - Not too flexible (limited rotatable bonds)
    
    Args:
        mol_or_smiles: RDKit molecule or SMILES string
        
    Returns:
        True if molecule has glue-like properties
    """
    props = get_molecular_properties(mol_or_smiles)
    if props is None:
        return False
    
    # Molecular weight: compact size
    if not (200 <= props['molecular_weight'] <= 500):
        return False
    
    # Must have aromatic content (rigid scaffold)
    if props['num_aromatic_rings'] < 1:
        return False
    
    # Mixed polarity: need some polar surface area but not too much
    if not (20 <= props['tpsa'] <= 120):
        return False
    
    # Need H-bond interaction points
    if props['hbd'] + props['hba'] < 2:
        return False
    
    # Not too flexible (unlike PROTACs)
    if props['rotatable_bonds'] > 8:
        return False
    
    # Should have heteroatoms for polarity
    if props['num_heteroatoms'] < 2:
        return False
    
    # Ring count check (compact, not extended)
    if props['num_rings'] < 1 or props['num_rings'] > 5:
        return False
    
    return True


def passes_pains_filter(mol_or_smiles) -> bool:
    """
    Check if molecule passes PAINS (Pan Assay Interference Compounds) filter.
    
    Args:
        mol_or_smiles: RDKit molecule or SMILES string
        
    Returns:
        True if molecule passes PAINS filter (no PAINS alerts)
    """
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    
    if mol is None:
        return False
    
    try:
        # Create PAINS filter catalog
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        
        # Check for matches
        entry = catalog.GetFirstMatch(mol)
        return entry is None
    except Exception:
        # If filter fails, assume it passes (conservative)
        return True


def calculate_sa_score(mol_or_smiles) -> Optional[float]:
    """
    Calculate synthetic accessibility score (1=easy, 10=hard).
    
    Based on Ertl & Schuffenhauer, J. Cheminform. 2009.
    
    Args:
        mol_or_smiles: RDKit molecule or SMILES string
        
    Returns:
        SA score (1-10) or None if calculation fails
    """
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    
    if mol is None:
        return None
    
    try:
        # Simplified SA score approximation
        # Full implementation would use fragment contributions
        
        num_atoms = mol.GetNumHeavyAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        ring_complexity = num_spiro + num_bridgehead
        
        # Heuristic SA score
        # Base score starts at 2 (easy)
        sa_score = 2.0
        
        # Size penalty
        if num_atoms > 30:
            sa_score += (num_atoms - 30) * 0.1
        
        # Ring complexity
        if num_rings > 3:
            sa_score += (num_rings - 3) * 0.5
        
        # Stereo centers
        sa_score += num_stereo * 0.3
        
        # Spiro and bridgehead atoms (hard to synthesize)
        sa_score += ring_complexity * 1.0
        
        # Clamp to 1-10 range
        sa_score = max(1.0, min(10.0, sa_score))
        
        return sa_score
    except Exception:
        return None


def is_synthetically_accessible(mol_or_smiles, threshold: float = 6.0) -> bool:
    """
    Check if molecule is synthetically accessible.
    
    Args:
        mol_or_smiles: RDKit molecule or SMILES string
        threshold: Maximum SA score (default 6.0)
        
    Returns:
        True if SA score is below threshold
    """
    sa_score = calculate_sa_score(mol_or_smiles)
    if sa_score is None:
        return False
    return sa_score <= threshold


def passes_all_filters(mol_or_smiles, require_glue_like: bool = True) -> bool:
    """
    Check if molecule passes all filters.
    
    Args:
        mol_or_smiles: RDKit molecule or SMILES string
        require_glue_like: If True, also check glue-like properties
        
    Returns:
        True if molecule passes all filters
    """
    # Basic drug-likeness
    if not is_drug_like(mol_or_smiles):
        return False
    
    # PAINS filter
    if not passes_pains_filter(mol_or_smiles):
        return False
    
    # Synthetic accessibility
    if not is_synthetically_accessible(mol_or_smiles):
        return False
    
    # Glue-like properties (optional)
    if require_glue_like and not is_glue_like(mol_or_smiles):
        return False
    
    return True
