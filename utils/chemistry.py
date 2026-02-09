"""
Chemistry utilities for molecular validation and property calculation.
"""
from typing import Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem


def is_valid_molecule(smiles: str) -> bool:
    """
    Check if a SMILES string represents a valid molecule.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if molecule is valid, False otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol is not None
    except Exception:
        return False


def check_valency(mol: Chem.Mol) -> bool:
    """
    Check if all atoms in the molecule have valid valencies.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        True if all valencies are valid
    """
    if mol is None:
        return False
    
    try:
        # This will raise an exception if valencies are invalid
        Chem.SanitizeMol(mol)
        
        # Additional check for explicit valence
        for atom in mol.GetAtoms():
            valence = atom.GetTotalValence()
            atomic_num = atom.GetAtomicNum()
            
            # Define valid valences for common atoms
            valid_valences = {
                6: [4],           # Carbon
                7: [3, 5],        # Nitrogen (3 for amines, 5 for nitro)
                8: [2],           # Oxygen
                16: [2, 4, 6],    # Sulfur
                9: [1],           # Fluorine
                17: [1, 3, 5, 7], # Chlorine
                35: [1, 3, 5],    # Bromine
                53: [1, 3, 5, 7], # Iodine
                15: [3, 5],       # Phosphorus
            }
            
            if atomic_num in valid_valences:
                if valence not in valid_valences[atomic_num]:
                    return False
                    
        return True
    except Exception:
        return False


def check_ring_stability(mol: Chem.Mol) -> bool:
    """
    Check for stable ring systems (no highly strained rings).
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        True if ring systems are stable
    """
    if mol is None:
        return False
    
    try:
        ring_info = mol.GetRingInfo()
        
        for ring in ring_info.AtomRings():
            ring_size = len(ring)
            
            # Flag highly strained 3-membered rings (but allow them in special cases)
            if ring_size == 3:
                # Check if it's a cyclopropane (might be intentional)
                # We'll allow but could add stricter checks
                pass
            
            # 4-membered rings are strained but sometimes valid
            # 5+ membered rings are generally okay
            
        # Check for proper aromaticity
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                # Aromatic atoms should be in aromatic rings
                if not any(mol.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), size) 
                          for size in [5, 6, 7]):
                    return False
                    
        return True
    except Exception:
        return False


def get_molecular_properties(mol_or_smiles) -> Optional[Dict[str, float]]:
    """
    Calculate molecular properties relevant for drug-likeness.
    
    Args:
        mol_or_smiles: RDKit molecule or SMILES string
        
    Returns:
        Dictionary of molecular properties or None if invalid
    """
    # Handle SMILES input
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    
    if mol is None:
        return None
    
    try:
        properties = {
            # Basic properties
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            
            # Lipinski properties
            'hbd': rdMolDescriptors.CalcNumHBD(mol),  # H-bond donors
            'hba': rdMolDescriptors.CalcNumHBA(mol),  # H-bond acceptors
            'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            
            # Ring information
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            
            # Additional descriptors
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'fraction_sp3': rdMolDescriptors.CalcFractionCSP3(mol),
            
            # Complexity
            'num_stereo_centers': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        }
        
        return properties
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert SMILES to canonical form.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_morgan_fingerprint(mol_or_smiles, radius: int = 2, n_bits: int = 2048):
    """
    Generate Morgan fingerprint for a molecule.
    
    Args:
        mol_or_smiles: RDKit molecule or SMILES string
        radius: Fingerprint radius
        n_bits: Number of bits in fingerprint
        
    Returns:
        Numpy array of fingerprint bits or None
    """
    import numpy as np
    
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    else:
        mol = mol_or_smiles
    
    if mol is None:
        return None
    
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None


def calculate_tanimoto_similarity(mol1, mol2, radius: int = 2) -> float:
    """
    Calculate Tanimoto similarity between two molecules.
    
    Args:
        mol1: First molecule (SMILES or mol object)
        mol2: Second molecule (SMILES or mol object)
        radius: Morgan fingerprint radius
        
    Returns:
        Tanimoto similarity (0-1)
    """
    from rdkit import DataStructs
    
    # Convert to mol objects if needed
    if isinstance(mol1, str):
        mol1 = Chem.MolFromSmiles(mol1)
    if isinstance(mol2, str):
        mol2 = Chem.MolFromSmiles(mol2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0
