"""
Chemistry utility functions for molecular property calculation and validation.
"""
from typing import Optional, Dict, List
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, AllChem,
    DataStructs, QED as QEDModule,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np


def is_valid_molecule(mol_or_smiles) -> bool:
    """
    Check if a molecule or SMILES string is chemically valid.

    Args:
        mol_or_smiles: RDKit Mol object or SMILES string

    Returns:
        True if valid molecule
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        elif hasattr(mol_or_smiles, 'GetNumAtoms'):
            mol = mol_or_smiles
        else:
            return False

        if mol is None:
            return False

        # Try sanitization
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def check_valency(mol_or_smiles) -> bool:
    """
    Check if all atoms have valid valency.

    Args:
        mol_or_smiles: RDKit Mol or SMILES

    Returns:
        True if all valencies are correct
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return False

        # Sanitize checks valency
        Chem.SanitizeMol(mol)
        return True
    except Chem.MolSanitizeException:
        return False
    except Exception:
        return False


def check_ring_stability(mol_or_smiles) -> bool:
    """
    Check if ring systems are stable (no 3-membered rings with double bonds, etc.).

    Args:
        mol_or_smiles: RDKit Mol or SMILES

    Returns:
        True if ring systems are stable
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return False

        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if len(ring) < 3:
                return False
            # 3-membered rings with double bonds are unstable
            if len(ring) == 3:
                for idx in ring:
                    atom = mol.GetAtomWithIdx(idx)
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.BondType.DOUBLE:
                            other = bond.GetOtherAtomIdx(idx)
                            if other in ring:
                                return False
        return True
    except Exception:
        return False


def get_molecular_properties(mol_or_smiles) -> Optional[Dict[str, float]]:
    """
    Calculate molecular properties.

    Args:
        mol_or_smiles: RDKit Mol or SMILES string

    Returns:
        Dictionary of properties or None if invalid
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return None

        # Count ring Types
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)

        # Calculate Fsp3
        num_sp3 = 0
        num_carbons = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                num_carbons += 1
                if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                    num_sp3 += 1
        fraction_sp3 = num_sp3 / max(1, num_carbons)

        # QED
        qed_score = calculate_qed(mol)

        return {
            'molecular_weight': Descriptors.ExactMolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': rdMolDescriptors.CalcNumHBD(mol),
            'hba': rdMolDescriptors.CalcNumHBA(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'num_rings': num_rings,
            'num_aromatic_rings': num_aromatic_rings,
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'fraction_sp3': fraction_sp3,
            'qed': qed_score,
        }
    except Exception:
        return None


def calculate_qed(mol_or_smiles) -> float:
    """
    Calculate Quantitative Estimate of Drug-likeness (QED).

    Args:
        mol_or_smiles: RDKit Mol or SMILES string

    Returns:
        QED score (0.0-1.0), 0.0 if invalid
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return 0.0

        return QEDModule.qed(mol)
    except Exception:
        return 0.0


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """
    Get the Murcko scaffold (generic framework) of a molecule.

    Args:
        smiles: SMILES string

    Returns:
        Canonical SMILES of the generic Murcko scaffold, or None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic, canonical=True)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string.

    Args:
        smiles: Input SMILES

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
    Get Morgan (circular) fingerprint.

    Args:
        mol_or_smiles: RDKit Mol or SMILES
        radius: Fingerprint radius
        n_bits: Number of bits

    Returns:
        RDKit fingerprint object or None
    """
    try:
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles

        if mol is None:
            return None

        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two molecules.

    Args:
        smiles1: First molecule SMILES
        smiles2: Second molecule SMILES

    Returns:
        Tanimoto similarity (0.0-1.0)
    """
    try:
        fp1 = get_morgan_fingerprint(smiles1)
        fp2 = get_morgan_fingerprint(smiles2)

        if fp1 is None or fp2 is None:
            return 0.0

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def calculate_tanimoto_batch(smiles: str, reference_smiles: List[str]) -> List[float]:
    """
    Calculate Tanimoto similarity of a molecule against a list of references.

    Args:
        smiles: Query molecule SMILES
        reference_smiles: List of reference SMILES

    Returns:
        List of Tanimoto similarities
    """
    fp = get_morgan_fingerprint(smiles)
    if fp is None:
        return [0.0] * len(reference_smiles)

    sims = []
    for ref in reference_smiles:
        ref_fp = get_morgan_fingerprint(ref)
        if ref_fp is not None:
            sims.append(DataStructs.TanimotoSimilarity(fp, ref_fp))
        else:
            sims.append(0.0)

    return sims
