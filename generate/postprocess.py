"""
Post-processing for generated molecules.
"""
from typing import List, Optional, Set
import os
import csv
from rdkit import Chem

import sys

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.molecular_graph import graph_to_smiles
from utils.chemistry import (
    is_valid_molecule, canonicalize_smiles, get_molecular_properties,
)
from utils.filters import (
    is_drug_like, is_glue_like, passes_pains_filter,
    is_synthetically_accessible, has_reactive_groups,
)


def postprocess_molecule(graph) -> Optional[str]:
    """
    Convert generated graph to valid SMILES.

    Args:
        graph: PyG Data object

    Returns:
        Canonical SMILES string or None
    """
    smiles = graph_to_smiles(graph)
    if smiles is None:
        return None

    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None

    if not is_valid_molecule(canonical):
        return None

    return canonical


def filter_generated(
    smiles_list: List[str],
    drug_like: bool = True,
    glue_like: bool = True,
    pains: bool = True,
    sa_accessible: bool = True,
    no_reactive: bool = True,
    max_heavy_atoms: int = 35,
    min_aromatic_rings: int = 1,
    min_hbd_hba: int = 2,
    min_fsp3: float = 0.2,
) -> List[str]:
    """
    Filter generated molecules by multiple criteria.

    Args:
        smiles_list: List of SMILES strings
        drug_like: Apply Lipinski's Rule of Five
        glue_like: Apply glue-likeness criteria
        pains: Apply PAINS filter
        sa_accessible: Check synthetic accessibility
        no_reactive: Reject molecules with reactive groups
        max_heavy_atoms: Maximum heavy atom count
        min_aromatic_rings: Minimum aromatic ring count
        min_hbd_hba: Minimum HBD + HBA
        min_fsp3: Minimum fraction sp3 carbons

    Returns:
        Filtered list of SMILES
    """
    filtered = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Heavy atom check
        if mol.GetNumHeavyAtoms() > max_heavy_atoms:
            continue

        # Aromatic ring check
        from rdkit.Chem import Descriptors
        if Descriptors.NumAromaticRings(mol) < min_aromatic_rings:
            continue

        # HBD + HBA check
        from rdkit.Chem import rdMolDescriptors
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        if (hbd + hba) < min_hbd_hba:
            continue

        # Fsp3 check
        props = get_molecular_properties(smiles)
        if props and props.get('fraction_sp3', 0) < min_fsp3:
            continue

        # Standard filters
        if drug_like and not is_drug_like(mol):
            continue
        if glue_like and not is_glue_like(mol):
            continue
        if pains and not passes_pains_filter(mol):
            continue
        if sa_accessible and not is_synthetically_accessible(mol):
            continue
        if no_reactive and has_reactive_groups(mol):
            continue

        filtered.append(smiles)

    return filtered


def deduplicate(
    smiles_list: List[str],
    training_smiles: Optional[List[str]] = None,
) -> List[str]:
    """
    Remove duplicate molecules and optionally training set molecules.

    Args:
        smiles_list: Generated SMILES
        training_smiles: Training set SMILES to exclude

    Returns:
        Deduplicated list
    """
    seen: Set[str] = set()

    # Add training set
    if training_smiles:
        for s in training_smiles:
            c = canonicalize_smiles(s)
            if c:
                seen.add(c)

    unique = []
    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical and canonical not in seen:
            seen.add(canonical)
            unique.append(canonical)

    return unique


def save_molecules(
    smiles_list: List[str],
    output_path: str,
    include_properties: bool = True,
):
    """
    Save molecules to CSV file with properties.

    Args:
        smiles_list: List of SMILES
        output_path: Path to output CSV
        include_properties: Include calculated properties
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    headers = ['smiles']
    if include_properties:
        headers.extend([
            'molecular_weight', 'logp', 'hbd', 'hba', 'tpsa',
            'rotatable_bonds', 'num_rings', 'num_aromatic_rings',
            'num_heavy_atoms', 'fraction_sp3', 'qed',
        ])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for smiles in smiles_list:
            row = [smiles]
            if include_properties:
                props = get_molecular_properties(smiles)
                if props:
                    row.extend([
                        f"{props.get('molecular_weight', 0):.2f}",
                        f"{props.get('logp', 0):.2f}",
                        props.get('hbd', 0),
                        props.get('hba', 0),
                        f"{props.get('tpsa', 0):.2f}",
                        props.get('rotatable_bonds', 0),
                        props.get('num_rings', 0),
                        props.get('num_aromatic_rings', 0),
                        props.get('num_heavy_atoms', 0),
                        f"{props.get('fraction_sp3', 0):.3f}",
                        f"{props.get('qed', 0):.3f}",
                    ])
                else:
                    row.extend([''] * 11)
            writer.writerow(row)

    print(f"Saved {len(smiles_list)} molecules to {output_path}")


def save_molecules_sdf(
    smiles_list: List[str],
    output_path: str,
):
    """
    Save molecules to SDF format.

    Args:
        smiles_list: List of SMILES
        output_path: Path to output SDF file
    """
    from rdkit.Chem import AllChem

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    writer = Chem.SDWriter(output_path)

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Add 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Add properties
        props = get_molecular_properties(smiles)
        if props:
            for key, value in props.items():
                mol.SetProp(key, str(value))

        writer.write(mol)

    writer.close()
    print(f"Saved {len(smiles_list)} molecules to {output_path}")
