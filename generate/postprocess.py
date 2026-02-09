"""
Post-processing utilities for generated molecules.
"""
from typing import List, Set, Optional, Tuple
import torch
from torch_geometric.data import Data

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.molecular_graph import graph_to_smiles
from utils.chemistry import is_valid_molecule, canonicalize_smiles
from utils.filters import is_drug_like, is_glue_like, passes_pains_filter, is_synthetically_accessible


def postprocess_molecule(graph: Data) -> Optional[str]:
    """
    Convert generated graph to valid SMILES.
    
    Args:
        graph: Generated molecular graph
        
    Returns:
        Canonical SMILES or None if invalid
    """
    # Convert graph to SMILES
    smiles = graph_to_smiles(graph)
    
    if smiles is None:
        return None
    
    # Validate and canonicalize
    if not is_valid_molecule(smiles):
        return None
    
    canonical = canonicalize_smiles(smiles)
    return canonical


def filter_generated(
    molecules: List[Data],
    apply_drug_like: bool = True,
    apply_glue_like: bool = False,
    apply_pains: bool = True,
    apply_sa_filter: bool = True,
    verbose: bool = True,
) -> Tuple[List[str], dict]:
    """
    Filter generated molecules by chemical validity and properties.
    
    Args:
        molecules: List of generated molecular graphs
        apply_drug_like: Filter by Lipinski's rules
        apply_glue_like: Filter by glue-like properties
        apply_pains: Filter out PAINS patterns
        apply_sa_filter: Filter by synthetic accessibility
        verbose: Print statistics
        
    Returns:
        Tuple of (valid SMILES list, statistics dict)
    """
    stats = {
        'total_generated': len(molecules),
        'valid_smiles': 0,
        'drug_like': 0,
        'glue_like': 0,
        'pains_pass': 0,
        'sa_pass': 0,
        'final_count': 0,
    }
    
    valid_smiles = []
    
    for graph in molecules:
        # Convert to SMILES
        smiles = postprocess_molecule(graph)
        if smiles is None:
            continue
        
        stats['valid_smiles'] += 1
        
        # Apply filters
        passes = True
        
        if apply_drug_like:
            if is_drug_like(smiles):
                stats['drug_like'] += 1
            else:
                passes = False
        
        if passes and apply_glue_like:
            if is_glue_like(smiles):
                stats['glue_like'] += 1
            else:
                passes = False
        
        if passes and apply_pains:
            if passes_pains_filter(smiles):
                stats['pains_pass'] += 1
            else:
                passes = False
        
        if passes and apply_sa_filter:
            if is_synthetically_accessible(smiles):
                stats['sa_pass'] += 1
            else:
                passes = False
        
        if passes:
            valid_smiles.append(smiles)
    
    stats['final_count'] = len(valid_smiles)
    
    if verbose:
        print(f"Generated: {stats['total_generated']}")
        print(f"Valid SMILES: {stats['valid_smiles']} ({100*stats['valid_smiles']/max(1,stats['total_generated']):.1f}%)")
        print(f"Final (after filters): {stats['final_count']} ({100*stats['final_count']/max(1,stats['total_generated']):.1f}%)")
    
    return valid_smiles, stats


def deduplicate(
    smiles_list: List[str],
    training_smiles: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Tuple[List[str], dict]:
    """
    Remove duplicates and optionally filter out training set molecules.
    
    Args:
        smiles_list: List of generated SMILES
        training_smiles: Set of training SMILES (for novelty check)
        verbose: Print statistics
        
    Returns:
        Tuple of (unique novel SMILES, statistics)
    """
    stats = {
        'input_count': len(smiles_list),
        'unique_count': 0,
        'novel_count': 0,
    }
    
    # Remove duplicates
    unique_smiles = list(set(smiles_list))
    stats['unique_count'] = len(unique_smiles)
    
    # Remove training set molecules
    if training_smiles:
        novel_smiles = [s for s in unique_smiles if s not in training_smiles]
        stats['novel_count'] = len(novel_smiles)
    else:
        novel_smiles = unique_smiles
        stats['novel_count'] = len(novel_smiles)
    
    if verbose:
        print(f"Input: {stats['input_count']}")
        print(f"Unique: {stats['unique_count']} ({100*stats['unique_count']/max(1,stats['input_count']):.1f}%)")
        if training_smiles:
            print(f"Novel: {stats['novel_count']} ({100*stats['novel_count']/max(1,stats['unique_count']):.1f}%)")
    
    return novel_smiles, stats


def save_molecules(
    smiles_list: List[str],
    output_path: str,
    include_properties: bool = True,
):
    """
    Save generated molecules to CSV file.
    
    Args:
        smiles_list: List of SMILES strings
        output_path: Output file path
        include_properties: Include computed properties
    """
    import pandas as pd
    from utils.chemistry import get_molecular_properties
    
    data = []
    for smiles in smiles_list:
        row = {'smiles': smiles}
        
        if include_properties:
            props = get_molecular_properties(smiles)
            if props:
                row.update(props)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(smiles_list)} molecules to {output_path}")
