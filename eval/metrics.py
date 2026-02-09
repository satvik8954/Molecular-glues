"""
Evaluation metrics for generated molecules.
"""
from typing import List, Dict, Set, Optional
import numpy as np
from collections import Counter

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.chemistry import (
    is_valid_molecule, 
    canonicalize_smiles, 
    get_molecular_properties,
    calculate_tanimoto_similarity,
    get_morgan_fingerprint
)


def validity_rate(molecules: List[str]) -> float:
    """
    Calculate the fraction of chemically valid molecules.
    
    Args:
        molecules: List of SMILES strings
        
    Returns:
        Validity rate (0-1)
    """
    if not molecules:
        return 0.0
    
    valid_count = sum(1 for m in molecules if is_valid_molecule(m))
    return valid_count / len(molecules)


def uniqueness_rate(molecules: List[str]) -> float:
    """
    Calculate the fraction of unique molecules.
    
    Args:
        molecules: List of SMILES strings
        
    Returns:
        Uniqueness rate (0-1)
    """
    if not molecules:
        return 0.0
    
    # Canonicalize for accurate comparison
    canonical = []
    for m in molecules:
        c = canonicalize_smiles(m)
        if c:
            canonical.append(c)
    
    if not canonical:
        return 0.0
    
    unique = set(canonical)
    return len(unique) / len(canonical)


def novelty_rate(
    molecules: List[str],
    training_set: Set[str],
) -> float:
    """
    Calculate the fraction of molecules not in the training set.
    
    Args:
        molecules: List of generated SMILES
        training_set: Set of training SMILES (canonicalized)
        
    Returns:
        Novelty rate (0-1)
    """
    if not molecules:
        return 0.0
    
    # Canonicalize training set if not already
    canonical_training = set()
    for s in training_set:
        c = canonicalize_smiles(s)
        if c:
            canonical_training.add(c)
    
    # Count novel molecules
    novel_count = 0
    for m in molecules:
        c = canonicalize_smiles(m)
        if c and c not in canonical_training:
            novel_count += 1
    
    return novel_count / len(molecules)


def internal_diversity(
    molecules: List[str],
    sample_size: int = 1000,
    fingerprint_radius: int = 2,
) -> float:
    """
    Calculate internal diversity using pairwise Tanimoto distances.
    
    Diversity = 1 - average_similarity
    
    Args:
        molecules: List of SMILES strings
        sample_size: Maximum pairs to sample (for efficiency)
        fingerprint_radius: Morgan fingerprint radius
        
    Returns:
        Internal diversity (0-1, higher is more diverse)
    """
    if len(molecules) < 2:
        return 0.0
    
    # Limit to valid molecules
    valid_mols = [m for m in molecules if is_valid_molecule(m)]
    if len(valid_mols) < 2:
        return 0.0
    
    # Sample pairs if too many molecules
    if len(valid_mols) > 100:
        indices = np.random.choice(len(valid_mols), size=min(100, len(valid_mols)), replace=False)
        valid_mols = [valid_mols[i] for i in indices]
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(valid_mols)):
        for j in range(i + 1, len(valid_mols)):
            sim = calculate_tanimoto_similarity(
                valid_mols[i], valid_mols[j], fingerprint_radius
            )
            similarities.append(sim)
    
    if not similarities:
        return 0.0
    
    avg_similarity = np.mean(similarities)
    return 1.0 - avg_similarity


def property_distribution(
    molecules: List[str],
    properties: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate distribution statistics for molecular properties.
    
    Args:
        molecules: List of SMILES strings
        properties: List of property names (uses defaults if None)
        
    Returns:
        Dictionary of {property: {mean, std, min, max}}
    """
    if properties is None:
        properties = [
            'molecular_weight', 'logp', 'tpsa', 
            'hbd', 'hba', 'rotatable_bonds',
            'num_rings', 'num_aromatic_rings'
        ]
    
    # Collect property values
    prop_values = {p: [] for p in properties}
    
    for smiles in molecules:
        props = get_molecular_properties(smiles)
        if props:
            for p in properties:
                if p in props:
                    prop_values[p].append(props[p])
    
    # Calculate statistics
    stats = {}
    for p, values in prop_values.items():
        if values:
            stats[p] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values),
            }
        else:
            stats[p] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    
    return stats


def compute_all_metrics(
    generated: List[str],
    training_set: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Compute all generation metrics.
    
    Args:
        generated: List of generated SMILES
        training_set: Optional set of training SMILES
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'validity': validity_rate(generated),
        'uniqueness': uniqueness_rate(generated),
        'diversity': internal_diversity(generated),
    }
    
    if training_set:
        metrics['novelty'] = novelty_rate(generated, training_set)
    
    return metrics


def print_metrics_report(
    generated: List[str],
    training_set: Optional[Set[str]] = None,
):
    """
    Print a formatted metrics report.
    
    Args:
        generated: List of generated SMILES
        training_set: Optional set of training SMILES
    """
    print("\n" + "="*50)
    print("MOLECULAR GENERATION METRICS")
    print("="*50)
    
    metrics = compute_all_metrics(generated, training_set)
    
    print(f"\nBasic Metrics:")
    print(f"  Validity:   {100*metrics['validity']:.1f}%")
    print(f"  Uniqueness: {100*metrics['uniqueness']:.1f}%")
    print(f"  Diversity:  {100*metrics['diversity']:.1f}%")
    
    if 'novelty' in metrics:
        print(f"  Novelty:    {100*metrics['novelty']:.1f}%")
    
    # Valid molecules only for property analysis
    valid_mols = [m for m in generated if is_valid_molecule(m)]
    
    if valid_mols:
        print(f"\nProperty Distribution (n={len(valid_mols)}):")
        props = property_distribution(valid_mols)
        
        for prop, stats in props.items():
            if stats['count'] > 0:
                print(f"  {prop}: {stats['mean']:.1f} ± {stats['std']:.1f} "
                      f"[{stats['min']:.1f}, {stats['max']:.1f}]")
    
    print("="*50 + "\n")
