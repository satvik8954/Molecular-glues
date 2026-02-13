"""
Evaluation metrics for molecular generation.
"""
from typing import List, Optional, Dict
from collections import Counter
import numpy as np

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.chemistry import (
    is_valid_molecule, get_molecular_properties, canonicalize_smiles,
    get_morgan_fingerprint, calculate_tanimoto_similarity,
    calculate_qed, get_murcko_scaffold,
)
from rdkit.Chem import DataStructs


def validity_rate(molecules: List[str]) -> float:
    """Fraction of molecules that are chemically valid."""
    if not molecules:
        return 0.0
    valid = sum(1 for m in molecules if is_valid_molecule(m))
    return valid / len(molecules)


def uniqueness_rate(molecules: List[str]) -> float:
    """Fraction of unique molecules among valid ones."""
    valid = [canonicalize_smiles(m) for m in molecules if is_valid_molecule(m)]
    valid = [m for m in valid if m is not None]
    if not valid:
        return 0.0
    return len(set(valid)) / len(valid)


def novelty_rate(molecules: List[str], training_set: List[str]) -> float:
    """Fraction of generated molecules not in training set."""
    train_canonical = set()
    for m in training_set:
        c = canonicalize_smiles(m)
        if c:
            train_canonical.add(c)

    valid = [canonicalize_smiles(m) for m in molecules if is_valid_molecule(m)]
    valid = [m for m in valid if m is not None]

    if not valid:
        return 0.0

    novel = sum(1 for m in valid if m not in train_canonical)
    return novel / len(valid)


def internal_diversity(molecules: List[str], sample_size: int = 1000) -> float:
    """
    Average pairwise Tanimoto distance among generated molecules.
    1.0 = maximally diverse, 0.0 = all identical.
    """
    valid = [m for m in molecules if is_valid_molecule(m)]
    if len(valid) < 2:
        return 0.0

    fps = [get_morgan_fingerprint(m) for m in valid]
    fps = [fp for fp in fps if fp is not None]

    if len(fps) < 2:
        return 0.0

    # Sample pairs if too many
    if len(fps) > sample_size:
        indices = np.random.choice(len(fps), sample_size, replace=False)
        fps = [fps[i] for i in indices]

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    avg_sim = np.mean(similarities)
    return 1.0 - avg_sim


def scaffold_diversity(molecules: List[str]) -> float:
    """
    Fraction of unique Murcko scaffolds among valid molecules.

    Higher = more scaffold-diverse output.
    """
    valid = [m for m in molecules if is_valid_molecule(m)]
    if not valid:
        return 0.0

    scaffolds = []
    for m in valid:
        s = get_murcko_scaffold(m)
        if s is not None:
            scaffolds.append(s)

    if not scaffolds:
        return 0.0

    return len(set(scaffolds)) / len(scaffolds)


def wasserstein_property_distance(
    generated: List[str],
    reference: List[str],
    property_name: str,
) -> float:
    """
    1D Wasserstein (earth mover's) distance between property distributions.

    Args:
        generated: Generated SMILES
        reference: Reference SMILES
        property_name: Property to compare (key in get_molecular_properties)

    Returns:
        Wasserstein distance (lower = more similar distributions)
    """
    from scipy.stats import wasserstein_distance

    gen_values = []
    for m in generated:
        props = get_molecular_properties(m)
        if props and property_name in props:
            gen_values.append(props[property_name])

    ref_values = []
    for m in reference:
        props = get_molecular_properties(m)
        if props and property_name in props:
            ref_values.append(props[property_name])

    if not gen_values or not ref_values:
        return float('inf')

    return wasserstein_distance(gen_values, ref_values)


def compute_glue_similarity(
    generated: List[str],
    known_glues: List[str],
) -> Dict[str, float]:
    """
    Compute similarity metrics between generated and known glue molecules.

    Returns:
        Dict with mean_nn_sim (mean nearest-neighbor Tanimoto),
        max_sim, fraction with sim > 0.4
    """
    gen_fps = [get_morgan_fingerprint(m) for m in generated if is_valid_molecule(m)]
    gen_fps = [fp for fp in gen_fps if fp is not None]

    glue_fps = [get_morgan_fingerprint(m) for m in known_glues if is_valid_molecule(m)]
    glue_fps = [fp for fp in glue_fps if fp is not None]

    if not gen_fps or not glue_fps:
        return {'mean_nn_sim': 0.0, 'max_sim': 0.0, 'frac_similar': 0.0}

    nn_sims = []
    for gen_fp in gen_fps:
        max_sim = max(DataStructs.TanimotoSimilarity(gen_fp, g) for g in glue_fps)
        nn_sims.append(max_sim)

    return {
        'mean_nn_sim': float(np.mean(nn_sims)),
        'max_sim': float(np.max(nn_sims)),
        'frac_similar': float(np.mean([s > 0.4 for s in nn_sims])),
    }


def qed_score_stats(molecules: List[str]) -> Dict[str, float]:
    """
    QED score statistics for generated molecules.
    """
    scores = [calculate_qed(m) for m in molecules if is_valid_molecule(m)]
    scores = [s for s in scores if s > 0]

    if not scores:
        return {'qed_mean': 0.0, 'qed_std': 0.0, 'qed_frac_good': 0.0}

    return {
        'qed_mean': float(np.mean(scores)),
        'qed_std': float(np.std(scores)),
        'qed_frac_good': float(np.mean([s > 0.4 for s in scores])),
    }


def property_distribution(
    molecules: List[str],
    property_name: str,
) -> Dict[str, float]:
    """Get statistics for a property across generated molecules."""
    values = []
    for m in molecules:
        props = get_molecular_properties(m)
        if props and property_name in props:
            values.append(props[property_name])

    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'n': 0}

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'n': len(values),
    }


def compute_all_metrics(
    generated: List[str],
    training_set: Optional[List[str]] = None,
    known_glues: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Compute all evaluation metrics.

    Args:
        generated: Generated SMILES
        training_set: Training SMILES for novelty
        known_glues: Known glue SMILES for similarity metrics

    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'validity': validity_rate(generated),
        'uniqueness': uniqueness_rate(generated),
        'diversity': internal_diversity(generated),
        'scaffold_diversity': scaffold_diversity(generated),
    }

    if training_set:
        metrics['novelty'] = novelty_rate(generated, training_set)

    # QED stats
    metrics.update(qed_score_stats(generated))

    # Property distributions
    for prop in ['molecular_weight', 'logp', 'num_aromatic_rings', 'hbd', 'hba', 'tpsa', 'fraction_sp3', 'qed']:
        dist = property_distribution(generated, prop)
        metrics[f'{prop}_mean'] = dist['mean']
        metrics[f'{prop}_std'] = dist['std']

    # Glue similarity
    if known_glues:
        metrics.update(compute_glue_similarity(generated, known_glues))

    # Wasserstein distances to training set
    if training_set:
        for prop in ['molecular_weight', 'logp', 'tpsa']:
            try:
                metrics[f'{prop}_wasserstein'] = wasserstein_property_distance(
                    generated, training_set, prop
                )
            except ImportError:
                pass  # scipy not available

    return metrics


def print_metrics_report(metrics: Dict[str, any]):
    """Print a formatted metrics report."""
    print("\n" + "=" * 60)
    print("GENERATION METRICS REPORT")
    print("=" * 60)

    # Core metrics
    print("\n--- Core Metrics ---")
    for key in ['validity', 'uniqueness', 'novelty', 'diversity', 'scaffold_diversity']:
        if key in metrics:
            print(f"  {key:25s}: {metrics[key]:.4f}")

    # QED
    print("\n--- QED ---")
    for key in ['qed_mean', 'qed_std', 'qed_frac_good']:
        if key in metrics:
            print(f"  {key:25s}: {metrics[key]:.4f}")

    # Glue similarity
    if 'mean_nn_sim' in metrics:
        print("\n--- Glue Similarity ---")
        for key in ['mean_nn_sim', 'max_sim', 'frac_similar']:
            print(f"  {key:25s}: {metrics[key]:.4f}")

    # Property distributions
    print("\n--- Property Distributions ---")
    for prop in ['molecular_weight', 'logp', 'num_aromatic_rings', 'hbd', 'hba', 'tpsa', 'fraction_sp3', 'qed']:
        mean_key = f'{prop}_mean'
        std_key = f'{prop}_std'
        if mean_key in metrics:
            print(f"  {prop:25s}: {metrics[mean_key]:.2f} ± {metrics.get(std_key, 0):.2f}")

    # Wasserstein
    wass_keys = [k for k in metrics if k.endswith('_wasserstein')]
    if wass_keys:
        print("\n--- Wasserstein Distances ---")
        for key in wass_keys:
            print(f"  {key:25s}: {metrics[key]:.4f}")

    print("\n" + "=" * 60)
