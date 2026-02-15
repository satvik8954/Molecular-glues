"""
Preprocess and merge datasets for training the molecular diffusion model.

Merges ChEMBL drug-like molecules with known glue chemotypes, deduplicates,
balances the glue fraction via oversampling, creates train/val splits, and
saves dataset statistics.

Usage:
    python scripts/preprocess_data.py \\
        --chembl data/raw/chembl_druglike.csv \\
        --glues data/glue_chemotypes.csv \\
        --output data/processed/training_data.csv \\
        --glue_fraction 0.15 \\
        --val_split 0.1
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
from utils.chemistry import (
    canonicalize_smiles,
    get_molecular_properties,
    is_valid_molecule,
)


def load_smiles_from_csv(path: str, smiles_col: str = "smiles") -> pd.DataFrame:
    """
    Load and validate SMILES from a CSV file.

    Args:
        path: Path to the CSV.
        smiles_col: Column containing SMILES strings.

    Returns:
        DataFrame with at least a ``smiles`` column of canonical SMILES.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        # Try common alternatives
        for alt in ("SMILES", "Smiles", "canonical_smiles"):
            if alt in df.columns:
                smiles_col = alt
                break
        else:
            raise KeyError(
                f"Column '{smiles_col}' not found. Available: {list(df.columns)}"
            )

    raw = df[smiles_col].dropna().tolist()
    canonical = []
    for smi in raw:
        c = canonicalize_smiles(str(smi))
        if c and is_valid_molecule(c):
            canonical.append(c)

    print(f"  Loaded {len(canonical)}/{len(raw)} valid SMILES from {path}")
    return pd.DataFrame({"smiles": canonical})


def compute_all_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute molecular properties for each SMILES and add as columns.

    Args:
        df: DataFrame with ``smiles`` column.

    Returns:
        DataFrame augmented with property columns.
    """
    records = []
    for smi in tqdm(df["smiles"], desc="Computing properties", unit="mol"):
        props = get_molecular_properties(smi)
        if props is None:
            props = {}
        props["smiles"] = smi
        records.append(props)
    return pd.DataFrame(records)


def merge_and_balance(
    df_chembl: pd.DataFrame,
    df_glues: pd.DataFrame,
    glue_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Merge ChEMBL and glue datasets, deduplicate, and balance via oversampling.

    Args:
        df_chembl: ChEMBL drug-like molecules (must have ``smiles``).
        df_glues: Known glue chemotypes (must have ``smiles``).
        glue_fraction: Target fraction of glue molecules in final dataset.
        seed: Random seed for shuffling.

    Returns:
        Merged, shuffled DataFrame with ``is_glue`` column.
    """
    # Tag sources
    df_chembl = df_chembl.copy()
    df_glues = df_glues.copy()
    df_chembl["is_glue"] = 0
    df_glues["is_glue"] = 1

    # Deduplicate within each set
    df_chembl = df_chembl.drop_duplicates(subset="smiles")
    df_glues = df_glues.drop_duplicates(subset="smiles")

    # Remove glues that happen to be in ChEMBL set already
    glue_set = set(df_glues["smiles"])
    df_chembl = df_chembl[~df_chembl["smiles"].isin(glue_set)]

    n_chembl = len(df_chembl)
    n_glues_orig = len(df_glues)

    # Compute how many glue copies are needed
    # Want: n_glue_final / (n_chembl + n_glue_final) = glue_fraction
    # => n_glue_final = glue_fraction * n_chembl / (1 - glue_fraction)
    if glue_fraction > 0 and n_glues_orig > 0:
        n_glue_target = int(glue_fraction * n_chembl / (1 - glue_fraction))
        n_glue_target = max(n_glue_target, n_glues_orig)  # at least original count

        if n_glue_target > n_glues_orig:
            # Oversample glues
            rng = np.random.RandomState(seed)
            repeats = n_glue_target // n_glues_orig
            remainder = n_glue_target % n_glues_orig
            parts = [df_glues] * repeats
            if remainder > 0:
                parts.append(df_glues.sample(n=remainder, random_state=rng))
            df_glues = pd.concat(parts, ignore_index=True)

        print(f"  Oversampled glues: {n_glues_orig} → {len(df_glues)} "
              f"(target fraction: {glue_fraction:.0%})")

    # Merge
    df_merged = pd.concat([df_chembl, df_glues], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset="smiles")

    # Shuffle
    df_merged = df_merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    actual_frac = df_merged["is_glue"].mean()
    print(f"  Final dataset: {len(df_merged)} molecules "
          f"({actual_frac:.1%} glues)")

    return df_merged


def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the dataset.

    Args:
        df: DataFrame with property columns.

    Returns:
        Dictionary of statistics suitable for JSON serialisation.
    """
    props = [
        "molecular_weight", "logp", "hbd", "hba", "tpsa",
        "rotatable_bonds", "num_aromatic_rings", "num_rings",
        "num_heavy_atoms", "fraction_sp3", "qed",
    ]
    stats: dict = {"n_molecules": len(df)}

    for p in props:
        if p not in df.columns:
            continue
        vals = df[p].dropna()
        stats[p] = {
            "mean": round(float(vals.mean()), 3),
            "std": round(float(vals.std()), 3),
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
            "median": round(float(vals.median()), 3),
        }

    if "is_glue" in df.columns:
        stats["n_glues"] = int(df["is_glue"].sum())
        stats["glue_fraction"] = round(float(df["is_glue"].mean()), 4)

    return stats


def train_val_split(
    df: pd.DataFrame,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Split DataFrame into train and validation sets.

    Args:
        df: Full dataset.
        val_split: Fraction for validation.
        seed: Random seed.

    Returns:
        (df_train, df_val) tuple.
    """
    n_val = max(1, int(len(df) * val_split))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(df))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and merge training datasets"
    )
    parser.add_argument(
        "--chembl", type=str, default="data/raw/chembl_druglike.csv",
        help="Path to ChEMBL drug-like CSV",
    )
    parser.add_argument(
        "--glues", type=str, default="data/glue_chemotypes.csv",
        help="Path to glue chemotypes CSV",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/training_data.csv",
        help="Output path for merged dataset",
    )
    parser.add_argument(
        "--glue_fraction", type=float, default=0.15,
        help="Target fraction of glue molecules (default: 0.15)",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1,
        help="Fraction for validation split (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output",
    )
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        print(f"Output file already exists: {args.output}")
        print("Use --force to overwrite.")
        return

    # ── Load data ───────────────────────────────────────────────────
    print("Loading ChEMBL data …")
    df_chembl = load_smiles_from_csv(args.chembl)

    print("Loading glue chemotypes …")
    df_glues = load_smiles_from_csv(args.glues)

    # ── Merge and balance ───────────────────────────────────────────
    print("Merging and balancing …")
    df_merged = merge_and_balance(
        df_chembl, df_glues,
        glue_fraction=args.glue_fraction,
        seed=args.seed,
    )

    # ── Compute properties ──────────────────────────────────────────
    print("Computing molecular properties …")
    df_props = compute_all_properties(df_merged)

    # Keep is_glue column
    if "is_glue" in df_merged.columns:
        df_props["is_glue"] = df_merged["is_glue"].values[
            : len(df_props)
        ]

    # ── Train/val split ─────────────────────────────────────────────
    df_train, df_val = train_val_split(df_props, val_split=args.val_split, seed=args.seed)
    print(f"Train: {len(df_train)} molecules, Val: {len(df_val)} molecules")

    # ── Save ────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Full dataset
    df_props.to_csv(args.output, index=False)
    print(f"Saved full dataset to {args.output}")

    # Train/val splits
    train_path = args.output.replace(".csv", "_train.csv")
    val_path = args.output.replace(".csv", "_val.csv")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    print(f"Saved train split to {train_path}")
    print(f"Saved val split to {val_path}")

    # Statistics
    stats = compute_dataset_stats(df_props)
    stats_path = os.path.join(out_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset statistics to {stats_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total molecules: {stats['n_molecules']}")
    if "n_glues" in stats:
        print(f"Glue molecules:  {stats['n_glues']} ({stats['glue_fraction']:.1%})")
    for prop in ["molecular_weight", "logp", "hbd", "hba", "tpsa"]:
        if prop in stats:
            s = stats[prop]
            print(f"  {prop:20s}: {s['mean']:.2f} ± {s['std']:.2f} "
                  f"[{s['min']:.1f} – {s['max']:.1f}]")


if __name__ == "__main__":
    main()
