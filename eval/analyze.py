"""
Analysis and visualization for generated molecules.
"""
import os
import argparse
from typing import Optional, List
import pandas as pd
import numpy as np

import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.chemistry import get_molecular_properties, get_murcko_scaffold


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze generated molecules')
    parser.add_argument('--input', required=True, help='Input CSV with generated SMILES')
    parser.add_argument('--training_data', default=None, help='Training data CSV for comparison')
    parser.add_argument('--output_dir', default='analysis', help='Output directory for plots')
    parser.add_argument('--smiles_column', default='smiles', help='SMILES column name')
    return parser.parse_args()


def analyze_molecules(smiles_list: List[str]) -> pd.DataFrame:
    """Compute properties for a list of molecules."""
    records = []
    for smiles in smiles_list:
        props = get_molecular_properties(smiles)
        if props:
            props['smiles'] = smiles
            props['scaffold'] = get_murcko_scaffold(smiles) or 'unknown'
            records.append(props)
    return pd.DataFrame(records)


def plot_property_distributions(
    df_gen: pd.DataFrame,
    df_train: Optional[pd.DataFrame] = None,
    output_dir: str = 'analysis',
):
    """Plot property distributions."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    properties = [
        'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba',
        'num_aromatic_rings', 'rotatable_bonds', 'fraction_sp3', 'qed',
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, prop in enumerate(properties):
        if prop not in df_gen.columns:
            continue

        ax = axes[idx]
        sns.histplot(df_gen[prop], ax=ax, label='Generated', kde=True, alpha=0.5, color='blue')
        if df_train is not None and prop in df_train.columns:
            sns.histplot(df_train[prop], ax=ax, label='Training', kde=True, alpha=0.5, color='orange')
        ax.set_title(prop.replace('_', ' ').title())
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'property_distributions.png'), dpi=150)
    plt.close()
    print(f"Saved property distributions to {output_dir}/property_distributions.png")


def plot_scatter_matrix(df: pd.DataFrame, output_dir: str = 'analysis'):
    """Plot scatter matrix of key properties."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    props = ['molecular_weight', 'logp', 'tpsa', 'num_aromatic_rings', 'qed']
    available = [p for p in props if p in df.columns]

    if len(available) >= 2:
        pd.plotting.scatter_matrix(df[available], figsize=(12, 12), alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scatter_matrix.png'), dpi=150)
        plt.close()
        print(f"Saved scatter matrix to {output_dir}/scatter_matrix.png")


def plot_scaffold_distribution(df: pd.DataFrame, output_dir: str = 'analysis', top_n: int = 15):
    """Plot distribution of top Murcko scaffolds."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    if 'scaffold' not in df.columns:
        return

    scaffold_counts = df['scaffold'].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    scaffold_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Count')
    ax.set_title(f'Top {top_n} Murcko Scaffolds')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaffold_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved scaffold distribution to {output_dir}/scaffold_distribution.png")


def print_summary_statistics(df: pd.DataFrame, label: str = "Generated"):
    """Print summary statistics."""
    print(f"\n--- {label} Summary ({len(df)} molecules) ---")

    props = [
        'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba',
        'num_aromatic_rings', 'rotatable_bonds', 'fraction_sp3', 'qed',
    ]

    for prop in props:
        if prop in df.columns:
            values = df[prop].dropna()
            print(f"  {prop:25s}: {values.mean():.2f} ± {values.std():.2f} "
                  f"[{values.min():.2f}, {values.max():.2f}]")

    if 'scaffold' in df.columns:
        n_unique = df['scaffold'].nunique()
        print(f"  {'unique_scaffolds':25s}: {n_unique} ({n_unique / len(df) * 100:.1f}%)")


if __name__ == '__main__':
    args = parse_args()

    # Load generated molecules
    df = pd.read_csv(args.input)
    smiles = df[args.smiles_column].dropna().tolist()
    df_gen = analyze_molecules(smiles)

    # Load training data for comparison
    df_train = None
    if args.training_data:
        train_df = pd.read_csv(args.training_data)
        train_smiles = train_df['smiles'].dropna().tolist()
        df_train = analyze_molecules(train_smiles)
        print_summary_statistics(df_train, "Training")

    print_summary_statistics(df_gen, "Generated")

    # Plot
    plot_property_distributions(df_gen, df_train, args.output_dir)
    plot_scatter_matrix(df_gen, args.output_dir)
    plot_scaffold_distribution(df_gen, args.output_dir)

    print(f"\nAnalysis complete. Results in {args.output_dir}/")
