"""
Analysis script for generated molecules.

Creates plots and analysis of molecular property distributions.
"""
import argparse
import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.chemistry import get_molecular_properties, is_valid_molecule


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze generated molecules')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV with generated molecules')
    parser.add_argument('--training_data', type=str, default=None,
                        help='Optional training data for comparison')
    parser.add_argument('--output_dir', type=str, default='analysis',
                        help='Output directory for plots')
    return parser.parse_args()


def analyze_molecules(smiles_list, label='Generated'):
    """Compute properties for a list of molecules."""
    properties = []
    for smiles in smiles_list:
        if is_valid_molecule(smiles):
            props = get_molecular_properties(smiles)
            if props:
                props['source'] = label
                properties.append(props)
    return pd.DataFrame(properties)


def plot_property_distributions(df_gen, df_train=None, output_dir='analysis'):
    """Create property distribution plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    properties_to_plot = [
        ('molecular_weight', 'Molecular Weight (Da)', (100, 600)),
        ('logp', 'logP', (-2, 6)),
        ('tpsa', 'TPSA (Å²)', (0, 160)),
        ('hbd', 'H-bond Donors', (0, 6)),
        ('hba', 'H-bond Acceptors', (0, 12)),
        ('rotatable_bonds', 'Rotatable Bonds', (0, 12)),
        ('num_aromatic_rings', 'Aromatic Rings', (0, 5)),
        ('fraction_sp3', 'Fraction sp³', (0, 1)),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (prop, label, xlim) in enumerate(properties_to_plot):
        ax = axes[i]
        
        if prop in df_gen.columns:
            sns.histplot(
                df_gen[prop].dropna(), 
                ax=ax, 
                label='Generated',
                color='steelblue',
                alpha=0.7,
                kde=True,
            )
            
            if df_train is not None and prop in df_train.columns:
                sns.histplot(
                    df_train[prop].dropna(),
                    ax=ax,
                    label='Training',
                    color='coral',
                    alpha=0.5,
                    kde=True,
                )
                ax.legend()
        
        ax.set_xlabel(label)
        ax.set_xlim(xlim)
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'property_distributions.png'), dpi=150)
    plt.close()
    
    print(f"Saved property distributions to {output_dir}/property_distributions.png")


def plot_scatter_matrix(df, output_dir='analysis'):
    """Create scatter plot matrix of key properties."""
    os.makedirs(output_dir, exist_ok=True)
    
    props = ['molecular_weight', 'logp', 'tpsa', 'num_aromatic_rings']
    props = [p for p in props if p in df.columns]
    
    if len(props) < 2:
        return
    
    fig = sns.pairplot(df[props].dropna(), diag_kind='kde', corner=True)
    fig.savefig(os.path.join(output_dir, 'property_scatter_matrix.png'), dpi=150)
    plt.close()
    
    print(f"Saved scatter matrix to {output_dir}/property_scatter_matrix.png")


def print_summary_statistics(df, label='Generated'):
    """Print summary statistics."""
    print(f"\n{'='*50}")
    print(f"{label} Molecules Summary (n={len(df)})")
    print('='*50)
    
    stats_props = [
        'molecular_weight', 'logp', 'tpsa', 
        'hbd', 'hba', 'rotatable_bonds',
        'num_rings', 'num_aromatic_rings'
    ]
    
    for prop in stats_props:
        if prop in df.columns:
            values = df[prop].dropna()
            print(f"{prop:20s}: {values.mean():6.1f} ± {values.std():5.1f} "
                  f"[{values.min():5.1f}, {values.max():5.1f}]")


def main():
    args = parse_args()
    
    # Load generated molecules
    print(f"Loading generated molecules: {args.input}")
    df_gen = pd.read_csv(args.input)
    
    # Analyze if needed
    if 'molecular_weight' not in df_gen.columns:
        print("Computing properties for generated molecules...")
        df_gen = analyze_molecules(df_gen['smiles'].tolist(), 'Generated')
    
    # Load training data if provided
    df_train = None
    if args.training_data:
        print(f"Loading training data: {args.training_data}")
        df_train_raw = pd.read_csv(args.training_data)
        print("Computing properties for training molecules...")
        df_train = analyze_molecules(df_train_raw['smiles'].tolist(), 'Training')
    
    # Print summaries
    print_summary_statistics(df_gen, 'Generated')
    if df_train is not None:
        print_summary_statistics(df_train, 'Training')
    
    # Create plots
    print(f"\nCreating plots in: {args.output_dir}")
    plot_property_distributions(df_gen, df_train, args.output_dir)
    plot_scatter_matrix(df_gen, args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
