"""
Comprehensive visualization for comparing generated and training molecules.

Functions:
    plot_property_distributions  – overlay histograms + KDE
    plot_molecule_grid           – 2D molecule grid coloured by QED
    plot_scatter_matrix          – pairwise scatter by molecule type
    plot_scaffold_distribution   – top-N Murcko scaffold bar chart
    plot_correlation_heatmap     – property correlation heatmap
    generate_full_report         – run everything and save to output_dir
"""
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.chemistry import (
    get_molecular_properties,
    get_murcko_scaffold,
    calculate_qed,
)


# ── Property calculation helpers ────────────────────────────────────

def _smiles_to_props_df(smiles_list: List[str], label: str = "") -> pd.DataFrame:
    """Convert a list of SMILES to a properties DataFrame."""
    records = []
    for smi in smiles_list:
        props = get_molecular_properties(smi)
        if props is not None:
            props["smiles"] = smi
            props["scaffold"] = get_murcko_scaffold(smi) or "none"
            if label:
                props["source"] = label
            records.append(props)
    return pd.DataFrame(records)


# ── 1. Property Distributions ──────────────────────────────────────

PROPERTY_DISPLAY = {
    "molecular_weight": "Molecular Weight (Da)",
    "logp": "LogP",
    "hbd": "H-Bond Donors",
    "hba": "H-Bond Acceptors",
    "tpsa": "TPSA (Å²)",
    "num_rings": "Total Rings",
    "num_aromatic_rings": "Aromatic Rings",
    "fraction_sp3": "Fraction sp³",
    "qed": "QED",
}


def plot_property_distributions(
    gen_smiles: List[str],
    train_smiles: List[str],
    output_file: str = "property_distributions.png",
):
    """
    Overlay histograms and KDE curves of key properties for generated
    vs. training molecules.

    Args:
        gen_smiles: Generated SMILES list.
        train_smiles: Training SMILES list.
        output_file: Path for saved figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_gen = _smiles_to_props_df(gen_smiles, "Generated")
    df_train = _smiles_to_props_df(train_smiles, "Training")

    props = list(PROPERTY_DISPLAY.keys())

    n_props = len(props)
    ncols = 3
    nrows = (n_props + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, prop in enumerate(props):
        ax = axes[idx]
        if prop in df_gen.columns:
            sns.histplot(
                df_gen[prop].dropna(), ax=ax, label="Generated",
                kde=True, stat="density", alpha=0.45, color="#4C72B0",
            )
            g_mean = df_gen[prop].mean()
            g_std = df_gen[prop].std()
            ax.axvline(g_mean, color="#4C72B0", ls="--", lw=1)
            ax.text(
                0.98, 0.95,
                f"Gen: {g_mean:.2f}±{g_std:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#4C72B0",
            )
        if prop in df_train.columns:
            sns.histplot(
                df_train[prop].dropna(), ax=ax, label="Training",
                kde=True, stat="density", alpha=0.40, color="#DD8452",
            )
            t_mean = df_train[prop].mean()
            t_std = df_train[prop].std()
            ax.axvline(t_mean, color="#DD8452", ls="--", lw=1)
            ax.text(
                0.98, 0.85,
                f"Train: {t_mean:.2f}±{t_std:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#DD8452",
            )
        ax.set_title(PROPERTY_DISPLAY.get(prop, prop))
        ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Property Distributions: Generated vs Training", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved property distributions to {output_file}")


# ── 2. Molecule Grid ───────────────────────────────────────────────

def plot_molecule_grid(
    smiles_list: List[str],
    output_file: str = "molecule_grid.png",
    n_mols: int = 20,
    mols_per_row: int = 5,
):
    """
    Render a grid of 2D molecule structures colour-coded by QED score.

    Args:
        smiles_list: SMILES to draw.
        output_file: Output image path.
        n_mols: Maximum number of molecules to show.
        mols_per_row: Molecules per row in the grid.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem

    mols, legends = [], []
    for smi in smiles_list[:n_mols]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        AllChem.Compute2DCoords(mol)
        qed = calculate_qed(mol)
        mols.append(mol)
        legends.append(f"QED={qed:.2f}")

    if not mols:
        print("No valid molecules to draw.")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(300, 300),
        legends=legends,
    )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    img.save(output_file)
    print(f"Saved molecule grid ({len(mols)} mols) to {output_file}")


# ── 3. Scatter Matrix ──────────────────────────────────────────────

def plot_scatter_matrix(
    gen_smiles: List[str],
    train_smiles: List[str],
    output_file: str = "scatter_matrix.png",
    glue_smiles: Optional[List[str]] = None,
):
    """
    Pairwise scatter plots of key properties coloured by molecule type.

    Args:
        gen_smiles: Generated SMILES.
        train_smiles: Training SMILES.
        output_file: Output image path.
        glue_smiles: Optional known-glue SMILES (plotted separately).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_gen = _smiles_to_props_df(gen_smiles, "Generated")
    df_train = _smiles_to_props_df(train_smiles, "Training")
    dfs = [df_gen, df_train]
    if glue_smiles:
        df_glue = _smiles_to_props_df(glue_smiles, "Known Glues")
        dfs.append(df_glue)

    df_all = pd.concat(dfs, ignore_index=True)

    props = ["molecular_weight", "logp", "tpsa", "num_aromatic_rings", "qed"]
    available = [p for p in props if p in df_all.columns]

    if len(available) < 2:
        print("Not enough properties to create scatter matrix.")
        return

    palette = {"Generated": "#4C72B0", "Training": "#DD8452", "Known Glues": "#55A868"}

    g = sns.pairplot(
        df_all,
        vars=available,
        hue="source",
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 12},
        height=2.2,
    )
    g.fig.suptitle("Property Scatter Matrix", y=1.02)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    g.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved scatter matrix to {output_file}")


# ── 4. Scaffold Distribution ───────────────────────────────────────

def plot_scaffold_distribution(
    gen_smiles: List[str],
    output_file: str = "scaffold_distribution.png",
    train_smiles: Optional[List[str]] = None,
    top_n: int = 15,
):
    """
    Bar chart of top-N Murcko scaffolds.

    Args:
        gen_smiles: Generated SMILES.
        output_file: Output image path.
        train_smiles: Optional training SMILES for comparison.
        top_n: Number of top scaffolds to show.
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    gen_scaffolds = [get_murcko_scaffold(s) or "none" for s in gen_smiles]
    gen_counts = Counter(gen_scaffolds).most_common(top_n)

    scaffolds = [s for s, _ in gen_counts]
    gen_vals = [c for _, c in gen_counts]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    x = np.arange(len(scaffolds))
    width = 0.35

    ax.barh(x, gen_vals, width, label="Generated", color="#4C72B0", alpha=0.8)

    if train_smiles:
        train_scaffolds = [get_murcko_scaffold(s) or "none" for s in train_smiles]
        train_counter = Counter(train_scaffolds)
        train_vals = [train_counter.get(s, 0) for s in scaffolds]
        ax.barh(x + width, train_vals, width, label="Training", color="#DD8452", alpha=0.8)

    ax.set_yticks(x + width / 2)
    # Truncate long SMILES for labels
    labels = [s[:30] + "…" if len(s) > 30 else s for s in scaffolds]
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_n} Murcko Scaffolds")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scaffold distribution to {output_file}")


# ── 5. Correlation Heatmap ─────────────────────────────────────────

def plot_correlation_heatmap(
    smiles_list: List[str],
    output_file: str = "correlation_heatmap.png",
    label: str = "",
):
    """
    Heatmap of pairwise correlations between molecular properties.

    Args:
        smiles_list: SMILES list.
        output_file: Output image path.
        label: Title annotation (e.g. "Generated" or "Training").
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = _smiles_to_props_df(smiles_list)
    props = [
        "molecular_weight", "logp", "hbd", "hba", "tpsa",
        "num_aromatic_rings", "rotatable_bonds", "fraction_sp3", "qed",
    ]
    available = [p for p in props if p in df.columns]
    if len(available) < 3:
        print("Not enough properties for correlation heatmap.")
        return

    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax, square=True,
        linewidths=0.5,
    )
    title = "Property Correlations"
    if label:
        title += f" ({label})"
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation heatmap to {output_file}")


# ── 6. Full Report ─────────────────────────────────────────────────

def generate_full_report(
    gen_smiles: List[str],
    train_smiles: List[str],
    output_dir: str = "outputs/report",
    glue_smiles: Optional[List[str]] = None,
):
    """
    Generate all visualisation outputs into *output_dir*.

    Args:
        gen_smiles: Generated SMILES.
        train_smiles: Training SMILES.
        output_dir: Directory for all output images.
        glue_smiles: Optional known-glue SMILES for extra comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"GENERATING FULL REPORT → {output_dir}/")
    print(f"{'='*50}\n")

    plot_property_distributions(
        gen_smiles, train_smiles,
        os.path.join(output_dir, "property_distributions.png"),
    )

    plot_molecule_grid(
        gen_smiles,
        os.path.join(output_dir, "molecule_grid_generated.png"),
    )

    plot_scatter_matrix(
        gen_smiles, train_smiles,
        os.path.join(output_dir, "scatter_matrix.png"),
        glue_smiles=glue_smiles,
    )

    plot_scaffold_distribution(
        gen_smiles,
        os.path.join(output_dir, "scaffold_distribution.png"),
        train_smiles=train_smiles,
    )

    plot_correlation_heatmap(
        gen_smiles,
        os.path.join(output_dir, "correlation_heatmap_generated.png"),
        label="Generated",
    )
    plot_correlation_heatmap(
        train_smiles,
        os.path.join(output_dir, "correlation_heatmap_training.png"),
        label="Training",
    )

    print(f"\n✅ Full report saved to {output_dir}/")
