"""
SMILES → 2D Molecular Graph Visualizer

This script lets you input SMILES strings and visualizes them as 2D molecular
graphs showing:
  1. The RDKit 2D structure (classic chemistry drawing)
  2. The graph representation used by the Graph Transformer
     (nodes = atoms, edges = bonds, with feature annotations)

Usage:
  python visualize_smiles_graph.py                          # interactive mode
  python visualize_smiles_graph.py "CCO" "c1ccccc1"         # from CLI args
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx

# ── RDKit imports ──────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors

# ── Project imports ────────────────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import (
    ATOM_TYPES, ATOM_TO_IDX,
    BOND_TYPES, BOND_TO_IDX,
    CHARGES, CHARGE_TO_IDX,
    HYBRIDIZATIONS, HYBRID_TO_IDX,
    PROPERTY_NAMES,
)

# ── Color palettes ─────────────────────────────────────────────────────────────
ATOM_COLORS = {
    'C':  '#4a4a4a',   # dark grey
    'N':  '#3050F8',   # blue
    'O':  '#FF0D0D',   # red
    'S':  '#FFFF30',   # yellow
    'F':  '#90E050',   # light green
    'Cl': '#1FF01F',   # green
    'Br': '#A62929',   # dark red
    'P':  '#FF8000',   # orange
    'I':  '#940094',   # purple
    'Other': '#808080', # grey
}

BOND_STYLES = {
    'SINGLE':   {'color': '#555555', 'width': 1.5, 'style': '-'},
    'DOUBLE':   {'color': '#2196F3', 'width': 2.5, 'style': '-'},
    'TRIPLE':   {'color': '#F44336', 'width': 3.5, 'style': '-'},
    'AROMATIC': {'color': '#9C27B0', 'width': 2.0, 'style': '--'},
}

HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP:   'SP',
    Chem.rdchem.HybridizationType.SP2:  'SP2',
    Chem.rdchem.HybridizationType.SP3:  'SP3',
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Core: SMILES → feature vectors (same encoding the Graph Transformer uses)
# ═══════════════════════════════════════════════════════════════════════════════

def smiles_to_graph_features(smiles: str):
    """
    Convert a SMILES string into the node / edge feature tensors that the
    Graph Transformer would receive.

    Returns
    -------
    mol        : RDKit Mol object (with 2D coords)
    node_info  : list of dicts  – per-atom features
    edge_info  : list of dicts  – per-bond features
    props      : dict           – molecular-level properties
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)       # add explicit H (for feature calc)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol = Chem.RemoveHs(mol)    # back to heavy-atom graph

    # Ensure 2D coords exist
    AllChem.Compute2DCoords(mol)

    ri = mol.GetRingInfo()

    # ── Node (atom) features ───────────────────────────────────────────────
    node_info = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_type = symbol if symbol in ATOM_TO_IDX else 'Other'

        hyb_raw = atom.GetHybridization()
        hyb = HYBRIDIZATION_MAP.get(hyb_raw, 'OTHER')

        charge = atom.GetFormalCharge()
        charge = max(-2, min(2, charge))

        node_info.append({
            'idx':          atom.GetIdx(),
            'symbol':       symbol,
            'atom_type':    atom_type,
            'atom_type_oh': _one_hot(ATOM_TO_IDX[atom_type], len(ATOM_TYPES)),
            'charge':       charge,
            'charge_oh':    _one_hot(CHARGE_TO_IDX[charge], len(CHARGES)),
            'hybrid':       hyb,
            'hybrid_oh':    _one_hot(HYBRID_TO_IDX[hyb], len(HYBRIDIZATIONS)),
            'aromatic':     int(atom.GetIsAromatic()),
            'in_ring':      int(atom.IsInRing()),
            'conjugated':   0,  # placeholder – per-atom conjugated isn't trivial
            'num_hs':       atom.GetTotalNumHs(),
        })

    # ── Edge (bond) features ───────────────────────────────────────────────
    edge_info = []
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.SINGLE:
            btype = 'SINGLE'
        elif bt == Chem.rdchem.BondType.DOUBLE:
            btype = 'DOUBLE'
        elif bt == Chem.rdchem.BondType.TRIPLE:
            btype = 'TRIPLE'
        elif bt == Chem.rdchem.BondType.AROMATIC:
            btype = 'AROMATIC'
        else:
            btype = 'NONE'

        edge_info.append({
            'src':        bond.GetBeginAtomIdx(),
            'dst':        bond.GetEndAtomIdx(),
            'bond_type':  btype,
            'bond_oh':    _one_hot(BOND_TO_IDX[btype], len(BOND_TYPES)),
            'aromatic':   int(bond.GetIsAromatic()),
            'conjugated': int(bond.GetIsConjugated()),
            'in_ring':    int(bond.IsInRing()),
        })

    # ── Molecular-level properties (the 8 conditioning values) ─────────────
    mol_h = Chem.AddHs(mol)
    props = {
        'molecular_weight':    round(Descriptors.ExactMolWt(mol), 2),
        'logp':                round(Descriptors.MolLogP(mol), 2),
        'num_aromatic_rings':  rdMolDescriptors.CalcNumAromaticRings(mol),
        'hbd':                 rdMolDescriptors.CalcNumHBD(mol),
        'hba':                 rdMolDescriptors.CalcNumHBA(mol),
        'fraction_sp3':        round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        'tpsa':                round(Descriptors.TPSA(mol), 2),
        'sa_score':            None,  # needs SA_Score module; skip
    }

    return mol, node_info, edge_info, props


def _one_hot(idx, length):
    """Return a simple one-hot list."""
    v = [0] * length
    v[idx] = 1
    return v


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_molecule(smiles: str, save_path: str = None):
    """
    Draw a side-by-side figure:
      Left  – classic 2D structure (RDKit)
      Right – graph representation with feature annotations
    """
    mol, node_info, edge_info, props = smiles_to_graph_features(smiles)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"SMILES:  {smiles}", fontsize=14, fontweight='bold', y=0.98)

    # ── Left panel: RDKit 2D ───────────────────────────────────────────────
    _draw_rdkit_2d(mol, axes[0])

    # ── Right panel: Graph Transformer view ────────────────────────────────
    _draw_graph_view(mol, node_info, edge_info, axes[1])

    # ── Property table below ───────────────────────────────────────────────
    prop_text = "  │  ".join(
        f"{k}: {v}" for k, v in props.items() if v is not None
    )
    fig.text(0.5, 0.02, prop_text, ha='center', fontsize=10,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0',
                       edgecolor='#cccccc'))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")

    plt.show()


def _draw_rdkit_2d(mol, ax):
    """Render the classic 2D structure on a matplotlib axes."""
    from rdkit.Chem import Draw
    from io import BytesIO
    from PIL import Image

    img = Draw.MolToImage(mol, size=(600, 500))
    ax.imshow(img)
    ax.set_title("RDKit 2D Structure", fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')


def _draw_graph_view(mol, node_info, edge_info, ax):
    """Draw the molecular graph the way the Graph Transformer sees it."""

    # Build NetworkX graph
    G = nx.Graph()
    for n in node_info:
        G.add_node(n['idx'], **n)
    for e in edge_info:
        G.add_edge(e['src'], e['dst'], **e)

    # Use RDKit 2D coordinates for layout
    conf = mol.GetConformer()
    pos = {}
    for i in range(mol.GetNumAtoms()):
        pt = conf.GetAtomPosition(i)
        pos[i] = (pt.x, pt.y)

    # ── Draw edges ─────────────────────────────────────────────────────────
    for e in edge_info:
        btype = e['bond_type']
        style = BOND_STYLES.get(btype, BOND_STYLES['SINGLE'])
        x0, y0 = pos[e['src']]
        x1, y1 = pos[e['dst']]
        ax.plot([x0, x1], [y0, y1],
                color=style['color'],
                linewidth=style['width'],
                linestyle=style['style'],
                zorder=1)

        # Bond label at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, btype[0], fontsize=7, ha='center', va='center',
                color=style['color'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.8),
                zorder=3)

    # ── Draw nodes ─────────────────────────────────────────────────────────
    for n in node_info:
        x, y = pos[n['idx']]
        color = ATOM_COLORS.get(n['atom_type'], '#808080')

        # Ring atoms get a highlighted border
        edge_color = '#FFD700' if n['in_ring'] else color
        edge_width = 3.0 if n['in_ring'] else 1.5

        circle = plt.Circle((x, y), 0.25, facecolor=color,
                             edgecolor=edge_color, linewidth=edge_width,
                             zorder=4)
        ax.add_patch(circle)

        # Atom label
        ax.text(x, y, n['symbol'], fontsize=11, ha='center', va='center',
                color='white', fontweight='bold', zorder=5)

        # Feature annotation (small text beneath)
        ann = f"{n['hybrid']}  q={n['charge']:+d}  H={n['num_hs']}"
        if n['aromatic']:
            ann += "  ★"
        ax.text(x, y - 0.38, ann, fontsize=6, ha='center', va='top',
                color='#333333', zorder=5,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          edgecolor='#dddddd', alpha=0.85))

    ax.set_title("Graph Transformer Representation", fontsize=13,
                 fontweight='bold', pad=10)
    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.axis('off')

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_elements = []
    for btype, style in BOND_STYLES.items():
        legend_elements.append(
            Line2D([0], [0], color=style['color'], linewidth=style['width'],
                   linestyle=style['style'], label=btype)
        )
    legend_elements.append(
        mpatches.Patch(facecolor='none', edgecolor='#FFD700',
                       linewidth=2.5, label='Ring atom')
    )
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor='#cccccc')


def print_feature_table(smiles: str):
    """Print a detailed text table of the node / edge features."""
    mol, node_info, edge_info, props = smiles_to_graph_features(smiles)

    print(f"\n{'═' * 70}")
    print(f"  SMILES: {smiles}")
    print(f"  Canonical: {Chem.MolToSmiles(mol)}")
    print(f"{'═' * 70}")

    # ── Node features ──────────────────────────────────────────────────────
    print(f"\n  {'─' * 66}")
    print(f"  NODE FEATURES  ({len(node_info)} atoms)")
    print(f"  {'─' * 66}")
    header = f"  {'Idx':>3}  {'Sym':>4}  {'Type':>6}  {'Hyb':>4}  " \
             f"{'Chg':>4}  {'Aro':>3}  {'Ring':>4}  {'nH':>3}  {'One-hot (atom type)'}"
    print(header)
    print(f"  {'─' * 66}")

    for n in node_info:
        oh = ''.join(str(x) for x in n['atom_type_oh'])
        print(f"  {n['idx']:>3}  {n['symbol']:>4}  {n['atom_type']:>6}  "
              f"{n['hybrid']:>4}  {n['charge']:>+4d}  "
              f"{'✓' if n['aromatic'] else '·':>3}  "
              f"{'✓' if n['in_ring'] else '·':>4}  "
              f"{n['num_hs']:>3}  [{oh}]")

    # ── Edge features ──────────────────────────────────────────────────────
    print(f"\n  {'─' * 66}")
    print(f"  EDGE FEATURES  ({len(edge_info)} bonds)")
    print(f"  {'─' * 66}")
    header = f"  {'Src':>3} → {'Dst':>3}  {'Type':>8}  {'Aro':>3}  " \
             f"{'Conj':>4}  {'Ring':>4}  {'One-hot (bond type)'}"
    print(header)
    print(f"  {'─' * 66}")

    for e in edge_info:
        oh = ''.join(str(x) for x in e['bond_oh'])
        print(f"  {e['src']:>3} → {e['dst']:>3}  {e['bond_type']:>8}  "
              f"{'✓' if e['aromatic'] else '·':>3}  "
              f"{'✓' if e['conjugated'] else '·':>4}  "
              f"{'✓' if e['in_ring'] else '·':>4}  [{oh}]")

    # ── Molecular properties ───────────────────────────────────────────────
    print(f"\n  {'─' * 66}")
    print(f"  MOLECULAR PROPERTIES (conditioning vector)")
    print(f"  {'─' * 66}")
    for k, v in props.items():
        if v is not None:
            print(f"    {k:<22}: {v}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize SMILES as 2D molecular graphs "
                    "(the way the Graph Transformer sees them)."
    )
    parser.add_argument(
        'smiles', nargs='*',
        help="One or more SMILES strings.  If none given, enters interactive mode."
    )
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help="Directory to save PNG figures (optional)."
    )
    args = parser.parse_args()

    smiles_list = args.smiles

    # Interactive mode
    if not smiles_list:
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║   SMILES → Graph Transformer Visualizer             ║")
        print("║   Type a SMILES string and press Enter.             ║")
        print("║   Type 'quit' or 'q' to exit.                      ║")
        print("╚══════════════════════════════════════════════════════╝\n")

        while True:
            try:
                smiles = input("SMILES> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not smiles or smiles.lower() in ('quit', 'q', 'exit'):
                print("Bye!")
                break

            try:
                print_feature_table(smiles)
                save_path = None
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    safe = smiles.replace('/', '_').replace('\\', '_')[:30]
                    save_path = os.path.join(args.save_dir, f"graph_{safe}.png")
                visualize_molecule(smiles, save_path=save_path)
            except Exception as exc:
                print(f"  ✗ Error: {exc}")

    # Batch mode (CLI args)
    else:
        for smiles in smiles_list:
            try:
                print_feature_table(smiles)
                save_path = None
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    safe = smiles.replace('/', '_').replace('\\', '_')[:30]
                    save_path = os.path.join(args.save_dir, f"graph_{safe}.png")
                visualize_molecule(smiles, save_path=save_path)
            except Exception as exc:
                print(f"  ✗ Error processing '{smiles}': {exc}")


if __name__ == "__main__":
    main()
