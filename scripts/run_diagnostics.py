"""
Diagnostic scripts for the molecular glue diffusion model.

Four tests to pinpoint generation failure modes:
  1. Graph round-trip integrity
  2. Loss component analysis (requires checkpoint)
  3. Generated graph inspection (requires checkpoint)
  4. Property computation validation

Run:
  python scripts/run_diagnostics.py --test roundtrip
  python scripts/run_diagnostics.py --test all --checkpoint path/to/checkpoint.pt
"""
import argparse
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
import torch


# ── Test 1: Graph Round-Trip Integrity ──────────────────────────────

REFERENCE_MOLECULES = {
    "Paracetamol":  "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Thalidomide":  "O=C1CCC(N1c2ccccc2C(=O)[O-])C(=O)[O-]",
    "Caffeine":     "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Anthracene":   "c1ccc2cc3ccccc3cc2c1",
}


def _get_props(smiles: str) -> dict:
    """Compute key properties from SMILES."""
    from utils.chemistry import get_molecular_properties
    return get_molecular_properties(smiles) or {}


def test_roundtrip():
    """SMILES → Graph → SMILES and compare properties."""
    from data.molecular_graph import smiles_to_graph, graph_to_smiles

    print("\n" + "=" * 60)
    print("TEST 1: Graph Round-Trip Integrity")
    print("=" * 60)

    passed = 0
    total = 0
    for name, smiles in REFERENCE_MOLECULES.items():
        total += 1
        print(f"\n--- {name}: {smiles} ---")

        graph = smiles_to_graph(smiles)
        if graph is None:
            print(f"  ❌ smiles_to_graph FAILED")
            continue

        reconstructed = graph_to_smiles(graph)
        if reconstructed is None:
            print(f"  ❌ graph_to_smiles FAILED (graph has {graph.x.shape[0]} nodes)")
            continue

        props_orig = _get_props(smiles)
        props_recon = _get_props(reconstructed)

        if not props_orig or not props_recon:
            print(f"  ❌ Property computation failed")
            continue

        checks = [
            ("HBD",            props_orig.get("hbd", -1),              props_recon.get("hbd", -1)),
            ("HBA",            props_orig.get("hba", -1),              props_recon.get("hba", -1)),
            ("TPSA",           round(props_orig.get("tpsa", -1), 1),   round(props_recon.get("tpsa", -1), 1)),
            ("RotBonds",       props_orig.get("rotatable_bonds", -1),  props_recon.get("rotatable_bonds", -1)),
            ("AromaticRings",  props_orig.get("num_aromatic_rings", -1), props_recon.get("num_aromatic_rings", -1)),
            ("MW",             round(props_orig.get("molecular_weight", -1), 1), round(props_recon.get("molecular_weight", -1), 1)),
        ]

        all_ok = True
        for prop_name, orig, recon in checks:
            match = "✓" if orig == recon else "✗"
            if orig != recon:
                all_ok = False
            print(f"  {match} {prop_name:15s}: original={orig}  reconstructed={recon}")

        print(f"  Reconstructed SMILES: {reconstructed}")
        if all_ok:
            passed += 1
            print(f"  ✅ PASSED")
        else:
            print(f"  ⚠️  PROPERTY MISMATCH")

    print(f"\nRound-trip results: {passed}/{total} molecules fully matched")
    return passed, total


# ── Test 2: Loss Component Analysis ─────────────────────────────────

def test_loss_components(checkpoint_path: str):
    """Load checkpoint, run forward pass, report loss magnitudes."""
    from config import Config
    from model.diffusion import MolecularDiffusion
    from data.molecular_graph import smiles_to_graph
    from torch_geometric.data import Batch

    print("\n" + "=" * 60)
    print("TEST 2: Loss Component Analysis")
    print("=" * 60)

    config = Config()
    config.device = "cpu"
    config.model.hidden_dim = 64
    config.model.num_layers = 2
    config.model.num_heads = 4

    model = MolecularDiffusion(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint — using randomly initialised model")

    model.eval()

    # Build a small batch from reference molecules
    graphs = []
    for smiles in REFERENCE_MOLECULES.values():
        g = smiles_to_graph(smiles)
        if g is not None:
            graphs.append(g)

    if len(graphs) < 2:
        print("  ❌ Not enough valid graphs for batch")
        return

    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        metrics = model.training_step(batch)

    print(f"\n  diffusion_loss:  {metrics.get('diffusion_loss', 'N/A')}")
    print(f"  valency_loss:    {metrics.get('valency_loss', 'N/A')}")
    print(f"  property_loss:   {metrics.get('property_loss', 'N/A')}")
    print(f"  total_loss:      {metrics.get('loss', 'N/A')}")

    # Ratios
    diff = float(metrics.get('diffusion_loss', 1.0))
    val = float(metrics.get('valency_loss', 0.0))
    prop = float(metrics.get('property_loss', 0.0))
    print(f"\n  valency / diffusion ratio:  {val / max(diff, 1e-8):.4f}")
    print(f"  property / diffusion ratio: {prop / max(diff, 1e-8):.4f}")


# ── Test 3: Generated Graph Inspection ──────────────────────────────

def test_generated_graphs(checkpoint_path: str):
    """Generate graphs and inspect atom/bond counts without SMILES conversion."""
    from config import Config, ATOM_TYPES, BOND_TYPES
    from model.diffusion import MolecularDiffusion

    print("\n" + "=" * 60)
    print("TEST 3: Generated Graph Inspection")
    print("=" * 60)

    config = Config()
    config.device = "cpu"
    config.model.hidden_dim = 64
    config.model.num_layers = 2
    config.model.num_heads = 4
    config.diffusion.num_timesteps = 50

    model = MolecularDiffusion(config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    model.eval()
    molecules = model.sample(num_molecules=5, num_atoms=15)

    for i, mol in enumerate(molecules):
        print(f"\n--- Molecule {i + 1} ---")
        atom_counts = {}
        for idx in mol.node_types.tolist():
            name = ATOM_TYPES[idx] if idx < len(ATOM_TYPES) else f"?{idx}"
            atom_counts[name] = atom_counts.get(name, 0) + 1
        print(f"  Atom counts: {atom_counts}")

        bond_counts = {}
        for idx in mol.edge_types.tolist():
            name = BOND_TYPES[idx] if idx < len(BOND_TYPES) else f"?{idx}"
            bond_counts[name] = bond_counts.get(name, 0) + 1
        print(f"  Bond counts: {bond_counts}")

        if hasattr(mol, "num_hs"):
            h_sum = mol.num_hs.sum().item()
            print(f"  Implicit H total: {h_sum}")

        # Flags
        if atom_counts.get("C", 0) == 0:
            print(f"  ⚠️  No carbon atoms!")
        if bond_counts.get("SINGLE", 0) == 0 and bond_counts.get("AROMATIC", 0) == 0:
            print(f"  ⚠️  No single or aromatic bonds!")
        none_bonds = bond_counts.get("NONE", 0)
        total_bonds = sum(bond_counts.values())
        if total_bonds > 0 and none_bonds / total_bonds > 0.5:
            print(f"  ⚠️  >50% of edges are NONE type ({none_bonds}/{total_bonds})")


# ── Test 4: Property Computation Validation ─────────────────────────

def test_property_pipeline():
    """Validate that property pipeline gives consistent results."""
    from data.dataset import _extract_property_tensor
    from config import PROPERTY_NAMES

    print("\n" + "=" * 60)
    print("TEST 4: Property Computation Validation")
    print("=" * 60)

    for name, smiles in REFERENCE_MOLECULES.items():
        print(f"\n--- {name}: {smiles} ---")
        props = _get_props(smiles)
        tensor = _extract_property_tensor(smiles)

        if props is None:
            print(f"  ❌ get_molecular_properties returned None")
            continue
        if tensor is None:
            print(f"  ❌ _extract_property_tensor returned None")
            continue

        print(f"  RDKit properties:")
        for k in ["molecular_weight", "logp", "hbd", "hba", "tpsa",
                   "num_aromatic_rings", "fraction_sp3"]:
            print(f"    {k:22s}: {props.get(k, 'N/A')}")

        print(f"  Normalized tensor (len={tensor.shape[0]}):")
        for i, pname in enumerate(PROPERTY_NAMES):
            print(f"    {pname:22s}: {tensor[i].item():.4f}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Molecular Glue Model Diagnostics")
    parser.add_argument("--test", type=str, default="roundtrip",
                        choices=["roundtrip", "loss", "graphs", "props", "all"],
                        help="Which diagnostic test to run")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (needed for loss/graphs tests)")
    args = parser.parse_args()

    if args.test in ("roundtrip", "all"):
        test_roundtrip()
    if args.test in ("loss", "all"):
        test_loss_components(args.checkpoint)
    if args.test in ("graphs", "all"):
        test_generated_graphs(args.checkpoint)
    if args.test in ("props", "all"):
        test_property_pipeline()

    print("\n✅ Diagnostics complete.\n")


if __name__ == "__main__":
    main()
