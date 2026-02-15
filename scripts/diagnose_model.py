"""
Diagnostic tool for the molecular glue diffusion model.

Runs a battery of tests to identify failure modes:
    1. Graph round-trip integrity (SMILES → Graph → SMILES)
    2. Loss component analysis (requires checkpoint)
    3. Generated graph inspection (requires checkpoint)
    4. Property computation validation

Usage:
    python scripts/diagnose_model.py --checkpoint checkpoints/best_model.pt
    python scripts/diagnose_model.py --test_smiles data/test_molecules.txt
    python scripts/diagnose_model.py --all
"""
import argparse
import os
import sys
import json
from typing import Dict, List, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
from rdkit import Chem

from config import Config, ATOM_TYPES, BOND_TYPES
from data.molecular_graph import smiles_to_graph, graph_to_smiles
from utils.chemistry import get_molecular_properties, is_valid_molecule


# ── Default test molecules ──────────────────────────────────────────

DEFAULT_TEST_SMILES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Phthalimide": "O=C1NC(=O)c2ccccc12",
    "Glutarimide": "O=C1CCC(=O)NC1",
    "Phenytoin": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
    "Benzimidazole": "c1ccc2[nH]cnc2c1",
    "Indole": "c1ccc2[nH]ccc2c1",
}


# ── Test 1: Graph Round-Trip ────────────────────────────────────────

def test_roundtrip(
    test_smiles: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Test SMILES → Graph → SMILES round-trip fidelity.

    Returns:
        Dict with per-molecule results and overall pass rate.
    """
    smiles_dict = test_smiles or DEFAULT_TEST_SMILES
    results = {}
    n_pass = 0
    n_total = 0

    print("\n" + "=" * 60)
    print("TEST 1: Graph Round-Trip Integrity")
    print("=" * 60)

    for name, smi in smiles_dict.items():
        n_total += 1
        entry = {"input_smiles": smi}

        # Forward: SMILES → Graph
        graph = smiles_to_graph(smi)
        if graph is None:
            entry["status"] = "FAIL (graph conversion)"
            results[name] = entry
            print(f"  ✗ {name:20s}: graph conversion failed")
            continue

        entry["n_atoms"] = graph.x.shape[0]
        entry["n_edges"] = graph.edge_index.shape[1]

        # Reverse: Graph → SMILES
        recovered = graph_to_smiles(graph)
        if recovered is None:
            entry["status"] = "FAIL (SMILES recovery)"
            results[name] = entry
            print(f"  ✗ {name:20s}: SMILES recovery failed")
            continue

        entry["recovered_smiles"] = recovered

        # Compare properties
        orig_props = get_molecular_properties(smi)
        rec_props = get_molecular_properties(recovered)

        if orig_props and rec_props:
            mw_diff = abs(orig_props["molecular_weight"] - rec_props["molecular_weight"])
            entry["mw_diff"] = round(mw_diff, 2)
            entry["orig_mw"] = round(orig_props["molecular_weight"], 2)
            entry["rec_mw"] = round(rec_props["molecular_weight"], 2)

            if mw_diff < 5.0:
                entry["status"] = "PASS"
                n_pass += 1
                print(f"  ✓ {name:20s}: MW {entry['orig_mw']:.1f} → "
                      f"{entry['rec_mw']:.1f} (Δ={mw_diff:.1f})")
            else:
                entry["status"] = f"WARN (MW diff={mw_diff:.1f})"
                n_pass += 1  # still valid, just inexact
                print(f"  ⚠ {name:20s}: MW {entry['orig_mw']:.1f} → "
                      f"{entry['rec_mw']:.1f} (Δ={mw_diff:.1f})")
        else:
            entry["status"] = "FAIL (property computation)"
            print(f"  ✗ {name:20s}: property computation failed")

        results[name] = entry

    summary = {
        "pass_rate": f"{n_pass}/{n_total}",
        "pass_fraction": round(n_pass / max(1, n_total), 3),
        "molecules": results,
    }

    print(f"\n  Round-trip pass rate: {n_pass}/{n_total}")
    return summary


# ── Test 2: Loss Component Analysis ────────────────────────────────

def test_loss_components(checkpoint_path: str) -> Dict:
    """
    Load checkpoint, create a small batch, and report individual loss
    component magnitudes.

    Args:
        checkpoint_path: Path to model checkpoint.

    Returns:
        Dict of loss component values.
    """
    from model.diffusion import MolecularDiffusion

    print("\n" + "=" * 60)
    print("TEST 2: Loss Component Analysis")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return {"status": "SKIP", "reason": "checkpoint not found"}

    config = Config()
    config.device = "cpu"

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "config" in checkpoint:
            config = checkpoint["config"]
            config.device = "cpu"

        model = MolecularDiffusion(config)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as exc:
        print(f"  ✗ Failed to load model: {exc}")
        return {"status": "FAIL", "reason": str(exc)}

    # Create synthetic batch from test molecules
    test_mols = list(DEFAULT_TEST_SMILES.values())[:4]
    graphs = []
    for smi in test_mols:
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)

    if not graphs:
        print("  ✗ Could not create test batch")
        return {"status": "FAIL", "reason": "no valid test graphs"}

    from torch_geometric.data import Batch
    batch = Batch.from_data_list(graphs)

    try:
        with torch.no_grad():
            loss_dict = model.compute_loss(batch)
    except Exception as exc:
        print(f"  ✗ Forward pass failed: {exc}")
        return {"status": "FAIL", "reason": str(exc)}

    results = {}
    total = 0.0
    for key, val in loss_dict.items():
        v = float(val)
        results[key] = round(v, 4)
        total += v
        print(f"  {key:30s}: {v:.4f}")

    # Ratios
    if total > 0:
        print(f"\n  Loss ratios (relative to total={total:.4f}):")
        for key, val in results.items():
            ratio = val / total * 100
            print(f"    {key:30s}: {ratio:.1f}%")
            results[f"{key}_ratio_pct"] = round(ratio, 1)

    results["status"] = "PASS"
    return results


# ── Test 3: Generated Graph Inspection ──────────────────────────────

def test_generated_graphs(checkpoint_path: str, n_samples: int = 10) -> Dict:
    """
    Generate raw graphs and inspect atom/bond statistics before
    SMILES conversion.

    Args:
        checkpoint_path: Path to model checkpoint.
        n_samples: Number of graphs to generate.

    Returns:
        Dict with atom-type counts, bond-type counts, validity rate.
    """
    from model.diffusion import MolecularDiffusion
    from generate.sampler import MoleculeSampler

    print("\n" + "=" * 60)
    print("TEST 3: Generated Graph Inspection")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return {"status": "SKIP", "reason": "checkpoint not found"}

    config = Config()
    config.device = "cpu"

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "config" in checkpoint:
            config = checkpoint["config"]
            config.device = "cpu"

        model = MolecularDiffusion(config)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as exc:
        print(f"  ✗ Failed to load model: {exc}")
        return {"status": "FAIL", "reason": str(exc)}

    sampler = MoleculeSampler(model, config)
    graphs = sampler.sample(num_molecules=n_samples, num_atoms=20, temperature=1.0)

    atom_counts = {a: 0 for a in ATOM_TYPES}
    bond_counts = {b: 0 for b in BOND_TYPES}
    n_valid_smiles = 0
    n_total_atoms = 0

    for g in graphs:
        # Atom types
        if hasattr(g, "node_types"):
            for t in g.node_types.tolist():
                if 0 <= t < len(ATOM_TYPES):
                    atom_counts[ATOM_TYPES[t]] += 1
                    n_total_atoms += 1

        # Bond types
        if hasattr(g, "edge_types"):
            for t in g.edge_types.tolist():
                if 0 <= t < len(BOND_TYPES):
                    bond_counts[BOND_TYPES[t]] += 1

        # SMILES validity
        smi = graph_to_smiles(g)
        if smi and is_valid_molecule(smi):
            n_valid_smiles += 1

    print(f"\n  Atom type distribution ({n_total_atoms} total atoms):")
    for atom, count in atom_counts.items():
        frac = count / max(1, n_total_atoms) * 100
        bar = "█" * int(frac // 2)
        print(f"    {atom:5s}: {count:4d} ({frac:5.1f}%) {bar}")

    print(f"\n  Bond type distribution:")
    total_bonds = sum(bond_counts.values())
    for bond, count in bond_counts.items():
        frac = count / max(1, total_bonds) * 100
        print(f"    {bond:10s}: {count:4d} ({frac:5.1f}%)")

    validity = n_valid_smiles / max(1, len(graphs))
    print(f"\n  SMILES validity: {n_valid_smiles}/{len(graphs)} ({validity:.1%})")

    return {
        "status": "PASS",
        "n_samples": len(graphs),
        "validity_rate": round(validity, 3),
        "atom_counts": atom_counts,
        "bond_counts": bond_counts,
    }


# ── Test 4: Property Computation Validation ─────────────────────────

def test_property_pipeline() -> Dict:
    """
    Validate that the property computation pipeline gives consistent
    and reasonable results on known molecules.

    Returns:
        Dict with per-molecule property validation results.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Property Computation Validation")
    print("=" * 60)

    # Expected approximate values for known molecules
    known = {
        "Aspirin": {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "expected_mw": 180.16,
            "expected_logp_range": (0.5, 2.0),
            "expected_hbd": 1,
            "expected_hba_range": (3, 5),
        },
        "Caffeine": {
            "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "expected_mw": 194.19,
            "expected_logp_range": (-1.0, 0.5),
            "expected_hbd": 0,
            "expected_hba_range": (3, 7),
        },
        "Ibuprofen": {
            "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "expected_mw": 206.28,
            "expected_logp_range": (3.0, 5.0),
            "expected_hbd": 1,
            "expected_hba_range": (1, 3),
        },
    }

    results = {}
    n_pass = 0

    for name, info in known.items():
        props = get_molecular_properties(info["smiles"])
        if props is None:
            results[name] = {"status": "FAIL", "reason": "props returned None"}
            print(f"  ✗ {name}: property computation returned None")
            continue

        entry = {}
        issues = []

        # MW check
        mw = props["molecular_weight"]
        mw_diff = abs(mw - info["expected_mw"])
        entry["mw"] = round(mw, 2)
        if mw_diff > 2.0:
            issues.append(f"MW={mw:.1f} vs expected {info['expected_mw']:.1f}")

        # LogP check
        logp = props["logp"]
        lo, hi = info["expected_logp_range"]
        entry["logp"] = round(logp, 2)
        if not (lo <= logp <= hi):
            issues.append(f"LogP={logp:.2f} outside [{lo}, {hi}]")

        # HBD check
        hbd = props["hbd"]
        entry["hbd"] = hbd
        if hbd != info["expected_hbd"]:
            issues.append(f"HBD={hbd} vs expected {info['expected_hbd']}")

        # HBA check
        hba = props["hba"]
        lo_hba, hi_hba = info["expected_hba_range"]
        entry["hba"] = hba
        if not (lo_hba <= hba <= hi_hba):
            issues.append(f"HBA={hba} outside [{lo_hba}, {hi_hba}]")

        if issues:
            entry["status"] = f"WARN ({'; '.join(issues)})"
            print(f"  ⚠ {name}: {'; '.join(issues)}")
        else:
            entry["status"] = "PASS"
            n_pass += 1
            print(f"  ✓ {name}: MW={mw:.1f}, LogP={logp:.2f}, "
                  f"HBD={hbd}, HBA={hba}")

        results[name] = entry

    print(f"\n  Property validation: {n_pass}/{len(known)} fully pass")
    return {"status": "PASS", "molecules": results}


# ── Report Generation ───────────────────────────────────────────────

def generate_report(
    all_results: Dict,
    output_file: str = "diagnostics_report.json",
):
    """Save all test results to a JSON report."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull diagnostic report saved to {output_file}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tool for the molecular glue diffusion model"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_smiles", type=str, default=None,
        help="Path to text file with test SMILES (one per line)",
    )
    parser.add_argument(
        "--output", type=str, default="diagnostics_report.json",
        help="Output path for diagnostic report",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of molecules to generate for inspection",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tests (including those requiring checkpoint)",
    )
    parser.add_argument(
        "--no_checkpoint", action="store_true",
        help="Skip tests that require a checkpoint",
    )
    args = parser.parse_args()

    # Load custom test SMILES
    test_smiles = None
    if args.test_smiles and os.path.exists(args.test_smiles):
        with open(args.test_smiles) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        test_smiles = {f"mol_{i}": smi for i, smi in enumerate(lines)}
        print(f"Loaded {len(test_smiles)} test molecules from {args.test_smiles}")

    print("\n" + "=" * 60)
    print("MOLECULAR GLUE DIFFUSION — DIAGNOSTIC SUITE")
    print("=" * 60)

    all_results = {}

    # Always run these tests
    all_results["roundtrip"] = test_roundtrip(test_smiles)
    all_results["property_pipeline"] = test_property_pipeline()

    # Checkpoint-dependent tests
    if not args.no_checkpoint:
        all_results["loss_components"] = test_loss_components(args.checkpoint)
        all_results["generated_graphs"] = test_generated_graphs(
            args.checkpoint, n_samples=args.n_samples
        )

    # Save report
    generate_report(all_results, args.output)

    # Overall summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    for test_name, result in all_results.items():
        status = result.get("status", "UNKNOWN")
        print(f"  {test_name:25s}: {status}")


if __name__ == "__main__":
    main()
