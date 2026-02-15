"""
Download drug-like molecules from ChEMBL for training the diffusion model.

Usage:
    pip install chembl_webresource_client   # if not already installed
    python scripts/collect_data.py --target_size 50000
    python scripts/collect_data.py --target_size 100 --output test_download.csv

Saves a CSV with SMILES and computed molecular properties.
"""
import argparse
import os
import sys
import time
import csv

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

from utils.chemistry import get_molecular_properties, canonicalize_smiles


# ── Property filters ────────────────────────────────────────────────

DEFAULT_FILTERS = {
    "mw_min": 200,
    "mw_max": 500,
    "logp_min": 0,
    "logp_max": 5,
    "hbd_max": 5,
    "hba_max": 10,
    "tpsa_max": 140,
    "aromatic_rings_min": 1,
    "rotatable_bonds_max": 10,
    "heavy_atoms_max": 35,
}


def passes_filters(props: dict, filters: dict = None) -> bool:
    """
    Check if computed properties pass all drug-like filters.

    Args:
        props: Dictionary from ``compute_properties``.
        filters: Override default filter thresholds.

    Returns:
        True if the molecule passes every filter.
    """
    f = {**DEFAULT_FILTERS, **(filters or {})}
    mw = props.get("molecular_weight", 0)
    if not (f["mw_min"] <= mw <= f["mw_max"]):
        return False
    logp = props.get("logp", -999)
    if not (f["logp_min"] <= logp <= f["logp_max"]):
        return False
    if props.get("hbd", 99) > f["hbd_max"]:
        return False
    if props.get("hba", 99) > f["hba_max"]:
        return False
    if props.get("tpsa", 999) > f["tpsa_max"]:
        return False
    if props.get("num_aromatic_rings", 0) < f["aromatic_rings_min"]:
        return False
    if props.get("rotatable_bonds", 99) > f["rotatable_bonds_max"]:
        return False
    if props.get("num_heavy_atoms", 99) > f["heavy_atoms_max"]:
        return False
    return True


def compute_properties(mol) -> dict:
    """
    Compute molecular properties for a validated RDKit Mol.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dictionary with MW, LogP, HBD, HBA, TPSA, RotBonds,
        AromaticRings, Rings, HeavyAtoms, Fsp3.
    """
    props = get_molecular_properties(mol)
    if props is None:
        return {}
    return props


def download_with_retry(query_fn, max_retries: int = 3, backoff: float = 5.0):
    """
    Execute *query_fn* with exponential back-off on failure.

    Args:
        query_fn: Callable that returns an iterable of records.
        max_retries: Maximum number of retry attempts.
        backoff: Initial wait in seconds (doubled each retry).

    Returns:
        Result of query_fn().

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    wait = backoff
    for attempt in range(1, max_retries + 1):
        try:
            return query_fn()
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Query failed after {max_retries} retries: {exc}"
                ) from exc
            print(f"  ⚠ Attempt {attempt} failed ({exc}), retrying in {wait:.0f}s …")
            time.sleep(wait)
            wait *= 2


def download_chembl_druglike(
    target_size: int = 50_000,
    output_file: str = "data/raw/chembl_druglike.csv",
    force: bool = False,
    batch_log_interval: int = 2000,
):
    """
    Download drug-like molecules from ChEMBL, filter, compute properties,
    and save to CSV.

    Args:
        target_size: Number of valid molecules to collect.
        output_file: Path for the output CSV.
        force: Overwrite existing file.
        batch_log_interval: Print progress every N accepted molecules.
    """
    if os.path.exists(output_file) and not force:
        print(f"Output file already exists: {output_file}")
        print("Use --force to overwrite.")
        return

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("ERROR: chembl_webresource_client not installed.")
        print("Install with: pip install chembl_webresource_client")
        sys.exit(1)

    molecule = new_client.molecule

    print(f"Querying ChEMBL for ~{target_size} drug-like molecules …")
    print(f"Filters: MW 200-500, LogP 0-5, HBD≤5, HBA≤10, arom≥1, rot≤10, HA≤35")

    # Use ChEMBL server-side filters to reduce bandwidth
    def _query():
        return molecule.filter(
            molecule_properties__mw_freebase__lte=500,
            molecule_properties__mw_freebase__gte=200,
            molecule_properties__alogp__lte=5,
            molecule_properties__alogp__gte=0,
            molecule_properties__hbd__lte=5,
            molecule_properties__hba__lte=10,
            molecule_properties__aromatic_rings__gte=1,
        ).only("molecule_structures")

    results = download_with_retry(_query)

    # Prepare output directory and CSV
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    headers = [
        "smiles", "molecular_weight", "logp", "hbd", "hba", "tpsa",
        "rotatable_bonds", "num_aromatic_rings", "num_rings",
        "num_heavy_atoms", "fraction_sp3", "qed", "source",
    ]

    seen_smiles: set = set()
    accepted = 0
    examined = 0

    with open(output_file, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(headers)

        pbar = tqdm(total=target_size, desc="Collecting", unit="mol")

        try:
            for rec in results:
                if accepted >= target_size:
                    break

                examined += 1
                struct = rec.get("molecule_structures")
                if not struct:
                    continue
                smi = struct.get("canonical_smiles")
                if not smi:
                    continue

                # RDKit validation
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    continue

                canon = Chem.MolToSmiles(mol, canonical=True)
                if canon in seen_smiles:
                    continue
                seen_smiles.add(canon)

                # Compute and filter properties
                props = compute_properties(mol)
                if not props:
                    continue
                if not passes_filters(props):
                    continue

                row = [
                    canon,
                    f"{props.get('molecular_weight', 0):.2f}",
                    f"{props.get('logp', 0):.2f}",
                    props.get("hbd", 0),
                    props.get("hba", 0),
                    f"{props.get('tpsa', 0):.2f}",
                    props.get("rotatable_bonds", 0),
                    props.get("num_aromatic_rings", 0),
                    props.get("num_rings", 0),
                    props.get("num_heavy_atoms", 0),
                    f"{props.get('fraction_sp3', 0):.3f}",
                    f"{props.get('qed', 0):.3f}",
                    "chembl",
                ]
                writer.writerow(row)
                accepted += 1
                pbar.update(1)

                # Flush periodically
                if accepted % batch_log_interval == 0:
                    fout.flush()

        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted. Saved {accepted} molecules so far.")
        finally:
            pbar.close()

    print(f"\nDone. Examined {examined} molecules, accepted {accepted}.")
    print(f"Saved to {output_file}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download drug-like molecules from ChEMBL"
    )
    parser.add_argument(
        "--target_size", type=int, default=50_000,
        help="Number of molecules to collect (default: 50000)",
    )
    parser.add_argument(
        "--output", type=str, default="data/raw/chembl_druglike.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output file",
    )
    args = parser.parse_args()

    download_chembl_druglike(
        target_size=args.target_size,
        output_file=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
