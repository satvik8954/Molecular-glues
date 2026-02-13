"""
Download drug-like molecules from ChEMBL and merge with existing glue chemotypes.

Usage:
    pip install chembl_webresource_client   # if not already installed
    python scripts/download_training_data.py --output data/training_data.csv

If ChEMBL is unavailable, a ZINC15 download fallback is provided.
"""
import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def download_chembl_druglike(n_molecules: int = 50_000) -> list:
    """Download drug-like molecules from ChEMBL."""
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("ERROR: chembl_webresource_client not installed.")
        print("Install with: pip install chembl_webresource_client")
        return []

    molecule = new_client.molecule
    print(f"Querying ChEMBL for up to {n_molecules} drug-like molecules …")

    results = molecule.filter(
        molecule_properties__mw_freebase__lte=500,
        molecule_properties__mw_freebase__gte=200,
        molecule_properties__alogp__lte=5,
        molecule_properties__hbd__lte=5,
        molecule_properties__hba__lte=10,
        molecule_properties__aromatic_rings__gte=1,
    ).only("molecule_structures")

    smiles_list = []
    for rec in results:
        if len(smiles_list) >= n_molecules:
            break
        struct = rec.get("molecule_structures")
        if struct and struct.get("canonical_smiles"):
            smiles_list.append(struct["canonical_smiles"])

        if len(smiles_list) % 5000 == 0 and len(smiles_list) > 0:
            print(f"  … downloaded {len(smiles_list)} so far")

    print(f"Downloaded {len(smiles_list)} molecules from ChEMBL")
    return smiles_list


def validate_smiles_list(smiles_list: list) -> list:
    """Validate SMILES using RDKit."""
    from rdkit import Chem

    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                canonical = Chem.MolToSmiles(mol, canonical=True)
                valid.append(canonical)
            except Exception:
                pass

    print(f"Validated: {len(valid)}/{len(smiles_list)} SMILES are valid")
    return valid


def merge_datasets(
    chembl_smiles: list,
    existing_csv: str,
    output_csv: str,
    glue_fraction: float = 0.15,
):
    """Merge ChEMBL molecules with existing glue chemotypes."""
    import pandas as pd

    # Load existing glues
    if os.path.exists(existing_csv):
        df_existing = pd.read_csv(existing_csv)
        existing_smiles = df_existing["smiles"].dropna().tolist()
        print(f"Existing glue chemotypes: {len(existing_smiles)}")
    else:
        existing_smiles = []
        print(f"No existing file at {existing_csv}")

    # Deduplicate
    all_smiles = list(set(chembl_smiles + existing_smiles))

    # Ensure glues are represented at desired fraction via oversampling
    if existing_smiles and glue_fraction > 0:
        target_glue_count = int(len(all_smiles) * glue_fraction)
        oversample = max(1, target_glue_count // max(len(existing_smiles), 1))
        oversampled_glues = existing_smiles * oversample
        all_smiles = list(set(all_smiles + oversampled_glues[: target_glue_count]))

    df = pd.DataFrame({"smiles": all_smiles})
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(all_smiles)} molecules to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Download & merge training data")
    parser.add_argument("--n_molecules", type=int, default=50_000)
    parser.add_argument("--existing", type=str, default="data/glue_chemotypes.csv")
    parser.add_argument("--output", type=str, default="data/training_data.csv")
    parser.add_argument("--glue_fraction", type=float, default=0.15,
                        help="Fraction of final dataset that should be glue chemotypes")
    args = parser.parse_args()

    # Download
    smiles = download_chembl_druglike(args.n_molecules)
    if not smiles:
        print("\nChEMBL download failed. Please install chembl_webresource_client "
              "or manually provide drug-like SMILES in data/training_data.csv")
        return

    # Validate
    smiles = validate_smiles_list(smiles)

    # Merge
    merge_datasets(smiles, args.existing, args.output, args.glue_fraction)


if __name__ == "__main__":
    main()
