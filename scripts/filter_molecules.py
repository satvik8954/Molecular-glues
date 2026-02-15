"""
Post-generation multi-stage molecule filtering and ranking.

Pipeline:
    1. Chemical validity (RDKit sanitisation)
    2. Drug-likeness (Lipinski Ro5, 1 violation allowed)
    3. Glue-likeness filters (MW, LogP, HBD, HBA, TPSA, aromatic rings)
    4. PAINS / reactive-group rejection
    5. Synthetic accessibility (SA score < 4.5)
    6. Deduplication (canonical SMILES)
    7. Ranking by glue-likeness score

Usage:
    python scripts/filter_molecules.py \\
        --input outputs/generated_molecules.csv \\
        --output outputs/filtered_candidates.csv \\
        --top_n 100 \\
        --min_glue_score 50
"""
import argparse
import csv
import os
import sys

import pandas as pd
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem

from utils.chemistry import (
    is_valid_molecule,
    canonicalize_smiles,
    get_molecular_properties,
)
from utils.filters import (
    is_drug_like,
    is_glue_like,
    passes_pains_filter,
    has_reactive_groups,
    is_synthetically_accessible,
)
from utils.glue_scoring import compute_glue_likeness_score


def filter_pipeline(
    smiles_list: list,
    min_glue_score: int = 0,
    top_n: int = 100,
    verbose: bool = True,
) -> tuple:
    """
    Run the full multi-stage filtering pipeline.

    Args:
        smiles_list: Raw SMILES from generation.
        min_glue_score: Minimum glue-likeness score (0-100).
        top_n: Number of top candidates to keep.
        verbose: Print per-stage statistics.

    Returns:
        (filtered_df, stats_dict)
    """
    stats = {"input": len(smiles_list)}

    # Stage 1: Chemical validity
    valid = []
    for smi in smiles_list:
        if is_valid_molecule(smi):
            valid.append(smi)
    stats["after_validity"] = len(valid)
    if verbose:
        print(f"  Stage 1 – Validity:      {len(valid)}/{stats['input']} pass")

    # Stage 2: Drug-likeness (Lipinski, 1 violation allowed)
    druglike = []
    for smi in valid:
        if is_drug_like(smi):
            druglike.append(smi)
    stats["after_druglike"] = len(druglike)
    if verbose:
        print(f"  Stage 2 – Drug-likeness: {len(druglike)}/{len(valid)} pass")

    # Stage 3: Glue-likeness
    gluelike = []
    for smi in druglike:
        if is_glue_like(smi):
            gluelike.append(smi)
    stats["after_gluelike"] = len(gluelike)
    if verbose:
        print(f"  Stage 3 – Glue-likeness: {len(gluelike)}/{len(druglike)} pass")

    # Stage 4: PAINS + reactive groups
    clean = []
    for smi in gluelike:
        if passes_pains_filter(smi) and not has_reactive_groups(smi):
            clean.append(smi)
    stats["after_pains_reactive"] = len(clean)
    if verbose:
        print(f"  Stage 4 – PAINS/React:   {len(clean)}/{len(gluelike)} pass")

    # Stage 5: Synthetic accessibility
    sa_pass = []
    for smi in clean:
        if is_synthetically_accessible(smi, max_score=4.5):
            sa_pass.append(smi)
    stats["after_sa"] = len(sa_pass)
    if verbose:
        print(f"  Stage 5 – SA score:      {len(sa_pass)}/{len(clean)} pass")

    # Stage 6: Deduplication
    seen = set()
    unique = []
    for smi in sa_pass:
        canon = canonicalize_smiles(smi)
        if canon and canon not in seen:
            seen.add(canon)
            unique.append(canon)
    stats["after_dedup"] = len(unique)
    if verbose:
        print(f"  Stage 6 – Dedup:         {len(unique)}/{len(sa_pass)} unique")

    # Stage 7: Score and rank
    scored = []
    for smi in tqdm(unique, desc="Scoring", disable=not verbose):
        score = compute_glue_likeness_score(smi)
        if score >= min_glue_score:
            props = get_molecular_properties(smi) or {}
            scored.append({
                "smiles": smi,
                "glue_score": score,
                **{k: props.get(k, "") for k in [
                    "molecular_weight", "logp", "hbd", "hba", "tpsa",
                    "rotatable_bonds", "num_aromatic_rings", "num_rings",
                    "num_heavy_atoms", "fraction_sp3", "qed",
                ]},
            })

    scored.sort(key=lambda x: x["glue_score"], reverse=True)
    scored = scored[:top_n]
    stats["after_score_filter"] = len(scored)
    stats["final_top_n"] = len(scored)
    if verbose:
        print(f"  Stage 7 – Score ≥{min_glue_score}:   {len(scored)} (top {top_n})")

    df = pd.DataFrame(scored)
    return df, stats


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage filtering and ranking of generated molecules"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input CSV of generated molecules",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/filtered_candidates.csv",
        help="Output CSV for filtered candidates",
    )
    parser.add_argument(
        "--smiles_column", type=str, default="smiles",
        help="Column name for SMILES",
    )
    parser.add_argument(
        "--top_n", type=int, default=100,
        help="Number of top candidates to keep (default: 100)",
    )
    parser.add_argument(
        "--min_glue_score", type=int, default=0,
        help="Minimum glue-likeness score (0-100, default: 0)",
    )
    args = parser.parse_args()

    # Load
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    df_in = pd.read_csv(args.input)
    col = args.smiles_column
    if col not in df_in.columns:
        for alt in ("SMILES", "Smiles", "canonical_smiles"):
            if alt in df_in.columns:
                col = alt
                break
        else:
            print(f"ERROR: Column '{args.smiles_column}' not found.")
            sys.exit(1)

    smiles_list = df_in[col].dropna().tolist()
    print(f"Loaded {len(smiles_list)} molecules from {args.input}")
    print(f"\nRunning filtering pipeline …")

    df_out, stats = filter_pipeline(
        smiles_list,
        min_glue_score=args.min_glue_score,
        top_n=args.top_n,
    )

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\nSaved {len(df_out)} candidates to {args.output}")

    # Report
    print(f"\n{'='*50}")
    print(f"FILTERING REPORT")
    print(f"{'='*50}")
    for key, val in stats.items():
        print(f"  {key:25s}: {val}")

    if len(df_out) > 0:
        print(f"\nTop 5 candidates:")
        for _, row in df_out.head(5).iterrows():
            print(f"  Score={row['glue_score']:3d}  "
                  f"MW={row.get('molecular_weight', '?'):>7s}  "
                  f"LogP={row.get('logp', '?'):>5s}  "
                  f"{row['smiles'][:60]}")


if __name__ == "__main__":
    main()
