"""
Script for generating molecules from a trained checkpoint.
"""
import argparse
import torch
from tqdm import tqdm

import sys
import os

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from model.diffusion import MolecularDiffusion
from generate.sampler import MoleculeSampler
from generate.postprocess import (
    postprocess_molecule, filter_generated, deduplicate,
    save_molecules, save_molecules_sdf,
)
from eval.metrics import compute_all_metrics, print_metrics_report


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Molecules')

    # Model
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')

    # Generation
    parser.add_argument('--n_molecules', type=int, default=100)
    parser.add_argument('--min_atoms', type=int, default=20)
    parser.add_argument('--max_atoms', type=int, default=35)
    parser.add_argument('--min_heavy_atoms', type=int, default=10,
                        help='Reject molecules with fewer heavy atoms than this')
    parser.add_argument('--oversample', type=float, default=3.0,
                        help='Generate this many times more raw molecules to compensate for filtering')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of diffusion steps (default 100, max 500). '
                             'Fewer steps = faster generation.')

    # Guidance
    parser.add_argument('--guided', action='store_true', help='Use property guidance')
    parser.add_argument('--guidance_scale', type=float, default=2.0)

    # Filtering
    parser.add_argument('--no_filter', action='store_true', default=False,
                        help='Skip all filtering (useful for early-stage models)')
    parser.add_argument('--filter_drug_like', action='store_true', default=False)
    parser.add_argument('--filter_glue_like', action='store_true', default=False)
    parser.add_argument('--filter_pains', action='store_true', default=False)
    parser.add_argument('--no_reactive', action='store_true', default=False)
    parser.add_argument('--use_rejection_sampling', action='store_true', default=False,
                        help='Use rejection sampling with hard property bounds')

    # Output
    parser.add_argument('--output', type=str, default='generated_molecules.csv')
    parser.add_argument('--output_sdf', type=str, default=None, help='Optional SDF output')
    parser.add_argument('--training_data', type=str, default=None)
    parser.add_argument('--known_glues', type=str, default='data/glue_chemotypes.csv')

    # System
    parser.add_argument('--device', type=str, default='auto')

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.device = device
    config.generation.num_molecules = args.n_molecules
    config.generation.batch_size = args.batch_size

    # Create model and load weights
    model = MolecularDiffusion(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Device: {device}")

    # Create sampler
    sampler = MoleculeSampler(model, config)

    # ── Rejection sampling path ──────────────────────────────────────
    if args.use_rejection_sampling:
        targets = MoleculeSampler.get_glue_targets() if args.guided else None
        final, rej_stats = sampler.generate_with_rejection(
            n_molecules=args.n_molecules,
            max_attempts_per_molecule=int(args.oversample * 5),
            num_atoms=(args.min_atoms + args.max_atoms) // 2,
            temperature=args.temperature,
            target_properties=targets,
            guidance_scale=args.guidance_scale if args.guided else None,
            num_sampling_steps=args.num_steps,
        )

        # Metrics
        known_glues = None
        if args.known_glues and os.path.exists(args.known_glues):
            import pandas as pd
            known_glues = pd.read_csv(args.known_glues)['smiles'].tolist()

        training_smiles = None
        if args.training_data:
            import pandas as pd
            training_smiles = pd.read_csv(args.training_data)['smiles'].tolist()

        metrics = compute_all_metrics(final, training_smiles, known_glues)
        print_metrics_report(metrics)
        save_molecules(final, args.output)
        if args.output_sdf:
            save_molecules_sdf(final, args.output_sdf)
        print(f"\nDone! {len(final)} molecules saved to {args.output}")
        return

    # ── Standard generation path ────────────────────────────────────
    # Generate molecules (with oversampling to compensate for size filtering)
    raw_target = int(args.n_molecules * args.oversample)
    print(f"\nGenerating {raw_target} raw molecules ({args.num_steps} diffusion steps, "
          f"{args.oversample}x oversample)...")
    if args.guided:
        print(f"Using guided sampling (scale={args.guidance_scale})")
        targets = MoleculeSampler.get_glue_targets()
        raw_graphs = sampler.sample_diverse(
            num_molecules=raw_target,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            target_properties=targets,
            guidance_scale=args.guidance_scale,
            num_sampling_steps=args.num_steps,
        )
    else:
        raw_graphs = sampler.sample_diverse(
            num_molecules=raw_target,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            num_sampling_steps=args.num_steps,
        )

    # Post-process: convert graphs to SMILES with size filtering
    print("Post-processing...")
    raw_smiles = []
    too_small = 0
    invalid = 0
    for graph in tqdm(raw_graphs, desc="Converting to SMILES"):
        smiles = postprocess_molecule(graph)
        if smiles is None:
            invalid += 1
            continue
        # Check heavy atom count
        from rdkit import Chem as _Chem
        _mol = _Chem.MolFromSmiles(smiles)
        if _mol is None:
            invalid += 1
            continue
        if _mol.GetNumHeavyAtoms() < args.min_heavy_atoms:
            too_small += 1
            continue
        raw_smiles.append(smiles)

    print(f"Valid SMILES: {len(raw_smiles)}/{len(raw_graphs)} "
          f"(rejected: {invalid} invalid, {too_small} too small <{args.min_heavy_atoms} atoms)")

    # Filter
    if args.no_filter:
        filtered = raw_smiles
        print(f"Skipping filters (--no_filter)")
    else:
        filtered = filter_generated(
            raw_smiles,
            drug_like=args.filter_drug_like,
            glue_like=args.filter_glue_like,
            pains=args.filter_pains,
            no_reactive=args.no_reactive,
        )
    print(f"After filtering: {len(filtered)}")

    # Deduplicate
    training_smiles = None
    if args.training_data:
        import pandas as pd
        training_smiles = pd.read_csv(args.training_data)['smiles'].tolist()

    final = deduplicate(filtered, training_smiles)
    print(f"After deduplication: {len(final)}")

    # Metrics
    known_glues = None
    if args.known_glues and os.path.exists(args.known_glues):
        import pandas as pd
        known_glues = pd.read_csv(args.known_glues)['smiles'].tolist()

    metrics = compute_all_metrics(final, training_smiles, known_glues)
    print_metrics_report(metrics)

    # Save
    save_molecules(final, args.output)

    if args.output_sdf:
        save_molecules_sdf(final, args.output_sdf)

    print(f"\nDone! {len(final)} molecules saved to {args.output}")


if __name__ == '__main__':
    main()
