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
    parser.add_argument('--n_molecules', type=int, default=1000)
    parser.add_argument('--min_atoms', type=int, default=10)
    parser.add_argument('--max_atoms', type=int, default=30)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=50)

    # Guidance
    parser.add_argument('--guided', action='store_true', help='Use property guidance')
    parser.add_argument('--guidance_scale', type=float, default=2.0)

    # Filtering
    parser.add_argument('--filter_drug_like', action='store_true', default=True)
    parser.add_argument('--filter_glue_like', action='store_true', default=True)
    parser.add_argument('--filter_pains', action='store_true', default=True)
    parser.add_argument('--no_reactive', action='store_true', default=True)

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
    checkpoint = torch.load(args.checkpoint, map_location=device)
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

    # Generate molecules
    print(f"\nGenerating {args.n_molecules} molecules...")
    if args.guided:
        print(f"Using guided sampling (scale={args.guidance_scale})")
        targets = MoleculeSampler.get_glue_targets()
        raw_graphs = sampler.sample_diverse(
            num_molecules=args.n_molecules,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            target_properties=targets,
            guidance_scale=args.guidance_scale,
        )
    else:
        raw_graphs = sampler.sample_diverse(
            num_molecules=args.n_molecules,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
        )

    # Post-process: convert graphs to SMILES
    print("Post-processing...")
    raw_smiles = []
    for graph in tqdm(raw_graphs, desc="Converting to SMILES"):
        smiles = postprocess_molecule(graph)
        if smiles:
            raw_smiles.append(smiles)

    print(f"Valid SMILES: {len(raw_smiles)}/{len(raw_graphs)}")

    # Filter
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
