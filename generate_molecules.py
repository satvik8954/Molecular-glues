"""
Molecule generation script.

Usage:
    python generate_molecules.py --checkpoint checkpoints/best_model.pt --n_molecules 1000
    python generate_molecules.py --checkpoint checkpoints/best_model.pt --n_molecules 100 --output generated.csv
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import Config
from model.diffusion import MolecularDiffusion
from generate.sampler import MoleculeSampler
from generate.postprocess import filter_generated, deduplicate, save_molecules
from eval.metrics import print_metrics_report


def parse_args():
    parser = argparse.ArgumentParser(description='Generate molecules from trained model')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Generation
    parser.add_argument('--n_molecules', type=int, default=1000,
                        help='Number of molecules to generate')
    parser.add_argument('--min_atoms', type=int, default=10,
                        help='Minimum atoms per molecule')
    parser.add_argument('--max_atoms', type=int, default=30,
                        help='Maximum atoms per molecule')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Generation batch size')
    
    # Filtering
    parser.add_argument('--filter_drug_like', action='store_true', default=True,
                        help='Apply drug-likeness filter')
    parser.add_argument('--filter_glue_like', action='store_true',
                        help='Apply glue-like filter')
    parser.add_argument('--no_pains', action='store_true',
                        help='Skip PAINS filter')
    
    # Output
    parser.add_argument('--output', type=str, default='generated_molecules.csv',
                        help='Output CSV file')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, or auto)')
    
    # Training data for novelty check
    parser.add_argument('--training_data', type=str, default=None,
                        help='Path to training data for novelty check')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
        config.device = device
    else:
        config = Config()
        config.device = device
    
    # Create and load model
    print("Creating model...")
    model = MolecularDiffusion(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create sampler
    sampler = MoleculeSampler(model, device)
    
    # Generate molecules
    print(f"\nGenerating {args.n_molecules} molecules...")
    molecules = sampler.sample_diverse(
        num_molecules=args.n_molecules,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        temperatures=[0.7, 1.0, 1.3],
        show_progress=True,
    )
    
    print(f"\nGenerated {len(molecules)} molecular graphs")
    
    # Filter molecules
    print("\nFiltering molecules...")
    valid_smiles, filter_stats = filter_generated(
        molecules,
        apply_drug_like=args.filter_drug_like,
        apply_glue_like=args.filter_glue_like,
        apply_pains=not args.no_pains,
        apply_sa_filter=True,
        verbose=True,
    )
    
    # Load training data for novelty check if provided
    training_smiles = None
    if args.training_data:
        print(f"\nLoading training data for novelty check: {args.training_data}")
        import pandas as pd
        df = pd.read_csv(args.training_data)
        training_smiles = set(df['smiles'].dropna().tolist())
        print(f"Training set size: {len(training_smiles)}")
    
    # Deduplicate
    print("\nDeduplicating...")
    unique_smiles, dedup_stats = deduplicate(
        valid_smiles,
        training_smiles=training_smiles,
        verbose=True,
    )
    
    # Print metrics
    print_metrics_report(unique_smiles, training_smiles)
    
    # Save results
    print(f"\nSaving to: {args.output}")
    save_molecules(unique_smiles, args.output, include_properties=True)
    
    # Print sample molecules
    print("\nSample generated molecules:")
    for i, smiles in enumerate(unique_smiles[:10]):
        print(f"  {i+1}: {smiles}")
    
    print(f"\nDone! Generated {len(unique_smiles)} unique, valid molecules.")


if __name__ == '__main__':
    main()
