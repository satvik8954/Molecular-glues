# run_multi_seed.py
"""
Train and evaluate the molecular glue classifier across multiple random seeds,
then report averaged metrics (MCC, AUC, F1, Accuracy, Precision, Recall, SPCC, PCC).

Usage:
  python run_multi_seed.py                          # 10 seeds (0-9)
  python run_multi_seed.py --seeds 42 123 456       # specific seeds
  python run_multi_seed.py --num_seeds 5            # 5 seeds (0-4)
  python run_multi_seed.py --epochs 30              # override training epochs
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)
from scipy.stats import spearmanr, pearsonr
from torch_geometric.loader import DataLoader

from config_classifier import ClassifierConfig
from data.classifier_dataset import MolecularGlueClassifierDataset
from model.classifier import MolecularGlueClassifier
from train.classifier_trainer import ClassifierTrainer


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_seed(seed, config):
    """Train the classifier with a specific random seed. Returns the checkpoint path."""
    print(f"\n{'='*80}")
    print(f"  SEED {seed}")
    print(f"{'='*80}\n")

    set_seed(seed)

    # Datasets
    train_dataset = MolecularGlueClassifierDataset(
        csv_file=config.train_path,
        split_name='train',
        augment=config.use_augmentation,
        augment_prob=config.augment_prob
    )
    val_dataset = MolecularGlueClassifierDataset(
        csv_file=config.val_path,
        split_name='val',
        augment=False
    )

    # Fresh model
    model = MolecularGlueClassifier(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        drop_edge_rate=config.drop_edge_rate
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Per-seed checkpoint directory
    seed_ckpt_dir = os.path.join(config.checkpoint_dir, f'seed_{seed}')
    seed_config = ClassifierConfig(**{
        k: v for k, v in vars(config).items()
    })
    seed_config.checkpoint_dir = seed_ckpt_dir
    Path(seed_ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Train
    trainer = ClassifierTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=seed_config
    )
    trainer.train(num_epochs=config.num_epochs)

    return os.path.join(seed_ckpt_dir, 'best_classifier.pt')


def evaluate_one_seed(checkpoint_path, test_dataset):
    """Evaluate a single checkpoint. Returns a dict of metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('config', {})
    model = MolecularGlueClassifier(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    test_loader = DataLoader(
        test_dataset, batch_size=32,
        collate_fn=test_dataset.collate_fn
    )

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            ).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())

    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs > 0.5).astype(float)

    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc_val = roc_auc_score(labels, probs)
    spcc, _ = spearmanr(labels, probs)
    pcc, _ = pearsonr(labels, probs)

    return {
        'MCC': mcc,
        'AUC': auc_val,
        'F1': f1,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'SPCC': spcc,
        'PCC': pcc,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train & evaluate classifier across multiple random seeds'
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Explicit list of seeds (e.g. --seeds 0 1 2 3 4)')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='Number of seeds to run (0..N-1). Ignored if --seeds is given.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training; only evaluate existing checkpoints.')
    args = parser.parse_args()

    # Determine seeds
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(args.num_seeds))

    # Config
    config = ClassifierConfig()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.train_path is not None:
        config.train_path = args.train_path
    if args.val_path is not None:
        config.val_path = args.val_path
    if args.test_path is not None:
        config.test_path = args.test_path
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.dropout is not None:
        config.dropout = args.dropout

    print("=" * 80)
    print("  MULTI-SEED TRAINING & EVALUATION")
    print("=" * 80)
    print(f"  Seeds: {seeds}")
    print(f"  Epochs per seed: {config.num_epochs}")
    print(f"  Eval-only mode: {args.eval_only}")
    print()

    # ─── Train ───────────────────────────────────────────────────────────
    checkpoint_paths = {}
    for seed in seeds:
        ckpt_path = os.path.join(config.checkpoint_dir, f'seed_{seed}', 'best_classifier.pt')
        if args.eval_only:
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] No checkpoint for seed {seed} at {ckpt_path}")
                continue
            checkpoint_paths[seed] = ckpt_path
        else:
            checkpoint_paths[seed] = train_one_seed(seed, config)

    if not checkpoint_paths:
        print("No checkpoints found. Nothing to evaluate.")
        return

    # ─── Evaluate ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  EVALUATION ON TEST SET")
    print(f"{'='*80}\n")

    test_dataset = MolecularGlueClassifierDataset(
        csv_file=config.test_path,
        split_name='test'
    )

    all_results = {}
    metric_names = ['MCC', 'AUC', 'F1', 'Accuracy', 'Precision', 'Recall', 'SPCC', 'PCC']

    for seed, ckpt_path in sorted(checkpoint_paths.items()):
        print(f"\n--- Seed {seed} ---")
        metrics = evaluate_one_seed(ckpt_path, test_dataset)
        all_results[seed] = metrics
        for name in metric_names:
            print(f"  {name:>10s}: {metrics[name]:.4f}")

    # ─── Aggregate ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  AGGREGATED RESULTS")
    print(f"{'='*80}\n")

    header = f"{'Metric':>12s} | {'Mean':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s}"
    print(header)
    print("-" * len(header))

    for name in metric_names:
        values = [all_results[s][name] for s in all_results]
        mean = np.mean(values)
        std = np.std(values)
        lo = np.min(values)
        hi = np.max(values)
        print(f"{name:>12s} | {mean:8.4f} | {std:8.4f} | {lo:8.4f} | {hi:8.4f}")

    # ─── Per-seed table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  PER-SEED RESULTS TABLE")
    print(f"{'='*80}\n")

    # Header row
    cols = ['Seed'] + metric_names
    col_widths = [6] + [10] * len(metric_names)
    header_row = ' | '.join(f"{c:>{w}s}" for c, w in zip(cols, col_widths))
    print(header_row)
    print("-" * len(header_row))

    for seed in sorted(all_results):
        row_vals = [f"{seed:>6d}"] + [f"{all_results[seed][m]:10.4f}" for m in metric_names]
        print(' | '.join(row_vals))

    # ─── Save CSV ────────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(config.checkpoint_dir, 'multi_seed_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed'] + metric_names)
        for seed in sorted(all_results):
            writer.writerow([seed] + [f"{all_results[seed][m]:.6f}" for m in metric_names])
        # Summary rows
        writer.writerow([])
        for stat_name, stat_fn in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]:
            row = [stat_name]
            for m in metric_names:
                vals = [all_results[s][m] for s in all_results]
                row.append(f"{stat_fn(vals):.6f}")
            writer.writerow(row)

    print(f"\n✓ Results saved to {csv_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
