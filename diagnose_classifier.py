"""
Comprehensive diagnostic for molecular glue classifier.

Tests whether the model might be learning shortcuts rather than real chemistry.
Generates multiple diagnostic plots and prints feature baseline comparisons.

Usage:
    python scripts/diagnose_classifier.py
    python scripts/diagnose_classifier.py --checkpoint checkpoints/classifier/best_classifier.pt

Output: diagnostic plots in diagnostics/ folder
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs, rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import argparse


# ─── Feature extraction ─────────────────────────────────────────────

FEATURE_NAMES = [
    'MW', 'LogP', 'Rings', 'AromRings', 'HeavyAtoms', 'HBD', 'HBA',
    'TPSA', 'FracCSP3', 'RotBonds', 'NumAtoms', 'NumBonds'
]


def extract_features(smiles):
    """Extract simple molecular descriptors from SMILES."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.RingCount(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumRotatableBonds(mol),
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
        ]
    except Exception:
        return None


def extract_features_df(df):
    """Extract features for an entire DataFrame. Returns X, y, valid_indices."""
    X, y, valid_idx = [], [], []
    for idx, row in df.iterrows():
        feats = extract_features(row['SMILES'])
        if feats is not None:
            X.append(feats)
            y.append(row['label'])
            valid_idx.append(idx)
    return np.array(X), np.array(y), valid_idx


# ─── Diagnostic 1: Simple Feature Baselines ─────────────────────────

def test_simple_baselines(train_df, test_df, out_dir):
    """Train simple classifiers on molecular descriptors to check for shortcuts."""
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 1: Simple Feature Baselines")
    print("=" * 60)
    print("  If simple features achieve high AUC, the GNN may be")
    print("  learning the same shortcuts rather than real chemistry.\n")

    print("  Extracting features...")
    X_train, y_train, _ = extract_features_df(train_df)
    X_test, y_test, _ = extract_features_df(test_df)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    results = {}

    # Individual feature baselines
    print("\n  --- Single-Feature Baselines ---")
    for i, name in enumerate(FEATURE_NAMES):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train[:, i:i+1], y_train)
        probs = lr.predict_proba(X_test[:, i:i+1])[:, 1]
        auc_val = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, lr.predict(X_test[:, i:i+1]))
        flag = " ⚠ SHORTCUT!" if auc_val > 0.65 else ""
        print(f"    {name:>12s}: AUC={auc_val:.4f}  Acc={acc:.4f}{flag}")
        results[name] = {'auc': auc_val, 'acc': acc}

    # Multi-feature baselines
    print("\n  --- Multi-Feature Baselines ---")
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        auc_val = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        flag = " ⚠ SHORTCUT!" if auc_val > 0.75 else ""
        print(f"    {model_name:>20s}: AUC={auc_val:.4f}  Acc={acc:.4f}  F1={f1:.4f}{flag}")
        results[model_name] = {'auc': auc_val, 'acc': acc, 'f1': f1}

    # Feature importances from RF
    rf = models['RandomForest']
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n  --- Random Forest Feature Importances ---")
    for i in sorted_idx:
        bar = '█' * int(importances[i] * 50)
        print(f"    {FEATURE_NAMES[i]:>12s}: {importances[i]:.4f}  {bar}")

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(FEATURE_NAMES)), importances[sorted_idx], color='steelblue')
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importances\n(Higher = model relies more on this feature)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importances.png'), dpi=150)
    plt.close()

    return results


# ─── Diagnostic 2: Feature Distribution Overlap ─────────────────────

def test_feature_distributions(train_df, out_dir):
    """Plot feature distributions for glues vs non-glues."""
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 2: Feature Distribution Overlap")
    print("=" * 60)

    X, y, _ = extract_features_df(train_df)
    glue_mask = y == 1
    nonglue_mask = y == 0

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        glue_vals = X[glue_mask, i]
        nonglue_vals = X[nonglue_mask, i]

        ax.hist(glue_vals, bins=50, alpha=0.5, label='Glue', color='coral', density=True)
        ax.hist(nonglue_vals, bins=50, alpha=0.5, label='Non-glue', color='steelblue', density=True)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

        # Compute overlap metric (KL proxy)
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(glue_vals, nonglue_vals)
        sep_flag = " ⚠" if stat > 0.3 else ""
        ax.set_xlabel(f'KS={stat:.3f}{sep_flag}', fontsize=9)

    plt.suptitle('Feature Distributions: Glues vs Non-Glues\n(Low overlap = easy shortcut)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_distributions.png'), dpi=150)
    plt.close()
    print("  ✓ Saved feature_distributions.png")


# ─── Diagnostic 3: Cross-Split Tanimoto Leakage ─────────────────────

def test_cross_split_leakage(train_df, test_df, out_dir, n_sample=500):
    """Check if test molecules are structurally too similar to training molecules."""
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 3: Cross-Split Tanimoto Leakage")
    print("=" * 60)
    print("  High similarity between test and train molecules suggests")
    print("  the scaffold split isn't providing true generalization.\n")

    # Sample for speed
    train_sample = train_df.sample(n=min(n_sample, len(train_df)), random_state=42)
    test_sample = test_df.sample(n=min(n_sample, len(test_df)), random_state=42)

    # Compute fingerprints
    def get_fps(smiles_list):
        fps, valid_smi = [], []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol:
                    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                    fp = fp_gen.GetFingerprint(mol)
                    fps.append(fp)
                    valid_smi.append(smi)
            except:
                pass
        return fps, valid_smi

    print("  Computing fingerprints...")
    train_fps, _ = get_fps(train_sample['SMILES'].tolist())
    test_fps, test_smi = get_fps(test_sample['SMILES'].tolist())
    test_labels = test_sample['label'].values[:len(test_fps)]

    # For each test molecule, find max similarity to any train molecule
    max_sims = []
    for i, test_fp in enumerate(tqdm(test_fps, desc="  Leakage check")):
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_sims.append(max(sims))
    max_sims = np.array(max_sims)

    print(f"\n  Max test→train Tanimoto similarity:")
    print(f"    Mean:   {max_sims.mean():.4f}")
    print(f"    Median: {np.median(max_sims):.4f}")
    print(f"    Max:    {max_sims.max():.4f}")
    print(f"    >0.8:   {(max_sims > 0.8).sum()} ({100*(max_sims > 0.8).mean():.1f}%)")
    print(f"    >0.9:   {(max_sims > 0.9).sum()} ({100*(max_sims > 0.9).mean():.1f}%)")
    print(f"    =1.0:   {(max_sims >= 0.999).sum()} ({100*(max_sims >= 0.999).mean():.1f}%)")

    if (max_sims >= 0.999).mean() > 0.01:
        print("  ⚠ WARNING: >1% of test molecules are identical to train molecules!")
    if (max_sims > 0.8).mean() > 0.1:
        print("  ⚠ WARNING: >10% of test molecules have Tc>0.8 to train — near-duplicates!")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(max_sims, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(0.8, color='red', linestyle='--', label='Tc=0.8 (near-duplicate)')
    ax1.set_xlabel('Max Tanimoto Similarity to Train Set')
    ax1.set_ylabel('Count')
    ax1.set_title('Test→Train Molecular Similarity\n(Many molecules near 1.0 = data leakage)')
    ax1.legend()

    # Separate by class
    glue_sims = max_sims[test_labels[:len(max_sims)] == 1]
    nonglue_sims = max_sims[test_labels[:len(max_sims)] == 0]
    ax2.hist(glue_sims, bins=30, alpha=0.5, label=f'Glue (mean={glue_sims.mean():.3f})', color='coral')
    ax2.hist(nonglue_sims, bins=30, alpha=0.5, label=f'Non-glue (mean={nonglue_sims.mean():.3f})', color='steelblue')
    ax2.set_xlabel('Max Tanimoto Similarity to Train Set')
    ax2.set_ylabel('Count')
    ax2.set_title('Leakage by Class')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cross_split_leakage.png'), dpi=150)
    plt.close()
    print("  ✓ Saved cross_split_leakage.png")


# ─── Diagnostic 4: SMILES Length & Atom Count Shortcuts ──────────────

def test_trivial_shortcuts(train_df, test_df, out_dir):
    """Check if SMILES length or atom counts separate classes."""
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 4: Trivial SMILES Shortcuts")
    print("=" * 60)

    for name, df in [('Train', train_df), ('Test', test_df)]:
        df = df.copy()
        df['smi_len'] = df['SMILES'].str.len()
        glue_len = df[df['label'] == 1]['smi_len']
        non_len = df[df['label'] == 0]['smi_len']

        # Logistic regression on SMILES length alone
        X = df['smi_len'].values.reshape(-1, 1)
        y = df['label'].values
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        acc = accuracy_score(y, lr.predict(X))
        auc_val = roc_auc_score(y, lr.predict_proba(X)[:, 1])

        print(f"\n  {name} set:")
        print(f"    Glue SMILES length:     mean={glue_len.mean():.0f} ± {glue_len.std():.0f}")
        print(f"    Non-glue SMILES length: mean={non_len.mean():.0f} ± {non_len.std():.0f}")
        print(f"    SMILES-length-only LR:  Acc={acc:.4f}  AUC={auc_val:.4f}")
        if auc_val > 0.6:
            print(f"    ⚠ SMILES length can partially separate classes!")


# ─── Diagnostic 5: Precision-Recall & Calibration (if model available) ─

def test_model_curves(checkpoint_path, test_df, out_dir):
    """Generate PR curve, calibration curve, and confidence distribution."""
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 5: Model Curves (PR, Calibration, Confidence)")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"  [!] No checkpoint found at {checkpoint_path}, skipping model diagnostics")
        return

    import torch
    from data.classifier_dataset import MolecularGlueClassifierDataset
    from model.classifier import MolecularGlueClassifier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model = MolecularGlueClassifier(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    test_dataset = MolecularGlueClassifierDataset(
        csv_file='data/classifier_test.csv', split_name='test'
    )
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=test_dataset.collate_fn)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- 1. ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'GNN (AUC={roc_auc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[0, 0].fill_between(fpr, tpr, alpha=0.1, color='blue')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()

    # --- 2. Precision-Recall Curve ---
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    axes[0, 1].plot(recall, precision, 'r-', linewidth=2, label=f'GNN (AP={ap:.4f})')
    # Random baseline for balanced data
    prevalence = all_labels.mean()
    axes[0, 1].axhline(y=prevalence, color='k', linestyle='--', alpha=0.5, label=f'Random ({prevalence:.2f})')
    axes[0, 1].fill_between(recall, precision, alpha=0.1, color='red')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve\n(More informative than ROC for balanced data)')
    axes[0, 1].legend()

    # --- 3. Calibration Curve ---
    fraction_of_positives, mean_predicted_value = calibration_curve(
        all_labels, all_probs, n_bins=10
    )
    axes[1, 0].plot(mean_predicted_value, fraction_of_positives, 's-', color='green',
                    linewidth=2, label='GNN')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly calibrated')
    axes[1, 0].set_xlabel('Mean Predicted Probability')
    axes[1, 0].set_ylabel('Fraction of Positives')
    axes[1, 0].set_title('Calibration Curve\n(Deviations show over/under-confidence)')
    axes[1, 0].legend()

    # --- 4. Confidence Distribution ---
    glue_probs = all_probs[all_labels == 1]
    nonglue_probs = all_probs[all_labels == 0]
    axes[1, 1].hist(glue_probs, bins=50, alpha=0.6, label=f'Glue (n={len(glue_probs)})',
                    color='coral', density=True)
    axes[1, 1].hist(nonglue_probs, bins=50, alpha=0.6, label=f'Non-glue (n={len(nonglue_probs)})',
                    color='steelblue', density=True)
    axes[1, 1].axvline(0.5, color='k', linestyle='--', alpha=0.5, label='Decision boundary')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Confidence Distribution\n(Peaks at 0 and 1 = overconfident model)')
    axes[1, 1].legend()

    plt.suptitle('Model Diagnostic Curves', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved model_curves.png")

    # Print summary stats
    print(f"\n  Model Performance:")
    print(f"    ROC AUC:          {roc_auc:.4f}")
    print(f"    Avg Precision:    {ap:.4f}")
    print(f"    Mean P(glue):     {glue_probs.mean():.4f} (should be ~1.0)")
    print(f"    Mean P(non-glue): {nonglue_probs.mean():.4f} (should be ~0.0)")
    print(f"    Uncertain (0.3-0.7): {((all_probs > 0.3) & (all_probs < 0.7)).sum()} samples")

    if glue_probs.mean() > 0.95 and nonglue_probs.mean() < 0.05:
        print("  ⚠ Model is EXTREMELY confident — likely learning shortcuts!")


# ─── Diagnostic 6: Per-Scaffold Performance ─────────────────────────

def test_scaffold_performance(test_df, out_dir):
    """Check if certain scaffolds are always correct (memorization indicator)."""
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC 6: Scaffold Diversity Check")
    print("=" * 60)

    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    for smi in test_df['SMILES']:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.append(Chem.MolToSmiles(scaffold))
            else:
                scaffolds.append('_invalid')
        except:
            scaffolds.append('_error')

    test_df = test_df.copy()
    test_df['scaffold'] = scaffolds

    scaffold_counts = Counter(scaffolds)
    n_unique = len(scaffold_counts)
    n_singleton = sum(1 for c in scaffold_counts.values() if c == 1)

    print(f"  Unique scaffolds in test: {n_unique}")
    print(f"  Singletons (1 molecule):  {n_singleton} ({100*n_singleton/n_unique:.1f}%)")
    print(f"  Top 10 scaffolds:")
    for scaffold, count in scaffold_counts.most_common(10):
        subset = test_df[test_df['scaffold'] == scaffold]
        label_dist = subset['label'].value_counts().to_dict()
        print(f"    {count:>4d} molecules  labels={label_dist}  {scaffold[:60]}...")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Diagnose classifier for shortcuts')
    parser.add_argument('--train_path', default='data/classifier_train.csv')
    parser.add_argument('--test_path', default='data/classifier_test.csv')
    parser.add_argument('--checkpoint', default='checkpoints/classifier/best_classifier.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--out_dir', default='diagnostics',
                        help='Output directory for diagnostic plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  MOLECULAR GLUE CLASSIFIER - COMPREHENSIVE DIAGNOSTICS")
    print("=" * 60)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    print(f"\n  Train: {len(train_df)} molecules")
    print(f"  Test:  {len(test_df)} molecules")

    # Run all diagnostics
    test_simple_baselines(train_df, test_df, args.out_dir)
    test_feature_distributions(train_df, args.out_dir)
    test_cross_split_leakage(train_df, test_df, args.out_dir)
    test_trivial_shortcuts(train_df, test_df, args.out_dir)
    test_scaffold_performance(test_df, args.out_dir)

    # Model diagnostics (only if checkpoint exists)
    test_model_curves(args.checkpoint, test_df, args.out_dir)

    print("\n" + "=" * 60)
    print("  DIAGNOSTICS COMPLETE")
    print(f"  Plots saved to: {args.out_dir}/")
    print("=" * 60)
    print(f"\n  Key files:")
    print(f"    {args.out_dir}/feature_importances.png  — Which simple features drive predictions")
    print(f"    {args.out_dir}/feature_distributions.png — Glue vs non-glue feature overlap")
    print(f"    {args.out_dir}/cross_split_leakage.png  — Test-train molecular similarity")
    print(f"    {args.out_dir}/model_curves.png         — PR, calibration, confidence curves")


if __name__ == '__main__':
    main()
