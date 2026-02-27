"""
Create balanced classifier dataset with scaffold-based train/val/test split.

KEY: Uses MW-stratified sampling to select negatives that match the
molecular weight distribution of glues. This prevents the model from
learning trivial size-based shortcuts instead of actual chemistry.

Scaffold split ensures molecules with the same Murcko scaffold (core ring system)
stay in the same split — prevents data leakage from structurally similar molecules.

Supports ZINC15 (.smi) and/or ChEMBL (.csv) as negative sources.

Output: data/classifier_train.csv, data/classifier_val.csv, data/classifier_test.csv
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import glob


def get_scaffold(smiles):
    """Extract Murcko scaffold from SMILES. Returns scaffold SMILES or None."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def load_smi_file(path):
    """Load SMILES from .smi file (tab/space separated: SMILES ID)."""
    smiles_list = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                smiles_list.append(parts[0])
    return pd.DataFrame({'SMILES': smiles_list})


def load_negatives(path):
    """Load negative molecules from .smi or .csv file."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.smi':
        df = load_smi_file(path)
        print(f'  Loaded {len(df)} molecules from .smi file: {path}')
    elif ext == '.csv':
        df = pd.read_csv(path)
        # Normalize column name to 'SMILES'
        if 'smiles' in df.columns and 'SMILES' not in df.columns:
            df = df.rename(columns={'smiles': 'SMILES'})
        print(f'  Loaded {len(df)} molecules from .csv file: {path}')
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .smi or .csv")
    
    return df[['SMILES']].drop_duplicates().reset_index(drop=True)


def compute_mw(smiles):
    """Compute molecular weight for a SMILES string. Returns None on failure."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return Descriptors.MolWt(mol)
    except Exception:
        pass
    return None


def mw_stratified_sample(neg_df, glue_mws, n_target, seed=42, n_bins=20):
    """
    Sample negatives to match the MW distribution of glues.
    
    Strategy:
    1. Bin glue MW distribution into n_bins histogram bins
    2. Compute how many negatives to sample from each bin
    3. Sample from each bin (with replacement if needed)
    
    This ensures negatives span the same MW range as glues, preventing
    the model from using MW as a trivial shortcut.
    
    Args:
        neg_df: DataFrame with 'SMILES' and 'MW' columns
        glue_mws: array of glue molecular weights
        n_target: target number of negatives to select
        seed: random seed
        n_bins: number of MW histogram bins
    
    Returns:
        DataFrame of selected negatives
    """
    rng = np.random.RandomState(seed)
    
    # Create MW bins from glue distribution (5th to 95th percentile + margins)
    mw_min = np.percentile(glue_mws, 2)
    mw_max = np.percentile(glue_mws, 98)
    bin_edges = np.linspace(mw_min, mw_max, n_bins + 1)
    
    # Compute glue histogram (target distribution)
    glue_hist, _ = np.histogram(glue_mws, bins=bin_edges)
    glue_frac = glue_hist / glue_hist.sum()
    
    # Assign negatives to bins
    neg_mws = neg_df['MW'].values
    neg_bins = np.digitize(neg_mws, bin_edges) - 1  # 0-indexed
    neg_bins = np.clip(neg_bins, 0, n_bins - 1)
    
    # Sample from each bin proportionally
    selected_indices = []
    for bin_idx in range(n_bins):
        target_count = max(1, int(n_target * glue_frac[bin_idx]))
        bin_mask = neg_bins == bin_idx
        bin_indices = np.where(bin_mask)[0]
        
        if len(bin_indices) == 0:
            continue
        
        if len(bin_indices) >= target_count:
            # Sample without replacement
            chosen = rng.choice(bin_indices, size=target_count, replace=False)
        else:
            # Sample with replacement if not enough negatives in this bin
            chosen = rng.choice(bin_indices, size=target_count, replace=True)
        
        selected_indices.extend(chosen)
    
    # Trim to exact target
    if len(selected_indices) > n_target:
        selected_indices = rng.choice(selected_indices, size=n_target, replace=False)
    
    result = neg_df.iloc[selected_indices].copy()
    result = result.drop_duplicates(subset='SMILES').reset_index(drop=True)
    
    return result


def scaffold_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Stratified scaffold split — preserves class balance across splits.
    
    All molecules sharing a scaffold go to the SAME split.
    Glue and non-glue scaffolds are distributed independently so that
    each split (train/val/test) has approximately 50/50 class balance.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    print("Computing Murcko scaffolds...")
    scaffold_to_indices = defaultdict(list)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scaffolds"):
        scaffold = get_scaffold(row['SMILES'])
        if scaffold is None:
            scaffold = f"_singleton_{idx}"
        scaffold_to_indices[scaffold].append(idx)
    
    print(f"  Found {len(scaffold_to_indices)} unique scaffolds for {len(df)} molecules")
    
    # --- Separate scaffolds by majority label (glue vs non-glue) ---
    glue_scaffolds = []      # scaffolds whose molecules are mostly glue (label=1)
    nonglue_scaffolds = []   # scaffolds whose molecules are mostly non-glue (label=0)
    
    for scaffold, indices in scaffold_to_indices.items():
        labels = df.loc[indices, 'label'].values
        majority_label = int(np.round(labels.mean()))  # 1 if majority glue, else 0
        if majority_label == 1:
            glue_scaffolds.append((scaffold, indices))
        else:
            nonglue_scaffolds.append((scaffold, indices))
    
    print(f"  Glue scaffolds: {len(glue_scaffolds)}, Non-glue scaffolds: {len(nonglue_scaffolds)}")
    
    # --- Helper: assign scaffolds of one class to train/val/test ---
    def _assign_scaffolds(scaffold_list_cls, rng):
        rng.shuffle(scaffold_list_cls)
        # Sort by size (largest first) for better ratio approximation
        scaffold_list_cls.sort(key=lambda x: len(x[1]), reverse=True)
        
        n_cls = sum(len(idxs) for _, idxs in scaffold_list_cls)
        train_cutoff = int(n_cls * train_ratio)
        val_cutoff = int(n_cls * (train_ratio + val_ratio))
        
        train_idx, val_idx, test_idx = [], [], []
        for _, indices in scaffold_list_cls:
            if len(train_idx) < train_cutoff:
                train_idx.extend(indices)
            elif len(train_idx) + len(val_idx) < val_cutoff:
                val_idx.extend(indices)
            else:
                test_idx.extend(indices)
        return train_idx, val_idx, test_idx
    
    rng = np.random.RandomState(seed)
    
    g_train, g_val, g_test = _assign_scaffolds(glue_scaffolds, rng)
    n_train, n_val, n_test = _assign_scaffolds(nonglue_scaffolds, rng)
    
    train_indices = g_train + n_train
    val_indices   = g_val   + n_val
    test_indices  = g_test  + n_test
    
    # Shuffle within each split so classes are interleaved
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    
    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[val_indices].reset_index(drop=True),
        df.loc[test_indices].reset_index(drop=True),
    )


def print_stats(df, name):
    """Print MW/ring stats for a set of molecules."""
    mols = [Chem.MolFromSmiles(s) for s in df['SMILES'][:1000] if Chem.MolFromSmiles(str(s))]
    mws = [Descriptors.MolWt(m) for m in mols]
    rings = [Descriptors.RingCount(m) for m in mols]
    print(f'  {name} (n={len(mols)}): MW={np.mean(mws):.0f}±{np.std(mws):.0f} '
          f'[{np.min(mws):.0f}-{np.max(mws):.0f}], Rings={np.mean(rings):.1f}')


def main():
    parser = argparse.ArgumentParser(description='Create classifier dataset with scaffold split')
    parser.add_argument('--glue_path', default='data/glue_chemotypes.smi',
                        help='Path to glue molecules (.smi)')
    parser.add_argument('--neg_paths', nargs='+', 
                        default=['data/zinc15_druglike.smi', 'data/chembl_druglike.csv'],
                        help='Paths to negative molecule files (.smi or .csv). '
                             'Multiple files are pooled together for better MW coverage.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_bins', type=int, default=20,
                        help='Number of MW bins for stratified sampling (default: 20)')
    args = parser.parse_args()

    # ---- Load glues ----
    glues = load_smi_file(args.glue_path)
    glues = glues.drop_duplicates(subset='SMILES').reset_index(drop=True)
    glues['label'] = 1
    print(f'Total unique glues: {len(glues)}')
    
    # Compute glue MWs
    print('Computing glue molecular weights...')
    glues['MW'] = glues['SMILES'].apply(compute_mw)
    glues = glues.dropna(subset=['MW']).reset_index(drop=True)
    print(f'  Valid glues: {len(glues)}')
    print(f'  Glue MW: mean={glues.MW.mean():.0f}±{glues.MW.std():.0f} '
          f'[{glues.MW.min():.0f}-{glues.MW.max():.0f}]')

    # ---- Load and pool ALL negative sources ----
    print(f'\nLoading negatives from {len(args.neg_paths)} source(s):')
    all_negatives = []
    for neg_path in args.neg_paths:
        if not os.path.exists(neg_path):
            print(f'  [!] Skipping {neg_path} (file not found)')
            continue
        neg_df = load_negatives(neg_path)
        all_negatives.append(neg_df)
    
    if not all_negatives:
        raise FileNotFoundError("No negative source files found!")
    
    negatives = pd.concat(all_negatives, ignore_index=True)
    negatives = negatives.drop_duplicates(subset='SMILES').reset_index(drop=True)
    print(f'\nTotal pooled negatives: {len(negatives)}')

    # ---- Deduplicate: remove any negatives that overlap with glues ----
    glue_set = set(glues['SMILES'].values)
    before = len(negatives)
    negatives = negatives[~negatives['SMILES'].isin(glue_set)].reset_index(drop=True)
    overlap = before - len(negatives)
    if overlap > 0:
        print(f'  [!] Removed {overlap} negatives that overlap with glue set')

    # ---- Compute negative MWs ----
    print('Computing negative molecular weights...')
    negatives['MW'] = negatives['SMILES'].apply(compute_mw)
    negatives = negatives.dropna(subset=['MW']).reset_index(drop=True)
    print(f'  Valid negatives: {len(negatives)}')
    print(f'  Negative MW (raw): mean={negatives.MW.mean():.0f}±{negatives.MW.std():.0f} '
          f'[{negatives.MW.min():.0f}-{negatives.MW.max():.0f}]')

    # ---- MW-stratified sampling ----
    n_target = len(glues)
    print(f'\nMW-stratified sampling: selecting {n_target} negatives to match glue MW distribution...')
    negatives_matched = mw_stratified_sample(
        negatives, glues['MW'].values, n_target, seed=args.seed, n_bins=args.n_bins
    )
    
    # If we got fewer than target (due to deduplication), downsample glues to match
    n_balanced = min(len(glues), len(negatives_matched))
    glues = glues.sample(n=n_balanced, random_state=args.seed).reset_index(drop=True)
    negatives_matched = negatives_matched.sample(n=n_balanced, random_state=args.seed).reset_index(drop=True)
    negatives_matched['label'] = 0
    
    print(f'\nBalanced to {n_balanced} glues + {n_balanced} non-glues = {2*n_balanced} total')
    
    # ---- Verify MW matching ----
    print(f'\n=== MW Distribution Verification ===')
    print(f'  Glue MW:     mean={glues.MW.mean():.0f}±{glues.MW.std():.0f}  '
          f'median={glues.MW.median():.0f}  [{glues.MW.min():.0f}-{glues.MW.max():.0f}]')
    print(f'  Non-glue MW: mean={negatives_matched.MW.mean():.0f}±{negatives_matched.MW.std():.0f}  '
          f'median={negatives_matched.MW.median():.0f}  [{negatives_matched.MW.min():.0f}-{negatives_matched.MW.max():.0f}]')
    
    # Quick separability check
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    X = np.concatenate([glues.MW.values, negatives_matched.MW.values]).reshape(-1, 1)
    y = np.array([1]*len(glues) + [0]*len(negatives_matched))
    lr = LogisticRegression()
    lr.fit(X, y)
    pred = lr.predict(X)
    mw_acc = accuracy_score(y, pred) * 100
    print(f'  MW-only logistic regression accuracy: {mw_acc:.1f}%')
    if mw_acc > 60:
        print(f'  [!] WARNING: MW still partially separates classes ({mw_acc:.0f}%)')
        print(f'    This is expected if negative pool lacks high-MW molecules.')
        print(f'    Consider adding more diverse negative sources.')
    else:
        print(f'  [OK] Good! MW no longer trivially separates classes.')

    # ---- Validate SMILES ----
    dataset = pd.concat([
        glues[['SMILES', 'label']], 
        negatives_matched[['SMILES', 'label']]
    ], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print('\nValidating SMILES...')
    valid_mask = dataset['SMILES'].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    dataset = dataset[valid_mask].reset_index(drop=True)

    print(f'\nDataset: {len(dataset)} valid molecules')
    print(f'  Glues:     {(dataset["label"]==1).sum()}')
    print(f'  Non-glues: {(dataset["label"]==0).sum()}')

    # ---- Distribution comparison ----
    print('\nDistribution check:')
    print_stats(dataset[dataset['label']==1], 'Glues')
    print_stats(dataset[dataset['label']==0], 'Non-glues')

    # ---- Scaffold split: 80/10/10 ----
    train_df, val_df, test_df = scaffold_split(dataset, seed=args.seed)

    # ---- Save ----
    train_df.to_csv('data/classifier_train.csv', index=False)
    val_df.to_csv('data/classifier_val.csv', index=False)
    test_df.to_csv('data/classifier_test.csv', index=False)

    print(f'\n=== Stratified Scaffold Split Results ===')
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n_glue = (split_df['label']==1).sum()
        n_nonglue = (split_df['label']==0).sum()
        n_total = len(split_df)
        pct_glue = 100 * n_glue / n_total if n_total > 0 else 0
        pct_nonglue = 100 * n_nonglue / n_total if n_total > 0 else 0
        print(f'  {name:5s}: {n_total} molecules  '
              f'(glues: {n_glue} [{pct_glue:.1f}%], non-glues: {n_nonglue} [{pct_nonglue:.1f}%])')
    print(f'\n[OK] Saved: data/classifier_train.csv, data/classifier_val.csv, data/classifier_test.csv')


if __name__ == '__main__':
    main()
