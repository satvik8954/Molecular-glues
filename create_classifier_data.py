"""
Create balanced classifier dataset with scaffold-based train/val/test split.

Scaffold split ensures molecules with the same Murcko scaffold (core ring system)
stay in the same split — prevents data leakage from structurally similar molecules.

Supports ZINC15 (.smi) or ChEMBL (.csv) as negative source with MW-matching.

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
        print(f'Loaded {len(df)} molecules from .smi file: {path}')
    elif ext == '.csv':
        df = pd.read_csv(path)
        # Normalize column name to 'SMILES'
        if 'smiles' in df.columns and 'SMILES' not in df.columns:
            df = df.rename(columns={'smiles': 'SMILES'})
        print(f'Loaded {len(df)} molecules from .csv file: {path}')
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .smi or .csv")
    
    return df[['SMILES']].drop_duplicates().reset_index(drop=True)


def mw_filter(df, mw_min, mw_max):
    """Filter molecules by molecular weight range."""
    print(f'  Filtering to MW range [{mw_min}, {mw_max}]...')
    
    valid = []
    for smiles in tqdm(df['SMILES'], desc='  MW filter', leave=False):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                if mw_min <= mw <= mw_max:
                    valid.append(smiles)
        except Exception:
            continue
    
    result = pd.DataFrame({'SMILES': valid})
    print(f'  {len(result)} molecules in MW range (from {len(df)})')
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
    parser.add_argument('--neg_path', default='data/zinc15_druglike.smi',
                        help='Path to negative molecules (.smi or .csv). '
                             'Can be ZINC15, ChEMBL, or any collection.')
    parser.add_argument('--mw_match', action='store_true', default=True,
                        help='Filter negatives to match glue MW distribution (default: True)')
    parser.add_argument('--no_mw_match', action='store_false', dest='mw_match',
                        help='Disable MW matching')
    parser.add_argument('--mw_margin', type=float, default=50,
                        help='MW margin beyond glue min/max (default: 50)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ---- Load glues ----
    glues = load_smi_file(args.glue_path)
    glues = glues.drop_duplicates(subset='SMILES').reset_index(drop=True)
    glues['label'] = 1
    print(f'Total unique glues: {len(glues)}')

    # ---- Load negatives ----
    print(f'\nLoading negatives from: {args.neg_path}')
    negatives = load_negatives(args.neg_path)

    # ---- Deduplicate: remove any negatives that overlap with glues ----
    glue_set = set(glues['SMILES'].values)
    before = len(negatives)
    negatives = negatives[~negatives['SMILES'].isin(glue_set)].reset_index(drop=True)
    overlap = before - len(negatives)
    if overlap > 0:
        print(f'  ⚠ Removed {overlap} negatives that overlap with glue set')

    # ---- MW matching ----
    if args.mw_match:
        # Compute glue MW range
        glue_mols = [Chem.MolFromSmiles(s) for s in glues['SMILES'][:2000]
                     if Chem.MolFromSmiles(str(s))]
        glue_mws = [Descriptors.MolWt(m) for m in glue_mols]
        mw_min = max(100, np.percentile(glue_mws, 5) - args.mw_margin)
        mw_max = np.percentile(glue_mws, 95) + args.mw_margin
        print(f'\n  Glue MW range (5th-95th percentile ± {args.mw_margin}): [{mw_min:.0f}, {mw_max:.0f}]')
        negatives = mw_filter(negatives, mw_min, mw_max)

    # ---- Balance: downsample BOTH to the smaller count for true 50/50 ----
    n_balanced = min(len(glues), len(negatives))
    glues = glues.sample(n=n_balanced, random_state=args.seed).reset_index(drop=True)
    negatives = negatives.sample(n=n_balanced, random_state=args.seed).reset_index(drop=True)
    negatives['label'] = 0
    print(f'\nBalanced to {n_balanced} glues + {n_balanced} non-glues = {2*n_balanced} total')

    # ---- Validate SMILES ----
    dataset = pd.concat([glues, negatives], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print('Validating SMILES...')
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
    print(f'\n✓ Saved: data/classifier_train.csv, data/classifier_val.csv, data/classifier_test.csv')


if __name__ == '__main__':
    main()
