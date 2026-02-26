"""Create balanced classifier dataset from existing CSVs."""
import pandas as pd
from rdkit import Chem

# Load existing glue chemotypes as positives
glues = pd.read_csv('data/glue_chemotypes.csv')
print(f'Loaded {len(glues)} glue chemotypes')

# Load ChEMBL druglike as negatives
chembl = pd.read_csv('data/chembl_druglike.csv')
print(f'Loaded {len(chembl)} ChEMBL druglike molecules')

# Prepare glues
glues_clean = glues[['smiles']].copy()
glues_clean['label'] = 1

# Prepare non-glues (balanced — same number as glues)
n_glues = len(glues_clean)
non_glues = chembl[['smiles']].sample(n=min(n_glues, len(chembl)), random_state=42).copy()
non_glues['label'] = 0

# Combine and shuffle
dataset = pd.concat([glues_clean, non_glues], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Validate SMILES (remove invalid)
valid_mask = dataset['smiles'].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
dataset = dataset[valid_mask].reset_index(drop=True)

dataset.to_csv('data/classifier_dataset.csv', index=False)

print(f'\nDataset created: {len(dataset)} molecules')
print(f'  Glues:     {(dataset["label"]==1).sum()}')
print(f'  Non-glues: {(dataset["label"]==0).sum()}')
print(f'  Balance:   {dataset["label"].mean():.2%} positive')
