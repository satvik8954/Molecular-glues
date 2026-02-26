"""Create balanced classifier dataset from .smi glue library + ChEMBL negatives."""
import pandas as pd
from rdkit import Chem

# ---- Load glues from .smi file (tab-separated: SMILES  MCULE_ID) ----
smi_path = 'data/glue_chemotypes.smi'
glues_smi = pd.read_csv(smi_path, sep='\t', header=None, names=['SMILES', 'id'])
print(f'Loaded {len(glues_smi)} glue molecules from .smi file')

# Also load original glue_chemotypes.csv (small curated set)
glues_csv = pd.read_csv('data/glue_chemotypes.csv')
glues_csv = glues_csv[['smiles']].rename(columns={'smiles': 'SMILES'})
print(f'Loaded {len(glues_csv)} glue chemotypes from CSV')

# Combine both glue sources, deduplicate
all_glues = pd.concat([
    glues_smi[['SMILES']],
    glues_csv
], ignore_index=True).drop_duplicates(subset='SMILES').reset_index(drop=True)
all_glues['label'] = 1
print(f'Total unique glues: {len(all_glues)}')

# ---- Load ChEMBL druglike as negatives ----
chembl = pd.read_csv('data/chembl_druglike.csv')
print(f'Loaded {len(chembl)} ChEMBL druglike molecules')

# Balance: take same number of non-glues as glues (up to available)
n_glues = len(all_glues)
n_non_glues = min(n_glues, len(chembl))
non_glues = chembl[['smiles']].rename(columns={'smiles': 'SMILES'}).sample(
    n=n_non_glues, random_state=42
).copy()
non_glues['label'] = 0
print(f'Sampled {len(non_glues)} non-glues for balance')

# ---- Combine and shuffle ----
dataset = pd.concat([all_glues, non_glues], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Validate SMILES (remove invalid) ----
print('Validating SMILES...')
valid_mask = dataset['SMILES'].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
dataset = dataset[valid_mask].reset_index(drop=True)

# ---- Save ----
dataset.to_csv('data/classifier_dataset.csv', index=False)

print(f'\nDataset created: {len(dataset)} molecules')
print(f'  Glues:     {(dataset["label"]==1).sum()}')
print(f'  Non-glues: {(dataset["label"]==0).sum()}')
print(f'  Balance:   {dataset["label"].mean():.2%} positive')
