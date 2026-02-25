# data_prep/collect_classifier_data.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

def collect_glue_vs_nonglue_data():
    """
    Collect balanced dataset of glues vs. non-glues.
    
    Target:
      - 154 molecular glues (positive class)
      - 154 non-glues (negative class, matched properties)
    """
    
    # 1. Load molecular glues
    glues_df = pd.read_csv('data/glue_chemotypes.csv')
    print(f"Loaded {len(glues_df)} molecular glues")
    
    # Compute properties for glues
    glue_properties = []
    for smiles in glues_df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            props = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'Rings': Descriptors.RingCount(mol)
            }
            glue_properties.append(props)
    
    # 2. Load candidate non-glues (ChEMBL drug-like)
    chembl_df = pd.read_csv('data/raw/chembl_druglike.csv')
    
    # 3. Match properties to glues (stratified sampling)
    non_glues = []
    
    for glue_prop in glue_properties:
        # Find ChEMBL molecules with similar properties
        candidates = chembl_df[
            (chembl_df['MW'].between(glue_prop['MW'] * 0.9, glue_prop['MW'] * 1.1)) &
            (chembl_df['LogP'].between(glue_prop['LogP'] - 0.5, glue_prop['LogP'] + 0.5)) &
            (chembl_df['Rings'] == glue_prop['Rings'])
        ]
        
        if len(candidates) > 0:
            # Pick random one
            selected = candidates.sample(n=1).iloc[0]
            non_glues.append(selected)
    
    # 4. Create labeled dataset
    glues_df['label'] = 1
    non_glues_df = pd.DataFrame(non_glues)
    non_glues_df['label'] = 0
    
    # Combine and shuffle
    dataset = pd.concat([glues_df, non_glues_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    dataset.to_csv('data/classifier_dataset.csv', index=False)
    
    print(f"\nDataset created:")
    print(f"  Glues: {len(glues_df)}")
    print(f"  Non-glues: {len(non_glues_df)}")
    print(f"  Total: {len(dataset)}")
    print(f"  Balance: {dataset['label'].mean():.2f}")
    
    return dataset


def augment_dataset(df, target_size=1000):
    """
    Augment dataset with SMILES enumeration.
    
    Technique: Generate different SMILES for same molecule
    (different atom orderings)
    """
    augmented = []
    
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            continue
        
        # Add original
        augmented.append({
            'SMILES': row['SMILES'],
            'label': row['label']
        })
        
        # Generate 5 random SMILES variations
        for _ in range(5):
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented.append({
                'SMILES': random_smiles,
                'label': row['label']
            })
    
    augmented_df = pd.DataFrame(augmented)
    augmented_df = augmented_df.drop_duplicates(subset=['SMILES'])
    augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
    
    print(f"Augmented from {len(df)} to {len(augmented_df)} examples")
    
    return augmented_df


if __name__ == '__main__':
    # Step 1: Collect balanced dataset
    dataset = collect_glue_vs_nonglue_data()
    
    # Step 2: Augment for more training data
    augmented = augment_dataset(dataset, target_size=1000)
    augmented.to_csv('data/classifier_dataset_augmented.csv', index=False)