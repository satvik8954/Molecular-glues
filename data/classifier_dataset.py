# data/classifier_dataset.py
"""
Classification dataset using existing smiles_to_graph converter.
Binary classification: molecular glue (1) vs non-glue (0).
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import pandas as pd

# Import existing converter function
from data.molecular_graph import smiles_to_graph


class MolecularGlueClassifierDataset(Dataset):
    """
    Classification dataset for molecular glue detection.
    
    Uses the existing smiles_to_graph() function from molecular_graph.py.
    Only addition vs diffusion dataset: binary label field.
    """
    
    def __init__(self, csv_file, split='train', val_split=0.2, seed=42):
        # Load data
        df = pd.read_csv(csv_file)
        
        # Must have 'SMILES' and 'label' columns
        assert 'SMILES' in df.columns, "Need SMILES column"
        assert 'label' in df.columns, "Need label column (0=non-glue, 1=glue)"
        
        # Train/val split
        n_val = int(len(df) * val_split)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        if split == 'train':
            self.df = df[n_val:].reset_index(drop=True)
        else:
            self.df = df[:n_val].reset_index(drop=True)
        
        print(f"{split.upper()} dataset: {len(self.df)} molecules")
        print(f"  Glues: {(self.df['label']==1).sum()}")
        print(f"  Non-glues: {(self.df['label']==0).sum()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Use existing smiles_to_graph function (returns None on failure)
        graph = smiles_to_graph(row['SMILES'])
        if graph is None:
            # If conversion fails, try next molecule
            return self.__getitem__((idx + 1) % len(self))
        
        # Add label (only new part vs diffusion dataset)
        graph.y = torch.tensor([row['label']], dtype=torch.float32)
        graph.smiles = row['SMILES']
        graph.idx = idx
        
        return graph
    
    @staticmethod
    def collate_fn(batch):
        """Batch graphs together using PyG's standard collate."""
        return Batch.from_data_list(batch)