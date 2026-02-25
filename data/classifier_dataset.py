# data/classifier_dataset.py

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pandas as pd
from rdkit import Chem

# Import your existing molecular graph converter
from data.molecular_graph import MolecularGraphConverter

class MolecularGlueClassifierDataset(Dataset):
    """
    Dataset for binary classification: glue vs. non-glue.
    """
    
    def __init__(self, csv_file, split='train', val_split=0.2, seed=42):
        """
        Args:
            csv_file: Path to CSV with SMILES and label columns
            split: 'train' or 'val'
            val_split: Fraction for validation
        """
        self.converter = MolecularGraphConverter()
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Train/val split
        n_val = int(len(df) * val_split)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        if split == 'train':
            self.df = df[n_val:]
        else:
            self.df = df[:n_val]
        
        print(f"{split.upper()} dataset: {len(self.df)} molecules")
        print(f"  Glues: {(self.df['label']==1).sum()} ({(self.df['label']==1).mean()*100:.1f}%)")
        print(f"  Non-glues: {(self.df['label']==0).sum()} ({(self.df['label']==0).mean()*100:.1f}%)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Convert SMILES to graph
        try:
            graph = self.converter.smiles_to_graph(row['SMILES'])
        except:
            # If conversion fails, try next molecule
            return self.__getitem__((idx + 1) % len(self))
        
        # Add label
        graph.y = torch.tensor([row['label']], dtype=torch.float32)
        graph.smiles = row['SMILES']
        
        return graph
    
    @staticmethod
    def collate_fn(batch):
        """Batch graphs together."""
        return Batch.from_data_list(batch)