# data/classifier_dataset.py
"""
Classification dataset using existing smiles_to_graph converter.
Binary classification: molecular glue (1) vs non-glue (0).

Pre-converts all SMILES to graphs at init time for fast training.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import pandas as pd
from tqdm import tqdm

# Import existing converter function
from data.molecular_graph import smiles_to_graph


class MolecularGlueClassifierDataset(Dataset):
    """
    Classification dataset for molecular glue detection.
    
    Pre-caches all graph conversions at init time so epochs are fast.
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
        
        # Pre-convert ALL SMILES to graphs (one-time cost, massive epoch speedup)
        print(f"  Pre-converting SMILES to graphs...")
        self.graphs = []
        failed = 0
        for idx in tqdm(range(len(self.df)), desc=f'  {split} graphs', leave=False):
            row = self.df.iloc[idx]
            graph = smiles_to_graph(row['SMILES'])
            if graph is not None:
                graph.y = torch.tensor([row['label']], dtype=torch.float32)
                graph.smiles = row['SMILES']
                graph.idx = idx
                self.graphs.append(graph)
            else:
                failed += 1
        
        if failed > 0:
            print(f"  ⚠ {failed} SMILES failed conversion (skipped)")
        print(f"  ✓ {len(self.graphs)} graphs cached in memory")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    @staticmethod
    def collate_fn(batch):
        """Batch graphs together using PyG's standard collate."""
        return Batch.from_data_list(batch)