# data/classifier_dataset.py
"""
Classification dataset using existing smiles_to_graph converter.
Binary classification: molecular glue (1) vs non-glue (0).

Pre-converts all SMILES to graphs at init time for fast training.
Expects pre-split CSV files (from create_classifier_data.py scaffold split).

Supports graph augmentation (node masking + edge dropping) for regularization.
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
    Takes a single CSV file (train, val, or test) — no internal splitting.
    
    Graph augmentation (when enabled):
    - Node feature masking: randomly zeros 15% of atom features
    - Edge dropping: randomly removes 10% of edges
    """
    
    def __init__(self, csv_file, split_name='train', augment=False, augment_prob=0.5):
        """
        Args:
            csv_file: Path to CSV with 'SMILES' and 'label' columns
            split_name: Name for display only (e.g. 'train', 'val', 'test')
            augment: Whether to apply graph augmentation on __getitem__
            augment_prob: Probability of augmenting each sample
        """
        # Load data
        df = pd.read_csv(csv_file)
        
        # Must have 'SMILES' and 'label' columns
        assert 'SMILES' in df.columns, "Need SMILES column"
        assert 'label' in df.columns, "Need label column (0=non-glue, 1=glue)"
        
        self.df = df
        self.augment = augment
        self.augment_prob = augment_prob
        
        print(f"{split_name.upper()} dataset: {len(self.df)} molecules")
        print(f"  Glues: {(self.df['label']==1).sum()}")
        print(f"  Non-glues: {(self.df['label']==0).sum()}")
        if augment:
            print(f"  Augmentation: ON (prob={augment_prob})")
        
        # Pre-convert ALL SMILES to graphs (one-time cost, massive epoch speedup)
        print(f"  Pre-converting SMILES to graphs...")
        self.graphs = []
        failed = 0
        for idx in tqdm(range(len(self.df)), desc=f'  {split_name} graphs', leave=False):
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
        graph = self.graphs[idx]
        
        # Apply graph augmentation during training
        if self.augment and torch.rand(1).item() < self.augment_prob:
            graph = self._augment_graph(graph)
        
        return graph
    
    def _augment_graph(self, graph):
        """
        Graph augmentation for regularization.
        
        1. Node feature masking — randomly zeros 15% of atoms' features
        2. Edge dropping — randomly removes 10% of edges
        
        Both are mild enough to preserve molecular identity while
        preventing the model from memorizing exact feature patterns.
        """
        graph = graph.clone()
        
        # 1. Node feature masking (15% of atoms)
        num_nodes = graph.x.size(0)
        mask = torch.rand(num_nodes, device=graph.x.device) < 0.15
        if mask.any():
            graph.x[mask] = 0  # Zero out selected atom features
        
        # 2. Edge dropping (10% of edges)
        num_edges = graph.edge_index.size(1)
        if num_edges > 2:  # Keep at least 1 edge (undirected = 2)
            keep_mask = torch.rand(num_edges, device=graph.edge_index.device) > 0.1
            # Ensure at least one edge pair survives
            if keep_mask.sum() < 2:
                keep_mask[:2] = True
            graph.edge_index = graph.edge_index[:, keep_mask]
            graph.edge_attr = graph.edge_attr[keep_mask]
        
        return graph
    
    @staticmethod
    def collate_fn(batch):
        """Batch graphs together using PyG's standard collate."""
        return Batch.from_data_list(batch)