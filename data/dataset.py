"""
Dataset class for molecular glue training data.
"""
import os
from typing import Optional, List, Callable
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from .molecular_graph import smiles_to_graph

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from utils.filters import is_drug_like, is_glue_like


class MolecularGlueDataset(Dataset):
    """
    PyTorch Dataset for molecular glue-like molecules.
    
    Loads molecules from CSV/SMILES files, converts to graphs,
    and filters by drug-likeness and glue-like properties.
    """
    
    def __init__(
        self,
        data_path: str,
        smiles_column: str = 'smiles',
        transform: Optional[Callable] = None,
        filter_drug_like: bool = True,
        filter_glue_like: bool = False,
        cache_graphs: bool = True,
        max_atoms: int = 50,
        verbose: bool = True,
    ):
        """
        Args:
            data_path: Path to CSV file or directory of SMILES files
            smiles_column: Column name for SMILES in CSV
            transform: Optional transform to apply to graphs
            filter_drug_like: Filter by Lipinski's Rule of Five
            filter_glue_like: Filter by glue-like properties
            cache_graphs: Cache converted graphs in memory
            max_atoms: Maximum number of atoms (filter larger molecules)
            verbose: Print loading progress
        """
        self.data_path = data_path
        self.smiles_column = smiles_column
        self.transform = transform
        self.filter_drug_like = filter_drug_like
        self.filter_glue_like = filter_glue_like
        self.cache_graphs = cache_graphs
        self.max_atoms = max_atoms
        self.verbose = verbose
        
        # Load and filter SMILES
        self.smiles_list = self._load_smiles()
        
        # Cache for converted graphs
        self._graph_cache = {}
        
        if verbose:
            print(f"Loaded {len(self.smiles_list)} molecules from {data_path}")
    
    def _load_smiles(self) -> List[str]:
        """Load SMILES from file(s) and apply filters."""
        smiles_list = []
        
        if os.path.isfile(self.data_path):
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
                smiles_list = df[self.smiles_column].dropna().tolist()
            elif self.data_path.endswith('.smi') or self.data_path.endswith('.txt'):
                with open(self.data_path, 'r') as f:
                    smiles_list = [line.strip().split()[0] for line in f if line.strip()]
        elif os.path.isdir(self.data_path):
            # Load from directory of files
            for filename in os.listdir(self.data_path):
                filepath = os.path.join(self.data_path, filename)
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                    smiles_list.extend(df[self.smiles_column].dropna().tolist())
                elif filename.endswith('.smi') or filename.endswith('.txt'):
                    with open(filepath, 'r') as f:
                        smiles_list.extend([line.strip().split()[0] for line in f if line.strip()])
        
        # Filter molecules
        filtered_smiles = []
        iterator = tqdm(smiles_list, desc="Filtering molecules") if self.verbose else smiles_list
        
        for smiles in iterator:
            # Convert to graph to check validity and size
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue
            
            if graph.x.shape[0] > self.max_atoms:
                continue
            
            # Drug-likeness filter
            if self.filter_drug_like and not is_drug_like(smiles):
                continue
            
            # Glue-like filter (optional, more restrictive)
            if self.filter_glue_like and not is_glue_like(smiles):
                continue
            
            filtered_smiles.append(smiles)
            
            # Cache the graph if enabled
            if self.cache_graphs:
                self._graph_cache[len(filtered_smiles) - 1] = graph
        
        return filtered_smiles
    
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Data:
        """Get graph for molecule at index."""
        # Check cache first
        if self.cache_graphs and idx in self._graph_cache:
            graph = self._graph_cache[idx]
        else:
            smiles = self.smiles_list[idx]
            graph = smiles_to_graph(smiles)
            if graph is None:
                # Return a dummy graph if conversion fails
                graph = Data(
                    x=torch.zeros((1, 17)),  # Minimal node features
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 7)),
                )
            if self.cache_graphs:
                self._graph_cache[idx] = graph
        
        if self.transform:
            graph = self.transform(graph)
        
        return graph
    
    def get_smiles(self, idx: int) -> str:
        """Get SMILES string at index."""
        return self.smiles_list[idx]
    
    def get_all_smiles(self) -> List[str]:
        """Get all SMILES strings."""
        return self.smiles_list.copy()
    
    @staticmethod
    def collate_fn(batch: List[Data]):
        """Custom collate function for batching graphs."""
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)


def create_train_val_split(
    dataset: MolecularGlueDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: MolecularGlueDataset instance
        val_ratio: Fraction of data for validation
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from torch.utils.data import Subset
    import numpy as np
    
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    
    val_size = int(len(dataset) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(dataset, val_indices.tolist())
    
    return train_dataset, val_dataset
