"""
Molecular graph representation and conversion utilities.
Uses PyTorch Geometric Data objects for graph neural networks.
"""
from typing import Optional, List, Tuple
import torch
from torch_geometric.data import Data
from rdkit import Chem

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import ATOM_TYPES, ATOM_TO_IDX, BOND_TYPES, BOND_TO_IDX, CHARGES, CHARGE_TO_IDX


class MolecularGraph:
    """
    Represents a molecule as a graph for diffusion models.
    
    Node features:
    - Atom type (one-hot)
    - Formal charge (one-hot)
    - Is aromatic (binary)
    - Is in ring (binary)
    
    Edge features:
    - Bond type (one-hot)
    - Is aromatic (binary)
    - Is conjugated (binary)
    - Is in ring (binary)
    """
    
    def __init__(
        self,
        node_types: torch.Tensor,      # [N] atom type indices
        node_charges: torch.Tensor,     # [N] charge indices  
        edge_index: torch.Tensor,       # [2, E] edge connectivity
        edge_types: torch.Tensor,       # [E] bond type indices
        node_aromatic: Optional[torch.Tensor] = None,  # [N] aromatic flags
        node_in_ring: Optional[torch.Tensor] = None,   # [N] ring flags
        edge_aromatic: Optional[torch.Tensor] = None,  # [E] aromatic flags
        edge_in_ring: Optional[torch.Tensor] = None,   # [E] ring flags
    ):
        self.node_types = node_types
        self.node_charges = node_charges
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.node_aromatic = node_aromatic if node_aromatic is not None else torch.zeros(len(node_types))
        self.node_in_ring = node_in_ring if node_in_ring is not None else torch.zeros(len(node_types))
        self.edge_aromatic = edge_aromatic if edge_aromatic is not None else torch.zeros(edge_types.shape[0])
        self.edge_in_ring = edge_in_ring if edge_in_ring is not None else torch.zeros(edge_types.shape[0])
    
    @property
    def num_nodes(self) -> int:
        return len(self.node_types)
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]
    
    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        # Create node feature matrix
        # One-hot encode atom types
        node_type_onehot = torch.zeros(self.num_nodes, len(ATOM_TYPES))
        node_type_onehot.scatter_(1, self.node_types.unsqueeze(1), 1)
        
        # One-hot encode charges
        node_charge_onehot = torch.zeros(self.num_nodes, len(CHARGES))
        node_charge_onehot.scatter_(1, self.node_charges.unsqueeze(1), 1)
        
        # Combine node features
        x = torch.cat([
            node_type_onehot,
            node_charge_onehot,
            self.node_aromatic.unsqueeze(1).float(),
            self.node_in_ring.unsqueeze(1).float(),
        ], dim=1)
        
        # Create edge feature matrix
        edge_type_onehot = torch.zeros(self.num_edges, len(BOND_TYPES))
        edge_type_onehot.scatter_(1, self.edge_types.unsqueeze(1), 1)
        
        edge_attr = torch.cat([
            edge_type_onehot,
            self.edge_aromatic.unsqueeze(1).float(),
            self.edge_in_ring.unsqueeze(1).float(),
        ], dim=1)
        
        return Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            node_types=self.node_types,
            node_charges=self.node_charges,
            edge_types=self.edge_types,
        )


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string
        
    Returns:
        PyG Data object or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens for proper valence, then remove for graph
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        
        # Extract node features
        node_types = []
        node_charges = []
        node_aromatic = []
        node_in_ring = []
        
        for atom in mol.GetAtoms():
            # Atom type
            symbol = atom.GetSymbol()
            if symbol in ATOM_TO_IDX:
                node_types.append(ATOM_TO_IDX[symbol])
            else:
                node_types.append(ATOM_TO_IDX['Other'])
            
            # Formal charge (clamp to valid range)
            charge = atom.GetFormalCharge()
            charge = max(-2, min(2, charge))
            node_charges.append(CHARGE_TO_IDX[charge])
            
            # Boolean features
            node_aromatic.append(1 if atom.GetIsAromatic() else 0)
            node_in_ring.append(1 if atom.IsInRing() else 0)
        
        # Extract edge features
        edge_indices = []
        edge_types = []
        edge_aromatic = []
        edge_in_ring = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Get bond type
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                bt = BOND_TO_IDX['SINGLE']
            elif bond_type == Chem.BondType.DOUBLE:
                bt = BOND_TO_IDX['DOUBLE']
            elif bond_type == Chem.BondType.TRIPLE:
                bt = BOND_TO_IDX['TRIPLE']
            elif bond_type == Chem.BondType.AROMATIC:
                bt = BOND_TO_IDX['AROMATIC']
            else:
                bt = BOND_TO_IDX['SINGLE']  # Default
            
            # Add edges in both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_types.extend([bt, bt])
            
            is_aromatic = 1 if bond.GetIsAromatic() else 0
            is_in_ring = 1 if bond.IsInRing() else 0
            edge_aromatic.extend([is_aromatic, is_aromatic])
            edge_in_ring.extend([is_in_ring, is_in_ring])
        
        # Handle molecules with no bonds (single atoms)
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)
            edge_aromatic = torch.zeros(0)
            edge_in_ring = torch.zeros(0)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_types = torch.tensor(edge_types, dtype=torch.long)
            edge_aromatic = torch.tensor(edge_aromatic, dtype=torch.float)
            edge_in_ring = torch.tensor(edge_in_ring, dtype=torch.float)
        
        # Create MolecularGraph
        mol_graph = MolecularGraph(
            node_types=torch.tensor(node_types, dtype=torch.long),
            node_charges=torch.tensor(node_charges, dtype=torch.long),
            edge_index=edge_index,
            edge_types=edge_types,
            node_aromatic=torch.tensor(node_aromatic, dtype=torch.float),
            node_in_ring=torch.tensor(node_in_ring, dtype=torch.float),
            edge_aromatic=edge_aromatic,
            edge_in_ring=edge_in_ring,
        )
        
        # Convert to PyG Data
        data = mol_graph.to_pyg_data()
        data.smiles = smiles
        
        return data
        
    except Exception as e:
        return None


def graph_to_smiles(data: Data) -> Optional[str]:
    """
    Convert a PyTorch Geometric Data object back to SMILES.
    
    This is a challenging task as the graph may not represent
    a valid molecule. We use RDKit's molecule editing capabilities.
    
    Args:
        data: PyG Data object with node_types and edge_types
        
    Returns:
        SMILES string or None if conversion fails
    """
    try:
        # Get node and edge types
        if hasattr(data, 'node_types'):
            node_types = data.node_types.cpu().numpy()
        else:
            # Decode from one-hot
            node_types = data.x[:, :len(ATOM_TYPES)].argmax(dim=1).cpu().numpy()
        
        if hasattr(data, 'edge_types'):
            edge_types = data.edge_types.cpu().numpy()
        else:
            edge_types = data.edge_attr[:, :len(BOND_TYPES)].argmax(dim=1).cpu().numpy()
        
        edge_index = data.edge_index.cpu().numpy()
        
        # Create editable molecule
        mol = Chem.RWMol()
        
        # Add atoms
        for atom_type_idx in node_types:
            symbol = ATOM_TYPES[atom_type_idx]
            if symbol == 'Other':
                symbol = 'C'  # Default to carbon
            atom = Chem.Atom(symbol)
            mol.AddAtom(atom)
        
        # Add bonds (only process each edge once)
        added_bonds = set()
        for idx in range(edge_index.shape[1]):
            i, j = edge_index[0, idx], edge_index[1, idx]
            if i >= j:  # Skip reverse edges
                continue
            
            bond_key = (min(i, j), max(i, j))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)
            
            bond_type_idx = edge_types[idx]
            bond_type_str = BOND_TYPES[bond_type_idx]
            
            if bond_type_str == 'SINGLE':
                bond_type = Chem.BondType.SINGLE
            elif bond_type_str == 'DOUBLE':
                bond_type = Chem.BondType.DOUBLE
            elif bond_type_str == 'TRIPLE':
                bond_type = Chem.BondType.TRIPLE
            elif bond_type_str == 'AROMATIC':
                bond_type = Chem.BondType.AROMATIC
            else:
                continue  # Skip NONE bonds
            
            mol.AddBond(int(i), int(j), bond_type)
        
        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # Verify the SMILES is valid
            check_mol = Chem.MolFromSmiles(smiles)
            if check_mol is None:
                return None
            
            return smiles
        except Exception:
            # Try without sanitization
            try:
                smiles = Chem.MolToSmiles(mol, canonical=False)
                return smiles
            except Exception:
                return None
        
    except Exception as e:
        return None


def batch_smiles_to_graphs(smiles_list: List[str]) -> List[Data]:
    """
    Convert a list of SMILES to graphs, filtering invalid ones.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of valid PyG Data objects
    """
    graphs = []
    for smiles in smiles_list:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
    return graphs
