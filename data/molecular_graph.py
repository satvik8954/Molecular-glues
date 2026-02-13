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

from config import (
    ATOM_TYPES, ATOM_TO_IDX, BOND_TYPES, BOND_TO_IDX,
    CHARGES, CHARGE_TO_IDX, HYBRIDIZATIONS, HYBRID_TO_IDX
)


class MolecularGraph:
    """
    Represents a molecule as a graph for diffusion models.

    Node features:
    - Atom type (one-hot, 10)
    - Formal charge (one-hot, 5)
    - Hybridization (one-hot, 4)
    - Is aromatic (binary)
    - Is in ring (binary)
    - Number of hydrogens (integer, 1)
    - Is conjugated (binary)

    Edge features:
    - Bond type (one-hot, 5)
    - Is aromatic (binary)
    - Is conjugated (binary)
    - Is in ring (binary)
    """

    def __init__(
        self,
        node_types: torch.Tensor,          # [N] atom type indices
        node_charges: torch.Tensor,         # [N] charge indices
        edge_index: torch.Tensor,           # [2, E] edge connectivity
        edge_types: torch.Tensor,           # [E] bond type indices
        node_hybridizations: Optional[torch.Tensor] = None,  # [N] hybridization indices
        node_aromatic: Optional[torch.Tensor] = None,        # [N] aromatic flags
        node_in_ring: Optional[torch.Tensor] = None,         # [N] ring flags
        node_num_hs: Optional[torch.Tensor] = None,          # [N] num hydrogens
        node_conjugated: Optional[torch.Tensor] = None,      # [N] conjugation flags
        edge_aromatic: Optional[torch.Tensor] = None,        # [E] aromatic flags
        edge_conjugated: Optional[torch.Tensor] = None,      # [E] conjugation flags
        edge_in_ring: Optional[torch.Tensor] = None,         # [E] ring flags
    ):
        N = len(node_types)
        E = edge_types.shape[0]
        self.node_types = node_types
        self.node_charges = node_charges
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.node_hybridizations = node_hybridizations if node_hybridizations is not None else torch.zeros(N, dtype=torch.long)
        self.node_aromatic = node_aromatic if node_aromatic is not None else torch.zeros(N)
        self.node_in_ring = node_in_ring if node_in_ring is not None else torch.zeros(N)
        self.node_num_hs = node_num_hs if node_num_hs is not None else torch.zeros(N)
        self.node_conjugated = node_conjugated if node_conjugated is not None else torch.zeros(N)
        self.edge_aromatic = edge_aromatic if edge_aromatic is not None else torch.zeros(E)
        self.edge_conjugated = edge_conjugated if edge_conjugated is not None else torch.zeros(E)
        self.edge_in_ring = edge_in_ring if edge_in_ring is not None else torch.zeros(E)

    @property
    def num_nodes(self) -> int:
        return len(self.node_types)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        # One-hot encode atom types
        node_type_onehot = torch.zeros(self.num_nodes, len(ATOM_TYPES))
        node_type_onehot.scatter_(1, self.node_types.unsqueeze(1), 1)

        # One-hot encode charges
        node_charge_onehot = torch.zeros(self.num_nodes, len(CHARGES))
        node_charge_onehot.scatter_(1, self.node_charges.unsqueeze(1), 1)

        # One-hot encode hybridizations
        node_hybrid_onehot = torch.zeros(self.num_nodes, len(HYBRIDIZATIONS))
        node_hybrid_onehot.scatter_(1, self.node_hybridizations.unsqueeze(1), 1)

        # Combine node features: [atom_types(10) + charges(5) + hybrid(4) + aromatic(1) + in_ring(1) + num_hs(1) + conjugated(1)] = 23
        x = torch.cat([
            node_type_onehot,
            node_charge_onehot,
            node_hybrid_onehot,
            self.node_aromatic.unsqueeze(1).float(),
            self.node_in_ring.unsqueeze(1).float(),
            self.node_num_hs.unsqueeze(1).float(),
            self.node_conjugated.unsqueeze(1).float(),
        ], dim=1)

        # Create edge feature matrix
        edge_type_onehot = torch.zeros(self.num_edges, len(BOND_TYPES))
        edge_type_onehot.scatter_(1, self.edge_types.unsqueeze(1), 1)

        # Edge features: [bond_types(5) + aromatic(1) + conjugated(1) + in_ring(1)] = 8
        edge_attr = torch.cat([
            edge_type_onehot,
            self.edge_aromatic.unsqueeze(1).float(),
            self.edge_conjugated.unsqueeze(1).float(),
            self.edge_in_ring.unsqueeze(1).float(),
        ], dim=1)

        return Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            node_types=self.node_types,
            node_charges=self.node_charges,
            node_hybridizations=self.node_hybridizations,
            edge_types=self.edge_types,
        )


def _get_hybridization_idx(atom) -> int:
    """Map RDKit hybridization to index."""
    hyb = atom.GetHybridization()
    mapping = {
        Chem.rdchem.HybridizationType.SP: HYBRID_TO_IDX['SP'],
        Chem.rdchem.HybridizationType.SP2: HYBRID_TO_IDX['SP2'],
        Chem.rdchem.HybridizationType.SP3: HYBRID_TO_IDX['SP3'],
    }
    return mapping.get(hyb, HYBRID_TO_IDX['OTHER'])


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
        node_hybridizations = []
        node_aromatic = []
        node_in_ring = []
        node_num_hs = []
        node_conjugated = []

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

            # Hybridization
            node_hybridizations.append(_get_hybridization_idx(atom))

            # Boolean / numeric features
            node_aromatic.append(1 if atom.GetIsAromatic() else 0)
            node_in_ring.append(1 if atom.IsInRing() else 0)
            node_num_hs.append(atom.GetTotalNumHs())
            # Atom-level conjugation: True if any neighboring bond is conjugated
            is_conj = any(b.GetIsConjugated() for b in atom.GetBonds()) if atom.GetBonds() else False
            node_conjugated.append(1 if is_conj else 0)

        # Extract edge features
        edge_indices = []
        edge_types = []
        edge_aromatic = []
        edge_conjugated = []
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
            is_conjugated = 1 if bond.GetIsConjugated() else 0
            is_in_ring = 1 if bond.IsInRing() else 0
            edge_aromatic.extend([is_aromatic, is_aromatic])
            edge_conjugated.extend([is_conjugated, is_conjugated])
            edge_in_ring.extend([is_in_ring, is_in_ring])

        # Handle molecules with no bonds (single atoms)
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types_t = torch.zeros(0, dtype=torch.long)
            edge_aromatic_t = torch.zeros(0)
            edge_conjugated_t = torch.zeros(0)
            edge_in_ring_t = torch.zeros(0)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_types_t = torch.tensor(edge_types, dtype=torch.long)
            edge_aromatic_t = torch.tensor(edge_aromatic, dtype=torch.float)
            edge_conjugated_t = torch.tensor(edge_conjugated, dtype=torch.float)
            edge_in_ring_t = torch.tensor(edge_in_ring, dtype=torch.float)

        # Create MolecularGraph
        mol_graph = MolecularGraph(
            node_types=torch.tensor(node_types, dtype=torch.long),
            node_charges=torch.tensor(node_charges, dtype=torch.long),
            edge_index=edge_index,
            edge_types=edge_types_t,
            node_hybridizations=torch.tensor(node_hybridizations, dtype=torch.long),
            node_aromatic=torch.tensor(node_aromatic, dtype=torch.float),
            node_in_ring=torch.tensor(node_in_ring, dtype=torch.float),
            node_num_hs=torch.tensor(node_num_hs, dtype=torch.float),
            node_conjugated=torch.tensor(node_conjugated, dtype=torch.float),
            edge_aromatic=edge_aromatic_t,
            edge_conjugated=edge_conjugated_t,
            edge_in_ring=edge_in_ring_t,
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

    Uses a connectivity-maximizing strategy: builds a spanning tree first
    to ensure all atoms are connected, then adds remaining bonds greedily
    while respecting valence limits. This avoids the fragmentation problem
    where greedy pruning produces tiny disconnected pieces.

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
            node_types = data.x[:, :len(ATOM_TYPES)].argmax(dim=1).cpu().numpy()

        if hasattr(data, 'edge_types'):
            edge_types = data.edge_types.cpu().numpy()
        else:
            edge_types = data.edge_attr[:, :len(BOND_TYPES)].argmax(dim=1).cpu().numpy()

        edge_index = data.edge_index.cpu().numpy()

        # ---- Collect bond list (deduplicated, skip NONE) ----
        bonds = {}
        for idx in range(edge_index.shape[1]):
            i, j = int(edge_index[0, idx]), int(edge_index[1, idx])
            if i >= j:
                continue
            bt_idx = int(edge_types[idx])
            bt_str = BOND_TYPES[bt_idx]
            if bt_str == 'NONE':
                continue
            key = (i, j)
            if key not in bonds:
                bonds[key] = bt_str

        if not bonds:
            return None

        # ---- Map atom types ----
        # Expanded max valences (allow N=5, S=6 for charged/oxidized forms)
        MAX_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'S': 6, 'F': 1,
                       'Cl': 1, 'Br': 1, 'P': 5, 'I': 1}
        BOND_ORDER = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 1.5}

        symbols = []
        for at_idx in node_types:
            sym = ATOM_TYPES[at_idx]
            if sym == 'Other':
                sym = 'C'
            symbols.append(sym)

        num_atoms = len(symbols)

        # Convert AROMATIC bonds to SINGLE for stability, TRIPLE→DOUBLE for safety
        clean_bonds = {}
        for (i, j), bt in bonds.items():
            if bt == 'AROMATIC':
                bt = 'SINGLE'
            elif bt == 'TRIPLE':
                bt = 'DOUBLE'
            clean_bonds[(i, j)] = bt

        # ---- PHASE 1: Build spanning tree to maximise connectivity ----
        # Scoring: prefer bonds involving carbon (higher-valence, backbone atoms)
        def bond_score(key):
            i, j = key
            bt = clean_bonds[key]
            score = 0
            # Strongly prefer C–C bonds (scaffold backbone)
            if symbols[i] == 'C' and symbols[j] == 'C':
                score += 100
            # Prefer bonds involving at least one carbon
            elif symbols[i] == 'C' or symbols[j] == 'C':
                score += 50
            # Prefer SINGLE bonds (use less valence budget)
            if bt == 'SINGLE':
                score += 10
            return score

        sorted_bond_keys = sorted(clean_bonds.keys(), key=bond_score, reverse=True)

        # Union-Find for spanning tree
        parent = list(range(num_atoms))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[ra] = rb
            return True

        atom_valence_used = [0.0] * num_atoms
        accepted_bonds = []

        # First pass: spanning tree edges (connect disconnected components)
        remaining = []
        for key in sorted_bond_keys:
            i, j = key
            bt = clean_bonds[key]
            order = BOND_ORDER.get(bt, 1)
            max_i = MAX_VALENCE.get(symbols[i], 4)
            max_j = MAX_VALENCE.get(symbols[j], 4)

            if atom_valence_used[i] + order <= max_i and atom_valence_used[j] + order <= max_j:
                if union(i, j):
                    accepted_bonds.append((key, bt))
                    atom_valence_used[i] += order
                    atom_valence_used[j] += order
                else:
                    remaining.append(key)
            else:
                # Try downgrading to SINGLE if it was DOUBLE
                if bt == 'DOUBLE' and atom_valence_used[i] + 1 <= max_i and atom_valence_used[j] + 1 <= max_j:
                    if union(i, j):
                        accepted_bonds.append((key, 'SINGLE'))
                        atom_valence_used[i] += 1
                        atom_valence_used[j] += 1
                    else:
                        remaining.append(key)

        # ---- PHASE 2: Add ring-closing / extra bonds from remaining ----
        for key in remaining:
            i, j = key
            bt = clean_bonds[key]
            order = BOND_ORDER.get(bt, 1)
            max_i = MAX_VALENCE.get(symbols[i], 4)
            max_j = MAX_VALENCE.get(symbols[j], 4)

            if atom_valence_used[i] + order <= max_i and atom_valence_used[j] + order <= max_j:
                accepted_bonds.append((key, bt))
                atom_valence_used[i] += order
                atom_valence_used[j] += order
            elif bt == 'DOUBLE' and atom_valence_used[i] + 1 <= max_i and atom_valence_used[j] + 1 <= max_j:
                accepted_bonds.append((key, 'SINGLE'))
                atom_valence_used[i] += 1
                atom_valence_used[j] += 1

        # ---- Identify connected atoms and remove isolates ----
        connected = set()
        for (i, j), _ in accepted_bonds:
            connected.add(i)
            connected.add(j)

        if len(connected) < 3:
            return None

        # Reindex to remove isolated atoms
        old_to_new = {}
        new_symbols = []
        for old_idx in sorted(connected):
            old_to_new[old_idx] = len(new_symbols)
            new_symbols.append(symbols[old_idx])

        # ---- Build RDKit molecule ----
        mol = Chem.RWMol()
        for sym in new_symbols:
            mol.AddAtom(Chem.Atom(sym))

        RDKIT_BOND = {
            'SINGLE': Chem.BondType.SINGLE,
            'DOUBLE': Chem.BondType.DOUBLE,
            'TRIPLE': Chem.BondType.TRIPLE,
        }
        for (i, j), bt in accepted_bonds:
            if i in old_to_new and j in old_to_new:
                ni, nj = old_to_new[i], old_to_new[j]
                mol.AddBond(ni, nj, RDKIT_BOND.get(bt, Chem.BondType.SINGLE))

        # ---- Sanitize and extract SMILES ----
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                 Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                 Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                 Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                 Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
                smiles = Chem.MolToSmiles(mol, canonical=False)
            except Exception:
                try:
                    smiles = Chem.MolToSmiles(mol, canonical=False)
                except Exception:
                    return None

        if not smiles:
            return None

        # Take largest fragment if disconnected
        if '.' in smiles:
            fragments = smiles.split('.')
            smiles = max(fragments, key=len)

        # Final validity check
        check_mol = Chem.MolFromSmiles(smiles)
        if check_mol is None:
            return None
        if check_mol.GetNumHeavyAtoms() < 3:
            return None

        return Chem.MolToSmiles(check_mol, canonical=True)

    except Exception:
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
