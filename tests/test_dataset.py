"""
Unit tests for molecular graph conversions.
"""
import pytest
import torch
import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.molecular_graph import smiles_to_graph, graph_to_smiles, MolecularGraph
from config import ATOM_TYPES, BOND_TYPES


class TestSmilesToGraph:
    """Tests for SMILES to graph conversion."""
    
    def test_simple_molecule(self):
        """Test conversion of simple molecule."""
        graph = smiles_to_graph('CCO')  # Ethanol
        assert graph is not None
        assert graph.x.shape[0] == 3  # 3 heavy atoms
        assert graph.edge_index.shape[1] > 0  # Has edges
    
    def test_aromatic_molecule(self):
        """Test conversion of aromatic molecule."""
        graph = smiles_to_graph('c1ccccc1')  # Benzene
        assert graph is not None
        assert graph.x.shape[0] == 6  # 6 carbons
    
    def test_phthalimide(self):
        """Test IMiD-like scaffold."""
        graph = smiles_to_graph('O=C1NC(=O)c2ccccc12')
        assert graph is not None
        assert graph.x.shape[0] == 11  # Heavy atoms
    
    def test_has_required_attributes(self):
        """Test that graph has required attributes."""
        graph = smiles_to_graph('CCO')
        
        # Check required tensors exist
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'edge_attr')
        assert hasattr(graph, 'node_types')
        assert hasattr(graph, 'edge_types')
    
    def test_invalid_smiles(self):
        """Test invalid SMILES returns None."""
        assert smiles_to_graph('invalid') is None
        assert smiles_to_graph('') is None


class TestGraphToSmiles:
    """Tests for graph to SMILES conversion."""
    
    def test_roundtrip_simple(self):
        """Test roundtrip conversion for simple molecules."""
        original = 'CCO'
        graph = smiles_to_graph(original)
        recovered = graph_to_smiles(graph)
        
        # Should get valid SMILES (may differ from original due to canonicalization)
        assert recovered is not None
        
        # Re-convert should work
        graph2 = smiles_to_graph(recovered)
        assert graph2 is not None
    
    def test_roundtrip_aromatic(self):
        """Test roundtrip for aromatic molecules."""
        original = 'c1ccccc1'
        graph = smiles_to_graph(original)
        recovered = graph_to_smiles(graph)
        
        assert recovered is not None


class TestNodeFeatures:
    """Tests for node feature encoding."""
    
    def test_feature_dimensions(self):
        """Test node feature dimensions are correct."""
        graph = smiles_to_graph('CCO')
        
        # Expected: atom_types + charges + aromatic + in_ring
        expected_dim = len(ATOM_TYPES) + 5 + 2  # 10 + 5 + 2 = 17
        assert graph.x.shape[1] == expected_dim
    
    def test_edge_feature_dimensions(self):
        """Test edge feature dimensions."""
        graph = smiles_to_graph('CCO')
        
        # Expected: bond_types + aromatic + in_ring
        expected_dim = len(BOND_TYPES) + 2  # 5 + 2 = 7
        assert graph.edge_attr.shape[1] == expected_dim


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
