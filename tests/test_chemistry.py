"""
Unit tests for chemistry utilities.
"""
import pytest
import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.chemistry import (
    is_valid_molecule,
    check_valency,
    get_molecular_properties,
    canonicalize_smiles,
    calculate_tanimoto_similarity,
)


class TestIsValidMolecule:
    """Tests for is_valid_molecule function."""
    
    def test_valid_smiles(self):
        """Test valid SMILES strings."""
        assert is_valid_molecule('CCO')  # Ethanol
        assert is_valid_molecule('c1ccccc1')  # Benzene
        assert is_valid_molecule('CC(=O)O')  # Acetic acid
        assert is_valid_molecule('O=C1NC(=O)c2ccccc12')  # Phthalimide
    
    def test_invalid_smiles(self):
        """Test invalid SMILES strings."""
        assert not is_valid_molecule('')
        assert not is_valid_molecule(None)
        assert not is_valid_molecule('not_a_smiles')
        assert not is_valid_molecule('XXXXX')
    
    def test_edge_cases(self):
        """Test edge cases."""
        assert not is_valid_molecule(123)
        assert not is_valid_molecule([])


class TestGetMolecularProperties:
    """Tests for get_molecular_properties function."""
    
    def test_aspirin(self):
        """Test properties of aspirin."""
        props = get_molecular_properties('CC(=O)Oc1ccccc1C(=O)O')
        assert props is not None
        assert 178 < props['molecular_weight'] < 182  # ~180.16
        assert props['hbd'] == 1
        assert props['hba'] >= 3
        assert props['num_aromatic_rings'] == 1
    
    def test_benzene(self):
        """Test properties of benzene."""
        props = get_molecular_properties('c1ccccc1')
        assert props is not None
        assert 76 < props['molecular_weight'] < 80  # ~78.11
        assert props['num_aromatic_rings'] == 1
        assert props['hbd'] == 0
        assert props['hba'] == 0
    
    def test_invalid_molecule(self):
        """Test invalid molecules return None."""
        assert get_molecular_properties('invalid') is None
        assert get_molecular_properties(None) is None


class TestCanonicalizeSMILES:
    """Tests for canonicalize_smiles function."""
    
    def test_canonical_forms(self):
        """Test that different representations give same canonical form."""
        # Different representations of ethanol
        assert canonicalize_smiles('CCO') == canonicalize_smiles('OCC')
        
        # Different representations of benzene
        assert canonicalize_smiles('c1ccccc1') == canonicalize_smiles('C1=CC=CC=C1')
    
    def test_invalid_smiles(self):
        """Test invalid SMILES returns None."""
        assert canonicalize_smiles('invalid') is None


class TestTanimotoSimilarity:
    """Tests for Tanimoto similarity."""
    
    def test_identical_molecules(self):
        """Identical molecules should have similarity 1.0."""
        sim = calculate_tanimoto_similarity('CCO', 'CCO')
        assert sim == 1.0
    
    def test_similar_molecules(self):
        """Similar molecules should have high similarity."""
        # Ethanol vs methanol
        sim = calculate_tanimoto_similarity('CCO', 'CO')
        assert 0.3 < sim < 1.0
    
    def test_dissimilar_molecules(self):
        """Dissimilar molecules should have low similarity."""
        # Ethanol vs coronene (large PAH)
        sim = calculate_tanimoto_similarity('CCO', 'c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67')
        assert sim < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
