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
    calculate_qed,
    get_murcko_scaffold,
)


class TestIsValidMolecule:
    """Tests for is_valid_molecule function."""

    def test_valid_smiles(self):
        assert is_valid_molecule('CCO')  # Ethanol
        assert is_valid_molecule('c1ccccc1')  # Benzene
        assert is_valid_molecule('CC(=O)O')  # Acetic acid
        assert is_valid_molecule('O=C1NC(=O)c2ccccc12')  # Phthalimide

    def test_invalid_smiles(self):
        assert not is_valid_molecule('')
        assert not is_valid_molecule(None)
        assert not is_valid_molecule('not_a_smiles')
        assert not is_valid_molecule('XXXXX')

    def test_edge_cases(self):
        assert not is_valid_molecule(123)
        assert not is_valid_molecule([])


class TestGetMolecularProperties:
    """Tests for get_molecular_properties function."""

    def test_aspirin(self):
        props = get_molecular_properties('CC(=O)Oc1ccccc1C(=O)O')
        assert props is not None
        assert 178 < props['molecular_weight'] < 182  # ~180.16
        assert props['hbd'] == 1
        assert props['hba'] >= 3
        assert props['num_aromatic_rings'] == 1
        assert 'fraction_sp3' in props
        assert 'qed' in props
        assert props['qed'] > 0

    def test_benzene(self):
        props = get_molecular_properties('c1ccccc1')
        assert props is not None
        assert 76 < props['molecular_weight'] < 80  # ~78.11
        assert props['num_aromatic_rings'] == 1
        assert props['hbd'] == 0
        assert props['hba'] == 0

    def test_invalid_molecule(self):
        assert get_molecular_properties('invalid') is None
        assert get_molecular_properties(None) is None


class TestCanonicalizeSMILES:
    """Tests for canonicalize_smiles function."""

    def test_canonical_forms(self):
        assert canonicalize_smiles('CCO') == canonicalize_smiles('OCC')
        assert canonicalize_smiles('c1ccccc1') == canonicalize_smiles('C1=CC=CC=C1')

    def test_invalid_smiles(self):
        assert canonicalize_smiles('invalid') is None


class TestTanimotoSimilarity:
    """Tests for Tanimoto similarity."""

    def test_identical_molecules(self):
        sim = calculate_tanimoto_similarity('CCO', 'CCO')
        assert sim == 1.0

    def test_similar_molecules(self):
        sim = calculate_tanimoto_similarity('CCO', 'CO')
        assert 0.3 < sim < 1.0

    def test_dissimilar_molecules(self):
        sim = calculate_tanimoto_similarity('CCO', 'c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67')
        assert sim < 0.5


class TestQED:
    """Tests for QED calculation."""

    def test_valid_molecule(self):
        qed = calculate_qed('CC(=O)Oc1ccccc1C(=O)O')  # Aspirin
        assert 0.0 < qed <= 1.0

    def test_invalid_molecule(self):
        qed = calculate_qed('invalid')
        assert qed == 0.0

    def test_caffeine(self):
        qed = calculate_qed('Cn1c(=O)c2c(ncn2C)n(C)c1=O')
        assert 0.0 < qed <= 1.0


class TestMurckoScaffold:
    """Tests for Murcko scaffold extraction."""

    def test_benzene(self):
        scaffold = get_murcko_scaffold('c1ccccc1')
        assert scaffold is not None

    def test_aspirin(self):
        scaffold = get_murcko_scaffold('CC(=O)Oc1ccccc1C(=O)O')
        assert scaffold is not None

    def test_invalid(self):
        scaffold = get_murcko_scaffold('invalid')
        assert scaffold is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
