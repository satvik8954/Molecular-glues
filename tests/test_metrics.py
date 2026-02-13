"""
Unit tests for evaluation metrics.
"""
import pytest
import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.metrics import (
    validity_rate,
    uniqueness_rate,
    novelty_rate,
    internal_diversity,
    scaffold_diversity,
    qed_score_stats,
    compute_glue_similarity,
    compute_all_metrics,
)


class TestValidityRate:
    def test_all_valid(self):
        molecules = ['CCO', 'c1ccccc1', 'CC(=O)O']
        assert validity_rate(molecules) == 1.0

    def test_some_invalid(self):
        molecules = ['CCO', 'invalid', 'c1ccccc1']
        rate = validity_rate(molecules)
        assert 0.5 < rate < 1.0

    def test_empty(self):
        assert validity_rate([]) == 0.0


class TestUniquenessRate:
    def test_all_unique(self):
        molecules = ['CCO', 'c1ccccc1', 'CC(=O)O']
        assert uniqueness_rate(molecules) == 1.0

    def test_duplicates(self):
        molecules = ['CCO', 'CCO', 'c1ccccc1']
        rate = uniqueness_rate(molecules)
        assert rate < 1.0


class TestScaffoldDiversity:
    def test_diverse_scaffolds(self):
        molecules = ['CCO', 'c1ccccc1', 'O=C1NC(=O)c2ccccc12']
        div = scaffold_diversity(molecules)
        assert div > 0.0

    def test_empty(self):
        assert scaffold_diversity([]) == 0.0


class TestQEDStats:
    def test_valid_molecules(self):
        molecules = ['CC(=O)Oc1ccccc1C(=O)O', 'c1ccccc1', 'CCO']
        stats = qed_score_stats(molecules)
        assert 'qed_mean' in stats
        assert 'qed_std' in stats
        assert 'qed_frac_good' in stats
        assert stats['qed_mean'] > 0

    def test_empty(self):
        stats = qed_score_stats([])
        assert stats['qed_mean'] == 0.0


class TestGlueSimilarity:
    def test_self_similarity(self):
        glues = ['O=C1NC(=O)c2ccccc12', 'c1ccc2c(c1)C(=O)NC2=O']
        result = compute_glue_similarity(glues, glues)
        assert result['mean_nn_sim'] == 1.0

    def test_some_similarity(self):
        generated = ['c1ccccc1', 'CCO']
        glues = ['O=C1NC(=O)c2ccccc12']
        result = compute_glue_similarity(generated, glues)
        assert 'mean_nn_sim' in result
        assert result['mean_nn_sim'] >= 0.0


class TestComputeAllMetrics:
    def test_basic_metrics(self):
        molecules = ['CCO', 'c1ccccc1', 'CC(=O)O']
        metrics = compute_all_metrics(molecules)
        assert 'validity' in metrics
        assert 'uniqueness' in metrics
        assert 'diversity' in metrics
        assert 'scaffold_diversity' in metrics
        assert 'qed_mean' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
