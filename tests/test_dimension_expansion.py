"""
Tests for dimension expansion utilities.
"""

import pytest
import numpy as np
import torch

from bomegabench.functions.synthetic.classical_core import (
    LevyFunction,
    StyblinskiTangFunction,
)
from bomegabench.utils.dimension_expansion import (
    DimensionMapping,
    DimensionExpandedFunction,
    DimensionDiscoveryResult,
    DimensionDiscoveryMetrics,
    create_dimension_expansion_test,
)


class TestDimensionExpandedFunction:
    """Tests for DimensionExpandedFunction class."""
    
    def test_basic_creation(self):
        """Test basic function creation."""
        base_func = LevyFunction(dim=3)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=5,
            shuffle=False
        )
        
        assert expanded.dim == 8
        assert expanded.mapping.original_dim == 3
        assert expanded.mapping.n_dummy_dims == 5
        assert expanded.mapping.total_dim == 8
        
    def test_dimension_mapping_no_shuffle(self):
        """Test dimension mapping without shuffling."""
        base_func = LevyFunction(dim=3)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=4,
            shuffle=False
        )
        
        # Without shuffle, real dims should be first
        assert expanded.mapping.real_dim_indices == [0, 1, 2]
        assert expanded.mapping.dummy_dim_indices == [3, 4, 5, 6]
        assert expanded.mapping.expanded_to_original == [0, 1, 2, None, None, None, None]
        
    def test_dimension_mapping_with_shuffle(self):
        """Test dimension mapping with shuffling."""
        base_func = LevyFunction(dim=3)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=4,
            shuffle=True,
            seed=42
        )
        
        # With shuffle, we should have 3 real and 4 dummy, but mixed
        assert len(expanded.mapping.real_dim_indices) == 3
        assert len(expanded.mapping.dummy_dim_indices) == 4
        assert set(expanded.mapping.real_dim_indices) | set(expanded.mapping.dummy_dim_indices) == set(range(7))
        
    def test_dummy_dimensions_invariance(self):
        """Test that dummy dimensions don't affect function output."""
        base_func = LevyFunction(dim=2)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=5,
            shuffle=False
        )
        
        # Create two inputs that differ only in dummy dimensions
        x1 = torch.zeros(1, 7)
        x1[0, :2] = torch.tensor([0.5, 0.5])  # Real dims
        
        x2 = x1.clone()
        x2[0, 2:] = torch.ones(5)  # Change all dummy dims
        
        y1 = expanded(x1)
        y2 = expanded(x2)
        
        assert torch.allclose(y1, y2, atol=1e-6), \
            f"Dummy dimensions should not affect output: {y1.item()} vs {y2.item()}"
            
    def test_real_dimensions_sensitivity(self):
        """Test that real dimensions do affect function output."""
        base_func = LevyFunction(dim=2)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=5,
            shuffle=False
        )
        
        # Create two inputs that differ in a real dimension
        x1 = torch.zeros(1, 7)
        x1[0, :2] = torch.tensor([0.5, 0.5])
        
        x2 = x1.clone()
        x2[0, 0] = 0.8  # Change first real dim
        
        y1 = expanded(x1)
        y2 = expanded(x2)
        
        assert not torch.allclose(y1, y2, atol=1e-3), \
            "Real dimensions should affect output"
            
    def test_bounds_normalization(self):
        """Test that bounds are normalized to [0, 1]."""
        base_func = LevyFunction(dim=3)  # Original bounds [-10, 10]
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=2,
            normalize_to_unit=True
        )
        
        bounds = expanded.bounds
        assert torch.allclose(bounds[0], torch.zeros(5))
        assert torch.allclose(bounds[1], torch.ones(5))
        
    def test_ground_truth_importance(self):
        """Test ground truth importance computation."""
        base_func = LevyFunction(dim=3)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=4,
            shuffle=False
        )
        
        importance = expanded.get_dimension_importance_ground_truth()
        
        assert len(importance) == 7
        assert np.allclose(importance[:3], 1.0)  # Real dims
        assert np.allclose(importance[3:], 0.0)  # Dummy dims
        
    def test_shuffled_consistency(self):
        """Test that same seed produces same shuffle."""
        base_func = LevyFunction(dim=3)
        
        expanded1 = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=5,
            shuffle=True,
            seed=123
        )
        
        expanded2 = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=5,
            shuffle=True,
            seed=123
        )
        
        assert expanded1.mapping.real_dim_indices == expanded2.mapping.real_dim_indices
        assert expanded1.mapping.permutation == expanded2.mapping.permutation


class TestDimensionDiscoveryMetrics:
    """Tests for DimensionDiscoveryMetrics class."""
    
    def test_metrics_creation(self):
        """Test metrics analyzer creation."""
        func, metrics = create_dimension_expansion_test(
            LevyFunction,
            original_dim=3,
            n_dummy_dims=5,
            seed=42
        )
        
        assert metrics.mapping.total_dim == 8
        assert len(metrics.ground_truth) == 8
    
    def test_gp_ard_analysis(self):
        """Test GP-ARD analysis with actual function evaluation."""
        func, metrics = create_dimension_expansion_test(
            LevyFunction,
            original_dim=3,
            n_dummy_dims=5,
            shuffle=False,
            seed=42
        )
        
        # Create trajectory with function values
        np.random.seed(42)
        n_points = 50
        X = np.random.rand(n_points, 8)
        Y = func(torch.from_numpy(X).float()).numpy()
        
        result = metrics.analyze_trajectory(X, Y)
        
        # Check all fields exist
        assert hasattr(result, 'importance_scores')
        assert hasattr(result, 'learned_length_scales')
        assert hasattr(result, 'auc_roc')
        assert hasattr(result, 'precision_at_k')
        assert result.method_used == "gp_ard"
        
        # Length scales should be learned
        assert result.learned_length_scales is not None
        assert len(result.learned_length_scales) == 8
        
    def test_report_generation(self):
        """Test report string generation."""
        func, metrics = create_dimension_expansion_test(
            LevyFunction,
            original_dim=3,
            n_dummy_dims=5
        )
        
        np.random.seed(42)
        X = np.random.rand(50, 8)
        Y = func(torch.from_numpy(X).float()).numpy()
        
        result = metrics.analyze_trajectory(X, Y)
        report = metrics.print_analysis_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 100
        assert "AUC-ROC" in report
        assert "GP-ARD" in report or "length_scale" in report.lower()

    def test_minimum_samples_requirement(self):
        """Test that GP-ARD requires minimum samples."""
        func, metrics = create_dimension_expansion_test(
            LevyFunction,
            original_dim=3,
            n_dummy_dims=5
        )
        
        # Only 3 samples - should raise error
        X = np.random.rand(3, 8)
        Y = np.random.rand(3)
        
        with pytest.raises(ValueError, match="at least 5 samples"):
            metrics.analyze_trajectory(X, Y)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_dimension_expansion_test(self):
        """Test create_dimension_expansion_test function."""
        func, metrics = create_dimension_expansion_test(
            StyblinskiTangFunction,
            original_dim=4,
            n_dummy_dims=6,
            seed=0
        )
        
        assert func.dim == 10
        assert isinstance(metrics, DimensionDiscoveryMetrics)
        assert metrics.mapping.original_dim == 4
        
    def test_different_base_functions(self):
        """Test with different base functions."""
        for FuncClass in [LevyFunction, StyblinskiTangFunction]:
            func, metrics = create_dimension_expansion_test(
                FuncClass,
                original_dim=2,
                n_dummy_dims=3
            )
            
            # Verify it works
            x = torch.rand(5, 5)
            y = func(x)
            assert y.shape == (5,)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_dummy_dims(self):
        """Test with zero dummy dimensions."""
        base_func = LevyFunction(dim=3)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=0,
            shuffle=False
        )
        
        assert expanded.dim == 3
        assert len(expanded.mapping.dummy_dim_indices) == 0
        
    def test_single_real_dim(self):
        """Test with single real dimension."""
        base_func = LevyFunction(dim=1)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=10,
            shuffle=True,
            seed=42
        )
        
        assert expanded.dim == 11
        assert len(expanded.mapping.real_dim_indices) == 1
        
    def test_many_dummy_dims(self):
        """Test with many dummy dimensions."""
        base_func = LevyFunction(dim=2)
        expanded = DimensionExpandedFunction(
            base_function=base_func,
            n_dummy_dims=100,
            shuffle=True,
            seed=42
        )
        
        assert expanded.dim == 102
        
        # Should still work correctly
        x = torch.rand(10, 102)
        y = expanded(x)
        assert y.shape == (10,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
