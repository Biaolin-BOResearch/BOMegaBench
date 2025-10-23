"""Tests for synthetic benchmark functions."""

import pytest
import torch
from bomegabench.functions.synthetic import create_synthetic_suite


def test_synthetic_suite_creation():
    """Test that synthetic suite can be created."""
    suite = create_synthetic_suite(dimensions=[2, 4])
    assert len(suite) > 0
    assert "F01_SphereRaw_2d" in suite.list_functions()


def test_sphere_function():
    """Test Sphere function evaluation."""
    suite = create_synthetic_suite(dimensions=[2])
    sphere = suite["F01_SphereRaw_2d"]

    # Test at origin (should be 0)
    X = torch.zeros(1, 2)
    y = sphere(X)
    assert torch.allclose(y, torch.zeros(1), atol=1e-6)

    # Test at [1, 1] (should be 2)
    X = torch.ones(1, 2)
    y = sphere(X)
    assert torch.allclose(y, torch.tensor([2.0]), atol=1e-6)


def test_function_metadata():
    """Test that functions have proper metadata."""
    suite = create_synthetic_suite(dimensions=[2])
    sphere = suite["F01_SphereRaw_2d"]

    metadata = sphere.metadata
    assert "name" in metadata
    assert "suite" in metadata
    assert "properties" in metadata
    assert "domain" in metadata


def test_batch_evaluation():
    """Test batch evaluation of functions."""
    suite = create_synthetic_suite(dimensions=[2])
    sphere = suite["F01_SphereRaw_2d"]

    # Evaluate batch of points
    X = torch.rand(10, 2)
    y = sphere(X)
    assert y.shape == (10,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
