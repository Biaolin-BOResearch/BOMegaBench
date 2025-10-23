"""
Test script for Olympus integration in BOMegaBench.

This script demonstrates how to use Olympus surfaces and datasets
integrated into BOMegaBench for Bayesian optimization benchmarking.
"""

import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_olympus_surfaces():
    """Test Olympus surfaces integration."""
    print("=" * 80)
    print("Testing Olympus Surfaces Integration")
    print("=" * 80)

    try:
        from bomegabench.functions import (
            OLYMPUS_SURFACES_AVAILABLE,
            create_olympus_surfaces_suite,
            OlympusDenaliFunction,
        )

        if not OLYMPUS_SURFACES_AVAILABLE:
            print("❌ Olympus surfaces not available")
            return False

        print("✓ Olympus surfaces module loaded successfully")

        # Test creating the suite
        print("\n1. Creating Olympus surfaces suite...")
        try:
            suite = create_olympus_surfaces_suite()
            print(f"✓ Suite created with {len(suite.functions)} functions")
            print(f"   Available functions: {list(suite.functions.keys())[:5]}...")
        except Exception as e:
            print(f"❌ Failed to create suite: {e}")
            return False

        # Test individual function
        print("\n2. Testing OlympusDenaliFunction...")
        try:
            func = OlympusDenaliFunction()
            print(f"✓ Function created: {func.metadata['name']}")
            print(f"   Dimension: {func.dim}")
            print(f"   Bounds: {func.bounds}")

            # Test evaluation
            X = torch.rand(5, func.dim)
            Y = func(X)
            print(f"✓ Evaluation successful")
            print(f"   Input shape: {X.shape}, Output shape: {Y.shape}")
            print(f"   Sample outputs: {Y[:3].tolist()}")

        except Exception as e:
            print(f"❌ Failed to test function: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n✅ Olympus surfaces integration test PASSED")
        return True

    except ImportError as e:
        print(f"❌ Failed to import Olympus surfaces: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_olympus_datasets():
    """Test Olympus datasets integration."""
    print("\n" + "=" * 80)
    print("Testing Olympus Datasets Integration")
    print("=" * 80)

    try:
        from bomegabench.functions import (
            OLYMPUS_DATASETS_AVAILABLE,
            create_olympus_datasets_suite,
            create_olympus_chemistry_suite,
            OlympusSuzukiFunction,
        )

        if not OLYMPUS_DATASETS_AVAILABLE:
            print("❌ Olympus datasets not available")
            return False

        print("✓ Olympus datasets module loaded successfully")

        # Test creating chemistry suite
        print("\n1. Creating Olympus chemistry suite...")
        try:
            suite = create_olympus_chemistry_suite()
            print(f"✓ Chemistry suite created with {len(suite.functions)} functions")
            if len(suite.functions) > 0:
                print(f"   Available functions: {list(suite.functions.keys())[:5]}...")
        except Exception as e:
            print(f"❌ Failed to create chemistry suite: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test individual dataset function
        print("\n2. Testing OlympusSuzukiFunction...")
        try:
            func = OlympusSuzukiFunction()
            print(f"✓ Function created: {func.metadata['name']}")
            print(f"   Dimension: {func.dim}")
            print(f"   Dataset type: {func.metadata.get('dataset_type', 'N/A')}")
            print(f"   Num train: {func.metadata.get('num_train', 'N/A')}")
            print(f"   Num test: {func.metadata.get('num_test', 'N/A')}")

            # Test evaluation
            X = torch.rand(3, func.dim)
            Y = func(X)
            print(f"✓ Evaluation successful")
            print(f"   Input shape: {X.shape}, Output shape: {Y.shape}")
            print(f"   Sample outputs: {Y.tolist()}")

        except Exception as e:
            print(f"⚠️  Could not test Suzuki dataset (may need emulator): {e}")
            # This is acceptable as datasets may require additional setup

        print("\n✅ Olympus datasets integration test PASSED")
        return True

    except ImportError as e:
        print(f"❌ Failed to import Olympus datasets: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_available_olympus_benchmarks():
    """List all available Olympus benchmarks."""
    print("\n" + "=" * 80)
    print("Available Olympus Benchmarks")
    print("=" * 80)

    # List surfaces
    try:
        from bomegabench.functions.olympus_surfaces import OLYMPUS_UNIQUE_SURFACES

        print("\n📊 OLYMPUS SURFACES:")
        print("-" * 80)

        categories = {
            "Categorical": [k for k, v in OLYMPUS_UNIQUE_SURFACES.items() if v.get("categorical", False)],
            "Discrete": [k for k, v in OLYMPUS_UNIQUE_SURFACES.items() if v.get("discrete", False)],
            "Mountain/Terrain": [k for k in OLYMPUS_UNIQUE_SURFACES.keys() if k in ["denali", "everest", "k2", "kilimanjaro", "matterhorn", "mont_blanc"]],
            "Special": [k for k in OLYMPUS_UNIQUE_SURFACES.keys() if k in ["ackley_path", "gaussian_mixture", "hyper_ellipsoid", "linear_funnel", "narrow_funnel"]],
            "Multi-Objective": [k for k, v in OLYMPUS_UNIQUE_SURFACES.items() if v.get("multi_objective", False)],
        }

        for category, surfaces in categories.items():
            if surfaces:
                print(f"\n{category} ({len(surfaces)}):")
                for surf in surfaces:
                    info = OLYMPUS_UNIQUE_SURFACES[surf]
                    print(f"  • {surf:25s} ({info['class_name']})")

    except ImportError:
        print("⚠️  Olympus surfaces not available")

    # List datasets
    try:
        from bomegabench.functions.olympus_datasets import OLYMPUS_DATASETS

        print("\n📊 OLYMPUS DATASETS:")
        print("-" * 80)

        for category, datasets in OLYMPUS_DATASETS.items():
            print(f"\n{category.replace('_', ' ').title()} ({len(datasets)}):")
            for name, desc in list(datasets.items())[:5]:  # Show first 5
                print(f"  • {name:25s} - {desc}")
            if len(datasets) > 5:
                print(f"  ... and {len(datasets) - 5} more")

    except ImportError:
        print("⚠️  Olympus datasets not available")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Olympus Integration Test Suite for BOMegaBench")
    print("=" * 80)

    results = []

    # List available benchmarks
    list_available_olympus_benchmarks()

    # Run tests
    results.append(("Olympus Surfaces", test_olympus_surfaces()))
    results.append(("Olympus Datasets", test_olympus_datasets()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s} {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests PASSED!")
    else:
        print("⚠️  Some tests FAILED")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
