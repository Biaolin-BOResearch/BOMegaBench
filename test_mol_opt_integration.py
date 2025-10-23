"""
Test script for mol_opt integration with BOMegaBench.

This script tests the integration of molecular optimization tasks
from mol_opt into the BOMegaBench framework.
"""

import sys
import traceback


def test_imports():
    """Test if mol_opt can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from bomegabench.functions.mol_opt import (
            MolOptFunction,
            create_mol_opt_suite,
            get_available_oracles,
        )
        print("✓ Successfully imported mol_opt components")
        return True
    except ImportError as e:
        print(f"✗ Failed to import mol_opt: {e}")
        traceback.print_exc()
        return False


def test_available_oracles():
    """Test getting available oracles."""
    print("\n" + "=" * 60)
    print("Testing available oracles...")
    print("=" * 60)

    try:
        from bomegabench.functions.mol_opt import get_available_oracles

        oracles = get_available_oracles()
        print(f"✓ Found {len(oracles)} available oracles")
        print(f"  Sample oracles: {oracles[:5]}")
        return True
    except Exception as e:
        print(f"✗ Failed to get available oracles: {e}")
        traceback.print_exc()
        return False


def test_create_function():
    """Test creating a MolOptFunction."""
    print("\n" + "=" * 60)
    print("Testing MolOptFunction creation...")
    print("=" * 60)

    try:
        from bomegabench.functions.mol_opt import MolOptFunction

        # Create a QED function
        func = MolOptFunction("QED")
        print(f"✓ Successfully created MolOptFunction for QED")
        print(f"  Function: {func}")
        print(f"  Metadata: {func.metadata}")
        return True
    except Exception as e:
        print(f"✗ Failed to create MolOptFunction: {e}")
        traceback.print_exc()
        return False


def test_evaluate_smiles():
    """Test evaluating SMILES strings."""
    print("\n" + "=" * 60)
    print("Testing SMILES evaluation...")
    print("=" * 60)

    try:
        from bomegabench.functions.mol_opt import MolOptFunction

        # Create a QED function
        func = MolOptFunction("QED")

        # Test molecules
        smiles_list = [
            "CCO",           # Ethanol
            "c1ccccc1",      # Benzene
            "CC(=O)O",       # Acetic acid
        ]

        print("Evaluating SMILES:")
        for smi in smiles_list:
            try:
                score = func.evaluate_smiles(smi)
                print(f"  {smi:20s} -> {score:.4f}")
            except Exception as e:
                print(f"  {smi:20s} -> Error: {e}")

        # Test batch evaluation
        scores = func.evaluate_smiles(smiles_list)
        print(f"\n✓ Batch evaluation successful: {scores}")
        return True
    except Exception as e:
        print(f"✗ Failed to evaluate SMILES: {e}")
        traceback.print_exc()
        return False


def test_create_suite():
    """Test creating mol_opt suite."""
    print("\n" + "=" * 60)
    print("Testing suite creation...")
    print("=" * 60)

    try:
        from bomegabench.functions.mol_opt import create_mol_opt_suite

        # Create a small suite
        oracle_names = ["QED", "LogP", "SA"]
        suite = create_mol_opt_suite(oracle_names)

        print(f"✓ Successfully created suite with {len(suite)} functions")
        print(f"  Functions: {suite.list_functions()}")

        # Test getting a function from suite
        qed_func = suite.get_function("QED")
        print(f"  Retrieved function: {qed_func}")
        return True
    except Exception as e:
        print(f"✗ Failed to create suite: {e}")
        traceback.print_exc()
        return False


def test_registry_integration():
    """Test integration with BOMegaBench registry."""
    print("\n" + "=" * 60)
    print("Testing registry integration...")
    print("=" * 60)

    try:
        from bomegabench.functions import list_suites, get_function

        # List all suites
        suites = list_suites()
        print(f"✓ Available suites: {suites}")

        # Check if mol_opt suites are available
        mol_opt_suites = [s for s in suites if "mol_opt" in s]
        print(f"  Mol_opt suites: {mol_opt_suites}")

        if mol_opt_suites:
            # Try to get a function from mol_opt suite
            func = get_function("QED", suite="mol_opt")
            print(f"✓ Successfully retrieved QED from registry: {func}")
            return True
        else:
            print("⚠ No mol_opt suites found in registry")
            return False
    except Exception as e:
        print(f"✗ Failed registry integration test: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MOL_OPT INTEGRATION TEST SUITE")
    print("=" * 60)

    results = []

    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Available Oracles", test_available_oracles),
        ("Function Creation", test_create_function),
        ("SMILES Evaluation", test_evaluate_smiles),
        ("Suite Creation", test_create_suite),
        ("Registry Integration", test_registry_integration),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
