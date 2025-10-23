"""
Function registry for unified access to all benchmark functions.
"""

from typing import Dict, List, Optional, Any
from ..core import BenchmarkFunction, BenchmarkSuite

# Import consolidated functions suite
from .consolidated_functions import create_consolidated_suite

# Import LassoBench suites (with optional dependency)
try:
    from .lasso_bench import (
        LassoBenchSyntheticSuite,
        LassoBenchRealSuite
    )
    LASSO_BENCH_AVAILABLE = True
except ImportError:
    LassoBenchSyntheticSuite = None
    LassoBenchRealSuite = None
    LASSO_BENCH_AVAILABLE = False

# Import HPO benchmarks (with optional dependency)
try:
    from .hpo_benchmarks import HPOBenchmarksSuite
    HPO_AVAILABLE = True
except ImportError:
    HPOBenchmarksSuite = None
    HPO_AVAILABLE = False

# Import HPOBench benchmarks (with optional dependency)
try:
    from .hpobench_benchmarks import (HPOBenchMLSuite, HPOBenchODSuite,
                                      HPOBenchNASSuite, HPOBenchRLSuite,
                                      HPOBenchSurrogatesSuite)
    HPOBENCH_AVAILABLE = True
except ImportError:
    HPOBenchMLSuite = None
    HPOBenchODSuite = None
    HPOBenchNASSuite = None
    HPOBenchRLSuite = None
    HPOBenchSurrogatesSuite = None
    HPOBENCH_AVAILABLE = False

# Import Database Tuning benchmarks
try:
    from .database_tuning import DatabaseTuningSuite
    DATABASE_TUNING_AVAILABLE = True
except ImportError:
    DatabaseTuningSuite = None
    DATABASE_TUNING_AVAILABLE = False

# Molecular Optimization benchmarks - REMOVED
# Reason: MolOpt uses SMILES string inputs, not supported in current scope
MOL_OPT_AVAILABLE = False
MolOptSuite = None
MolOptBasicSuite = None
MolOptTargetsSuite = None
MolOptMPOSuite = None
MolOptRediscoverySuite = None

# Guacamol benchmarks - REMOVED
# Reason: GuacaMol uses SMILES string inputs (molecular optimization), not supported
GUACAMOL_AVAILABLE = False
GuacamolSuiteV1 = None
GuacamolSuiteV2 = None
GuacamolSuiteTrivial = None
GuacamolRediscoverySuite = None
GuacamolMPOSuite = None
GuacamolPropertiesSuite = None
GuacamolScaffoldSuite = None

# Create consolidated suite
ConsolidatedSuite = create_consolidated_suite()

# Global registry of all suites
_SUITES: Dict[str, BenchmarkSuite] = {
    "consolidated": ConsolidatedSuite,
}

# Add LassoBench suites if available
if LASSO_BENCH_AVAILABLE and LassoBenchSyntheticSuite is not None and LassoBenchRealSuite is not None:
    _SUITES.update({
        "lasso_synthetic": LassoBenchSyntheticSuite,
        "lasso_real": LassoBenchRealSuite,
    })

# Add HPO benchmarks suite if available
if HPO_AVAILABLE and HPOBenchmarksSuite is not None:
    _SUITES.update({
        "hpo": HPOBenchmarksSuite,
    })

# Add HPOBench suites if available
if HPOBENCH_AVAILABLE:
    if HPOBenchMLSuite is not None:
        _SUITES.update({
            "hpobench_ml": HPOBenchMLSuite,
        })
    if HPOBenchODSuite is not None:
        _SUITES.update({
            "hpobench_od": HPOBenchODSuite,
        })
    if HPOBenchNASSuite is not None:
        _SUITES.update({
            "hpobench_nas": HPOBenchNASSuite,
        })
    if HPOBenchRLSuite is not None:
        _SUITES.update({
            "hpobench_rl": HPOBenchRLSuite,
        })
    if HPOBenchSurrogatesSuite is not None:
        _SUITES.update({
            "hpobench_surrogates": HPOBenchSurrogatesSuite,
        })

# Add Database Tuning suite if available
if DATABASE_TUNING_AVAILABLE and DatabaseTuningSuite is not None:
    _SUITES.update({
        "database_tuning": DatabaseTuningSuite,
    })

# Add Molecular Optimization suites if available
if MOL_OPT_AVAILABLE:
    # MolOpt suites - REMOVED (string-based inputs not supported)
    pass

# Guacamol suites - REMOVED (string-based inputs not supported)


def get_function(name: str, suite: Optional[str] = None) -> BenchmarkFunction:
    """
    Get a benchmark function by name.
    
    Args:
        name: Function name
        suite: Suite name (optional). If None, searches all suites.
        
    Returns:
        BenchmarkFunction instance
        
    Raises:
        ValueError: If function not found
    """
    if suite is not None:
        if suite not in _SUITES:
            available_suites = list(_SUITES.keys())
            raise ValueError(f"Suite '{suite}' not found. Available: {available_suites}")
        return _SUITES[suite].get_function(name)
    
    # Search all suites
    for suite_name, suite_obj in _SUITES.items():
        try:
            return suite_obj.get_function(name)
        except ValueError:
            continue
    
    # Function not found in any suite
    all_functions = []
    for suite_obj in _SUITES.values():
        all_functions.extend(suite_obj.list_functions())
    
    raise ValueError(f"Function '{name}' not found in any suite. Available: {sorted(all_functions)}")


def list_functions(suite: Optional[str] = None) -> List[str]:
    """
    List all available function names.
    
    Args:
        suite: Suite name (optional). If None, lists all functions.
        
    Returns:
        List of function names
    """
    if suite is not None:
        if suite not in _SUITES:
            available_suites = list(_SUITES.keys())
            raise ValueError(f"Suite '{suite}' not found. Available: {available_suites}")
        return _SUITES[suite].list_functions()
    
    # List all functions from all suites
    all_functions = []
    for suite_obj in _SUITES.values():
        all_functions.extend(suite_obj.list_functions())
    
    return sorted(all_functions)


def list_suites() -> List[str]:
    """List all available suite names."""
    return list(_SUITES.keys())


def get_suite(name: str) -> BenchmarkSuite:
    """Get a benchmark suite by name."""
    if name not in _SUITES:
        available = list(_SUITES.keys())
        raise ValueError(f"Suite '{name}' not found. Available: {available}")
    return _SUITES[name]


def get_functions_by_property(property_name: str, property_value: Any, suite: Optional[str] = None) -> Dict[str, BenchmarkFunction]:
    """
    Get functions matching a specific property value.
    
    Args:
        property_name: Property to match
        property_value: Value to match
        suite: Suite name (optional). If None, searches all suites.
        
    Returns:
        Dictionary mapping function names to instances
    """
    matching = {}
    
    suites_to_search = [_SUITES[suite]] if suite else _SUITES.values()
    
    for suite_obj in suites_to_search:
        for func_name in suite_obj.list_functions():
            func = suite_obj.get_function(func_name)
            if func.metadata.get(property_name) == property_value:
                matching[func_name] = func
                
    return matching


def get_multimodal_functions(suite: Optional[str] = None) -> Dict[str, BenchmarkFunction]:
    """Get all multimodal functions."""
    multimodal = {}
    
    suites_to_search = [_SUITES[suite]] if suite else _SUITES.values()
    
    for suite_obj in suites_to_search:
        for func_name in suite_obj.list_functions():
            func = suite_obj.get_function(func_name)
            properties = func.metadata.get("properties", [])
            if "multimodal" in properties:
                multimodal[func_name] = func
                
    return multimodal


def get_unimodal_functions(suite: Optional[str] = None) -> Dict[str, BenchmarkFunction]:
    """Get all unimodal functions."""
    unimodal = {}
    
    suites_to_search = [_SUITES[suite]] if suite else _SUITES.values()
    
    for suite_obj in suites_to_search:
        for func_name in suite_obj.list_functions():
            func = suite_obj.get_function(func_name)
            properties = func.metadata.get("properties", [])
            if "unimodal" in properties:
                unimodal[func_name] = func
                
    return unimodal


def print_function_info(name: str, suite: Optional[str] = None) -> None:
    """Print detailed information about a function."""
    func = get_function(name, suite)
    metadata = func.metadata
    
    print(f"Function: {metadata.get('name', name)}")
    print(f"Suite: {metadata.get('suite', 'Unknown')}")
    print(f"Dimension: {func.dim}")
    print(f"Domain: {metadata.get('domain', 'Unknown')}")
    print(f"Global minimum: {metadata.get('global_min', 'Unknown')}")
    
    if 'properties' in metadata:
        print(f"Properties: {', '.join(metadata['properties'])}")
        
    if 'description' in metadata:
        print(f"Description: {metadata['description']}")
        
    print(f"Bounds: {func.bounds.tolist()}")


def get_function_summary() -> Dict[str, int]:
    """Get summary statistics of all functions."""
    summary = {}
    
    for suite_name, suite_obj in _SUITES.items():
        summary[suite_name] = len(suite_obj)
        
    summary["total"] = sum(summary.values())
    
    return summary 