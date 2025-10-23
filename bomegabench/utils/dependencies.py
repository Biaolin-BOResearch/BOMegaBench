"""
Unified dependency management for BOMegaBench.

This module provides centralized dependency checking and error handling
for all optional dependencies used throughout the project.
"""

import importlib
import warnings
from typing import Dict, Optional, Tuple, Any


# Dependency configuration with installation instructions
DEPENDENCY_CONFIG = {
    "lassobench": {
        "import_name": "LassoBench.LassoBench",
        "package_name": "LassoBench",
        "install_cmd": "pip install git+https://github.com/ksehic/LassoBench.git",
        "description": "High-dimensional sparse regression benchmarks",
        "extras_note": "Requires additional dependencies: celer, sparse-ho, etc."
    },
    "bayesmark": {
        "import_name": "bayesmark",
        "package_name": "bayesmark",
        "install_cmd": "pip install bayesmark",
        "description": "Machine learning hyperparameter optimization benchmarks"
    },
    "hpobench": {
        "import_name": "hpobench",
        "package_name": "hpobench",
        "install_cmd": "pip install hpobench",
        "description": "Hyperparameter optimization benchmark library"
    },
    "configspace": {
        "import_name": "ConfigSpace",
        "package_name": "ConfigSpace",
        "install_cmd": "pip install ConfigSpace",
        "description": "Configuration space definitions for hyperparameter tuning"
    },
    "benchbase": {
        "import_name": "benchbase",
        "package_name": "benchbase",
        "install_cmd": "Custom installation required - see database tuning documentation",
        "description": "Database benchmarking tool"
    }
}


class DependencyStatus:
    """Track status of all optional dependencies."""

    def __init__(self):
        self._status: Dict[str, bool] = {}
        self._modules: Dict[str, Any] = {}
        self._errors: Dict[str, Exception] = {}
        self._check_all()

    def _check_all(self):
        """Check availability of all dependencies."""
        for dep_name, config in DEPENDENCY_CONFIG.items():
            self._status[dep_name] = self._try_import(dep_name, config)

    def _try_import(self, dep_name: str, config: Dict[str, str]) -> bool:
        """
        Try to import a dependency.

        Args:
            dep_name: Internal dependency name
            config: Dependency configuration

        Returns:
            True if import successful, False otherwise
        """
        try:
            module = importlib.import_module(config["import_name"])
            self._modules[dep_name] = module
            return True
        except ImportError as e:
            self._errors[dep_name] = e
            return False

    def is_available(self, dep_name: str) -> bool:
        """Check if a dependency is available."""
        if dep_name not in self._status:
            raise ValueError(f"Unknown dependency: {dep_name}")
        return self._status[dep_name]

    def get_module(self, dep_name: str) -> Optional[Any]:
        """Get the imported module if available."""
        return self._modules.get(dep_name)

    def get_error(self, dep_name: str) -> Optional[Exception]:
        """Get the import error if dependency failed to load."""
        return self._errors.get(dep_name)

    def get_all_status(self) -> Dict[str, bool]:
        """Get status of all dependencies."""
        return self._status.copy()

    def get_missing(self) -> Dict[str, Dict[str, str]]:
        """Get configuration for all missing dependencies."""
        return {
            name: config
            for name, config in DEPENDENCY_CONFIG.items()
            if not self._status[name]
        }


# Global dependency status instance
DEPENDENCIES_STATUS = DependencyStatus()


def check_dependency(dep_name: str, silent: bool = False) -> bool:
    """
    Check if a dependency is available.

    Args:
        dep_name: Name of the dependency to check
        silent: If True, don't show warnings

    Returns:
        True if dependency is available, False otherwise
    """
    available = DEPENDENCIES_STATUS.is_available(dep_name)

    if not available and not silent:
        config = DEPENDENCY_CONFIG[dep_name]
        warning_msg = (
            f"\n{'='*80}\n"
            f"{config['description']} not available.\n"
            f"Package '{config['package_name']}' is not installed.\n\n"
            f"Install with:\n"
            f"  {config['install_cmd']}\n"
        )
        if "extras_note" in config:
            warning_msg += f"\nNote: {config['extras_note']}\n"
        warning_msg += f"{'='*80}"
        warnings.warn(warning_msg, ImportWarning, stacklevel=2)

    return available


def require_dependency(dep_name: str, feature_description: str = "") -> Any:
    """
    Require a dependency or raise an ImportError with helpful message.

    Args:
        dep_name: Name of the required dependency
        feature_description: Description of the feature requiring this dependency

    Returns:
        The imported module

    Raises:
        ImportError: If the dependency is not available
    """
    if not DEPENDENCIES_STATUS.is_available(dep_name):
        config = DEPENDENCY_CONFIG[dep_name]
        error_msg = (
            f"{config['description']} is required "
            f"{f'for {feature_description} ' if feature_description else ''}"
            f"but package '{config['package_name']}' is not installed.\n\n"
            f"Install with:\n  {config['install_cmd']}"
        )
        if "extras_note" in config:
            error_msg += f"\n\nNote: {config['extras_note']}"

        raise ImportError(error_msg)

    return DEPENDENCIES_STATUS.get_module(dep_name)


def get_missing_dependencies() -> Dict[str, Dict[str, str]]:
    """
    Get information about all missing dependencies.

    Returns:
        Dictionary mapping dependency names to their configuration
    """
    return DEPENDENCIES_STATUS.get_missing()


def print_dependency_status():
    """Print a formatted status report of all dependencies."""
    print("\n" + "=" * 80)
    print("BOMegaBench Dependency Status")
    print("=" * 80)

    status = DEPENDENCIES_STATUS.get_all_status()

    print("\nCore Dependencies:")
    print("  ✓ torch")
    print("  ✓ numpy")

    print("\nOptional Dependencies:")
    for dep_name, config in DEPENDENCY_CONFIG.items():
        available = status[dep_name]
        symbol = "✓" if available else "✗"
        status_str = "installed" if available else "not installed"
        print(f"  {symbol} {config['package_name']:<20} {status_str:<15} - {config['description']}")

    missing = get_missing_dependencies()
    if missing:
        print("\nTo install missing dependencies:")
        for dep_name, config in missing.items():
            print(f"  {config['install_cmd']}")

    print("=" * 80 + "\n")


# For backward compatibility - specific dependency checks
def check_lassobench() -> Tuple[bool, Optional[Any], Optional[Any]]:
    """Check LassoBench availability (backward compatible)."""
    available = check_dependency("lassobench", silent=True)
    if available:
        module = DEPENDENCIES_STATUS.get_module("lassobench")
        return True, module.SyntheticBenchmark, module.RealBenchmark
    return False, None, None


def check_hpobench() -> bool:
    """Check HPOBench availability (backward compatible)."""
    return check_dependency("hpobench", silent=True)


def check_bayesmark() -> bool:
    """Check Bayesmark availability (backward compatible)."""
    return check_dependency("bayesmark", silent=True)


def check_configspace() -> bool:
    """Check ConfigSpace availability (backward compatible)."""
    return check_dependency("configspace", silent=True)
