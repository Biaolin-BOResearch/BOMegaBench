"""Tests for dependency management system."""

import pytest
from bomegabench.utils.dependencies import (
    check_dependency,
    get_missing_dependencies,
    DEPENDENCIES_STATUS
)


def test_dependency_status():
    """Test dependency status tracking."""
    status = DEPENDENCIES_STATUS.get_all_status()
    assert isinstance(status, dict)
    assert "lassobench" in status
    assert "hpobench" in status
    assert "bayesmark" in status


def test_check_dependency():
    """Test dependency checking."""
    # Should not raise, just return bool
    result = check_dependency("lassobench", silent=True)
    assert isinstance(result, bool)


def test_get_missing_dependencies():
    """Test getting missing dependencies."""
    missing = get_missing_dependencies()
    assert isinstance(missing, dict)

    # Check that missing dependencies have proper config
    for dep_name, config in missing.items():
        assert "package_name" in config
        assert "install_cmd" in config
        assert "description" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
