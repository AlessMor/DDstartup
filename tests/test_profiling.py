"""
Tests for system profiling utilities.

These tests validate that the lightweight profiling helpers set sensible
defaults based on the current machine and respect user-specified values.
"""

import pytest

from ddstartup.utils.system_profiler import (
    apply_parallelization_defaults,
    get_optimal_parameters,
    get_system_info,
    print_system_profile,
)


class TestSystemInfo:
    """Validate basic system information helpers."""

    def test_get_system_info_contains_expected_fields(self):
        info = get_system_info()
        for key in ("n_cores", "total_ram_gb", "available_ram_gb", "ram_percent_used"):
            assert key in info, f"Missing {key} in system info"
        assert isinstance(info["n_cores"], int)
        assert info["n_cores"] >= 1
        assert info["total_ram_gb"] > 0
        assert info["available_ram_gb"] > 0

    def test_get_system_info_values_are_bounded(self):
        info = get_system_info()
        assert info["available_ram_gb"] <= info["total_ram_gb"]
        assert 0 <= info["ram_percent_used"] <= 100
        assert "cpu_freq_mhz" in info

    def test_get_system_info_numeric_types(self):
        info = get_system_info()
        numeric_keys = ("total_ram_gb", "available_ram_gb", "ram_percent_used")
        for key in numeric_keys:
            assert isinstance(info[key], (int, float))


class TestOptimalParameters:
    """Ensure heuristics return sane parallelization parameters."""

    def test_parametric_defaults_respect_hardware(self):
        info = get_system_info()
        params = get_optimal_parameters("parametric", verbose=False)
        for key in ("n_jobs", "chunk_size", "batch_size"):
            assert key in params
            assert isinstance(params[key], int)
            assert params[key] > 0
        assert params["n_jobs"] <= info["n_cores"]
        assert params["chunk_size"] >= params["n_jobs"]
        assert "system_info" in params and isinstance(params["system_info"], dict)

    def test_sobol_defaults_include_sampling(self):
        info = get_system_info()
        params = get_optimal_parameters("sobol", verbose=False)
        for key in ("n_jobs", "chunk_size", "batch_size", "N_SAMPLES", "order"):
            assert key in params
            assert isinstance(params[key], int)
            assert params[key] > 0
        assert params["n_jobs"] <= info["n_cores"]
        assert params["order"] >= 2


class TestApplyParallelizationDefaults:
    """Check config population and preservation rules."""

    def test_populates_missing_fields(self):
        config = {"method": "parametric", "n_jobs": None, "chunk_size": None, "batch_size": None}
        updated = apply_parallelization_defaults(config.copy(), verbose=False)
        for key in ("n_jobs", "chunk_size", "batch_size"):
            assert isinstance(updated[key], int)
            assert updated[key] > 0

    def test_preserves_existing_values(self):
        config = {"method": "parametric", "n_jobs": 2, "chunk_size": 123, "batch_size": 456}
        updated = apply_parallelization_defaults(config.copy(), verbose=False)
        assert updated["n_jobs"] == 2
        assert updated["chunk_size"] == 123
        assert updated["batch_size"] == 456

    def test_sobol_specific_fields_are_filled(self):
        config = {
            "method": "sobol",
            "n_jobs": None,
            "chunk_size": None,
            "batch_size": None,
            "N_SAMPLES": None,
            "order": None,
        }
        updated = apply_parallelization_defaults(config.copy(), verbose=False)
        for key in ("n_jobs", "chunk_size", "batch_size", "N_SAMPLES", "order"):
            assert isinstance(updated[key], int)
            assert updated[key] > 0

    def test_apply_defaults_is_idempotent(self):
        config = {"method": "parametric", "n_jobs": None, "chunk_size": None, "batch_size": None}
        first = apply_parallelization_defaults(config.copy(), verbose=False)
        second = apply_parallelization_defaults(first.copy(), verbose=False)
        assert first == second


class TestPrintSystemProfile:
    """Smoke test for profile printing."""

    def test_print_system_profile_outputs_text(self, capsys):
        params = get_optimal_parameters("parametric", verbose=False)
        print_system_profile(params, analysis_method="parametric")
        out = capsys.readouterr().out
        assert "SYSTEM PROFILE" in out
        assert "CPU Cores" in out or "n_jobs" in out
