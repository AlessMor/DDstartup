"""
Integration tests for all analysis methods and physics models.

Tests the full workflow of main.py with:
- Parametric analysis (T_seeded and lump)
- Sobol analysis (T_seeded and lump)

Uses params_test.py for faster execution with minimal parameter points.
Verifies that output HDF5 files are created with at least 1 successful result.
"""

import pytest
import subprocess
import sys
from pathlib import Path
import h5py
import shutil
import yaml


# Get the project root directory
# tests/test_all_methods.py -> dd_startup/
project_root = Path(__file__).resolve().parent.parent
ddstartup_root = project_root / "ddstartup"
inputs_dir = project_root / "inputs"
outputs_dir = project_root / "outputs"


def run_ddstartup_main(params_file, config_file, timeout=120):
    """Helper function to run main.py with correct environment."""
    import os
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    result = subprocess.run(
        [sys.executable, str(ddstartup_root / "main.py"),
         str(params_file),
         str(config_file)],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    
    return result


class TestParametricAnalysis:
    """Test parametric analysis with both physics models."""
    
    def test_parametric_tseeded(self, tmp_path):
        """Test parametric analysis with T_seeded model."""
        # Create a temporary config with custom output directory
        config_data = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'vector_length': 50,
            'total_time': 315360000,
            'n_jobs': 1,
            'chunk_size': None,
            'batch_size': None,
            'output_dir': str(tmp_path / "test_outputs"),
            'verbose': False
        }
        
        config_file = tmp_path / "test_parametric_tseeded.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Run main.py
        result = run_ddstartup_main(
            inputs_dir / "params_test.py",
            config_file,
            timeout=120
        )
        
        # Check that the command succeeded
        assert result.returncode == 0, f"main.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        
        # Find the output directory (should have timestamp)
        output_base = tmp_path / "test_outputs"
        assert output_base.exists(), f"Output directory not created: {output_base}"
        
        # Find the timestamped subdirectory
        subdirs = [d for d in output_base.iterdir() if d.is_dir()]
        assert len(subdirs) > 0, "No timestamped output directory created"
        
        output_dir = subdirs[0]
        
        # Check for HDF5 files
        h5_files = list(output_dir.glob("*.h5"))
        assert len(h5_files) > 0, f"No HDF5 files found in {output_dir}"
        
        # Verify at least one success in the HDF5 file
        h5_file = h5_files[0]
        with h5py.File(h5_file, 'r') as f:
            # HDF5 structure is flat with datasets at top level
            assert 'sol_success' in f, "No 'sol_success' dataset in HDF5 file"
            
            # Check for success field
            successes = f['sol_success'][:]
            num_successes = successes.sum()
            assert num_successes >= 1, f"No successful results found (0/{len(successes)})"
            print(f"✓ Parametric T_seeded: {num_successes}/{len(successes)} successes")
    
    def test_parametric_lump(self, tmp_path):
        """Test parametric analysis with lump model."""
        # Create a temporary config with custom output directory
        config_data = {
            'analysis_type': 'lump',
            'method': 'parametric',
            'vector_length': 50,
            'total_time': 315360000,
            'n_jobs': 1,
            'chunk_size': None,
            'batch_size': None,
            'output_dir': str(tmp_path / "test_outputs"),
            'verbose': False
        }
        
        config_file = tmp_path / "test_parametric_lump.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Run main.py
        result = run_ddstartup_main(
            inputs_dir / "params_test.py",
            config_file,
            timeout=120
        )
        
        # Check that the command succeeded
        assert result.returncode == 0, f"main.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        
        # Find the output directory (should have timestamp)
        output_base = tmp_path / "test_outputs"
        assert output_base.exists(), f"Output directory not created: {output_base}"
        
        # Find the timestamped subdirectory
        subdirs = [d for d in output_base.iterdir() if d.is_dir()]
        assert len(subdirs) > 0, "No timestamped output directory created"
        
        output_dir = subdirs[0]
        
        # Check for HDF5 files
        h5_files = list(output_dir.glob("*.h5"))
        assert len(h5_files) > 0, f"No HDF5 files found in {output_dir}"
        
        # Verify at least one success in the HDF5 file
        h5_file = h5_files[0]
        with h5py.File(h5_file, 'r') as f:
            # HDF5 structure is flat with datasets at top level
            assert 'sol_success' in f, "No 'sol_success' dataset in HDF5 file"
            
            # Check for success field
            successes = f['sol_success'][:]
            num_successes = successes.sum()
            assert num_successes >= 1, f"No successful results found (0/{len(successes)})"
            print(f"✓ Parametric lump: {num_successes}/{len(successes)} successes")


class TestSobolAnalysis:
    """Test Sobol sensitivity analysis with both physics models."""
    
    def test_sobol_tseeded(self, tmp_path):
        """Test Sobol analysis with T_seeded model."""
        # Create a temporary config with custom output directory
        # Use small N_SAMPLES and order=1 for faster testing
        config_data = {
            'analysis_type': 'T_seeded',
            'method': 'sobol',
            'vector_length': 50,
            'total_time': 315360000,
            'n_jobs': 1,
            'N_SAMPLES': 32,  # Small sample size for testing (default would be ~1024)
            'order': 1,  # First-order only (faster than second-order)
            'chunk_size': None,
            'batch_size': None,
            'output_dir': str(tmp_path / "test_outputs"),
            'verbose': False
        }
        
        config_file = tmp_path / "test_sobol_tseeded.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Run main.py
        result = run_ddstartup_main(
            inputs_dir / "params_test.py",
            config_file,
            timeout=120  # 2 minute timeout (reduced N_SAMPLES for testing)
        )
        
        # Check that the command succeeded
        assert result.returncode == 0, f"main.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        
        # Find the output directory (should have timestamp)
        output_base = tmp_path / "test_outputs"
        assert output_base.exists(), f"Output directory not created: {output_base}"
        
        # Find the timestamped subdirectory
        subdirs = [d for d in output_base.iterdir() if d.is_dir()]
        assert len(subdirs) > 0, "No timestamped output directory created"
        
        output_dir = subdirs[0]
        
        # Check for HDF5 files
        h5_files = list(output_dir.glob("*.h5"))
        assert len(h5_files) > 0, f"No HDF5 files found in {output_dir}"
        
        # Verify at least one success in the HDF5 file
        h5_file = h5_files[0]
        with h5py.File(h5_file, 'r') as f:
            # HDF5 structure is flat with datasets at top level
            assert 'sol_success' in f, "No 'sol_success' dataset in HDF5 file"
            
            # Check for success field
            successes = f['sol_success'][:]
            num_successes = successes.sum()
            assert num_successes >= 1, f"No successful results found (0/{len(successes)})"
            print(f"✓ Sobol T_seeded: {num_successes}/{len(successes)} successes")
            
            # Verify Sobol indices are present
            assert 'sobol_indices' in f, "No 'sobol_indices' group in HDF5 file"
            sobol_group = f['sobol_indices']
            assert 'first_order' in sobol_group, "First-order Sobol indices not found"
            assert 'total_order' in sobol_group, "Total Sobol indices not found"
    
    def test_sobol_lump(self, tmp_path):
        """Test Sobol analysis with lump model."""
        # Create a temporary config with custom output directory
        # Use small N_SAMPLES and order=1 for faster testing
        config_data = {
            'analysis_type': 'lump',
            'method': 'sobol',
            'vector_length': 50,
            'total_time': 315360000,
            'n_jobs': 1,
            'N_SAMPLES': 32,  # Small sample size for testing (default would be ~1024)
            'order': 1,  # First-order only (faster than second-order)
            'chunk_size': None,
            'batch_size': None,
            'output_dir': str(tmp_path / "test_outputs"),
            'verbose': False
        }
        
        config_file = tmp_path / "test_sobol_lump.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Run main.py
        result = run_ddstartup_main(
            inputs_dir / "params_test.py",
            config_file,
            timeout=120  # 2 minute timeout (reduced N_SAMPLES for testing)
        )
        
        # Check that the command succeeded
        assert result.returncode == 0, f"main.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        
        # Find the output directory (should have timestamp)
        output_base = tmp_path / "test_outputs"
        assert output_base.exists(), f"Output directory not created: {output_base}"
        
        # Find the timestamped subdirectory
        subdirs = [d for d in output_base.iterdir() if d.is_dir()]
        assert len(subdirs) > 0, "No timestamped output directory created"
        
        output_dir = subdirs[0]
        
        # Check for HDF5 files
        h5_files = list(output_dir.glob("*.h5"))
        assert len(h5_files) > 0, f"No HDF5 files found in {output_dir}"
        
        # Verify at least one success in the HDF5 file
        h5_file = h5_files[0]
        with h5py.File(h5_file, 'r') as f:
            # HDF5 structure is flat with datasets at top level
            assert 'sol_success' in f, "No 'sol_success' dataset in HDF5 file"
            
            # Check for success field
            successes = f['sol_success'][:]
            num_successes = successes.sum()
            assert num_successes >= 1, f"No successful results found (0/{len(successes)})"
            print(f"✓ Sobol lump: {num_successes}/{len(successes)} successes")
            
            # Verify Sobol indices are present
            assert 'sobol_indices' in f, "No 'sobol_indices' group in HDF5 file"
            sobol_group = f['sobol_indices']
            assert 'first_order' in sobol_group, "First-order Sobol indices not found"
            assert 'total_order' in sobol_group, "Total Sobol indices not found"


class TestEndToEndWorkflow:
    """Test complete workflow with all methods."""
    
    def test_all_methods_sequential(self, tmp_path):
        """
        Run all four analysis combinations sequentially and verify results.
        
        This test ensures that:
        1. All methods (parametric, sobol) work with both physics models (T_seeded, lump)
        2. Each produces valid HDF5 output files
        3. Each has at least one successful result
        4. Output directories are properly organized
        """
        configs = [
            ('parametric', 'T_seeded', 90),
            ('parametric', 'lump', 90),
            ('sobol', 'T_seeded', 180),
            ('sobol', 'lump', 180),
        ]
        
        results_summary = []
        
        for method, analysis_type, timeout in configs:
            test_name = f"{method}_{analysis_type}"
            
            # Create config
            config_data = {
                'analysis_type': analysis_type,
                'method': method,
                'vector_length': 50,
                'total_time': 315360000,
                'n_jobs': 1,
                'chunk_size': None,
                'batch_size': None,
                'output_dir': str(tmp_path / test_name / "outputs"),
                'verbose': False
            }
            
            # Add Sobol-specific settings for faster testing
            if method == 'sobol':
                config_data['N_SAMPLES'] = 32
                config_data['order'] = 1
            
            config_file = tmp_path / f"test_{test_name}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            # Run analysis
            result = run_ddstartup_main(
                inputs_dir / "params_test.py",
                config_file,
                timeout=timeout
            )
            
            # Verify success
            assert result.returncode == 0, f"{test_name} failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            
            # Find output directory
            output_base = tmp_path / test_name / "outputs"
            assert output_base.exists(), f"Output directory not created for {test_name}"
            
            subdirs = [d for d in output_base.iterdir() if d.is_dir()]
            assert len(subdirs) > 0, f"No timestamped directory for {test_name}"
            
            output_dir = subdirs[0]
            
            # Verify HDF5 file
            h5_files = list(output_dir.glob("*.h5"))
            assert len(h5_files) > 0, f"No HDF5 files for {test_name}"
            
            # Count successes
            h5_file = h5_files[0]
            with h5py.File(h5_file, 'r') as f:
                # HDF5 structure is flat with datasets at top level
                if 'sol_success' in f:
                    successes = f['sol_success'][:]
                    num_successes = successes.sum()
                    total = len(successes)
                    
                    assert num_successes >= 1, f"No successes in {test_name}"
                    results_summary.append(f"{test_name}: {num_successes}/{total} successes")
                    
                    # For Sobol, verify indices
                    if method == 'sobol':
                        assert 'sobol_indices' in f, f"No Sobol indices in {test_name}"
                        assert 'first_order' in f['sobol_indices'], f"No first_order indices in {test_name}"
                        assert 'total_order' in f['sobol_indices'], f"No total_order indices in {test_name}"
        
        # Print summary
        print("\n" + "="*60)
        print("END-TO-END TEST SUMMARY")
        print("="*60)
        for summary in results_summary:
            print(f"✓ {summary}")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
