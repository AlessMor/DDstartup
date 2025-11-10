"""Integration tests for the main execution.

These tests run the full parametric computation for both T_seeded and lump
analysis types using the test parameter file located under `inputs/params_test.yaml`
and the corresponding config files `parametric_tseeded` and `parametric_lump`.

The test checks:
 - the main() function returns success (0)
 - a new HDF5 output file is created under `outputs/`
 - the HDF5 file contains expected metadata and groups
 - at least one numeric output dataset contains finite (non-NaN) values

NOTE: These are integration-level tests and can be slow. They are written to be
robust to small differences in filenames/timestamps by discovering the newest
created output file.
"""

from pathlib import Path
import sys
import os
import time
import h5py
import numpy as np
import importlib


def _find_repo_root():
	# Walk up from this file until we find setup.py or .git to detect repo root
	p = Path(__file__).resolve()
	for parent in p.parents:
		if (parent / 'setup.py').exists() or (parent / '.git').exists():
			return parent
	# Fallback: assume parent of tests/ is repo root
	return Path(__file__).resolve().parents[2]


def _latest_h5_in_outputs(repo_root, before_set=None):
	outputs_dir = repo_root / 'outputs'
	pattern = 'ddstartup_*.h5'
	candidates = list(outputs_dir.glob(f'**/{pattern}')) if outputs_dir.exists() else []
	if before_set is not None:
		candidates = [p for p in candidates if p not in before_set]
	if not candidates:
		return None
	return max(candidates, key=lambda p: p.stat().st_mtime)


def test_parametric_analyses_create_h5(tmp_path, monkeypatch):
	"""Run parametric T_seeded and lump analyses and validate outputs.

	Uses the parameters file name `params_test` (resolved by ddstartup) and the
	config names `parametric_tseeded` and `parametric_lump` which exist under
	`inputs/` in this repository.
	"""

	repo_root = _find_repo_root()
	sys.path.insert(0, str(repo_root))

	import ddstartup.main as mainmod

	# Ensure importlib reload to pick up any local edits
	importlib.reload(mainmod)

	configs = [
		('params_test', 'parametric_tseeded', 'T_seeded'),
		('params_test', 'parametric_lump', 'lump'),
	]

	for params_name, config_name, expected_analysis_type in configs:
		# Record existing h5 files so we can detect newly created one
		outputs_dir = repo_root / 'outputs'
		before = set(outputs_dir.glob('**/ddstartup_*.h5')) if outputs_dir.exists() else set()

		# Build argv as the CLI would receive it
		argv = ['ddstartup', params_name, config_name, '--verbose']
		monkeypatch.setattr(sys, 'argv', argv)

		# Run main (should return 0 on success)
		ret = mainmod.main()
		assert ret == 0, f"main() returned non-zero for {config_name}: {ret}"

		# Allow filesystem timestamp resolution
		time.sleep(0.1)

		# Find the newly created h5 file
		h5_file = _latest_h5_in_outputs(repo_root, before_set=before)
		assert h5_file is not None, "No new HDF5 output file found in outputs/"

		# Basic structure checks
		with h5py.File(h5_file, 'r') as f:
			# Metadata
			assert 'analysis_type' in f.attrs
			assert f.attrs['analysis_type'] == expected_analysis_type

			# Parameter fields group should exist and contain at least one _values dataset
			assert 'parameter_fields' in f
			param_group = f['parameter_fields']
			assert any(name.endswith('_values') for name in param_group.keys()), "No parameter_values datasets found"

			# At least one numeric output dataset should be present and contain finite values
			numeric_found = False
			for key in f.keys():
				if key == 'parameter_fields':
					continue
				# Skip attributes group-like fields
				ds = f[key]
				if isinstance(ds, h5py.Dataset) and ds.dtype.kind in ('f', 'i'):
					arr = ds[:]
					if np.any(np.isfinite(arr)):
						numeric_found = True
						break
			assert numeric_found, "No numeric dataset with finite values found in HDF5 file"

			# --- Additional check: verify per-combination consistency ---
			#  - Inputs at root should match parameter_fields/<name>_values
			#  - Outputs must exist at root and contain finite numeric values
			#  - If a field has aliases, ensure alias datasets (if present) match canonical
			from ddstartup.utils.parameter_registry import get_registry
			registry = get_registry()
			input_names = registry.get_input_names(expected_analysis_type)
			output_names = registry.get_output_names(expected_analysis_type)
			all_fields = registry.get_all_field_names(expected_analysis_type)

			# Determine number of combinations
			n_combinations = f.attrs.get('total_combinations', None)
			if n_combinations is None:
				# Fallback: take length of any scalar dataset
				n_combinations = None
				for key in f.keys():
					if key == 'parameter_fields':
						continue
					ds = f[key]
					if isinstance(ds, h5py.Dataset) and getattr(ds, 'shape', None):
						if len(ds.shape) >= 1 and ds.shape[0] > 0:
							n_combinations = ds.shape[0]
							break
			assert n_combinations is not None, "Unable to determine number of combinations from HDF5 file"
			n_combinations = int(n_combinations)

			tol = 1e-9

			# 1) Inputs: root dataset must match parameter_fields/<name>_values
			for name in input_names:
				param_ds_name = f"{name}_values"
				assert 'parameter_fields' in f and param_ds_name in f['parameter_fields'], (
					f"Parameter values for '{name}' not found under parameter_fields/{param_ds_name}"
				)
				pv = f['parameter_fields'][param_ds_name][:]

				assert name in f, f"Input field '{name}' missing at HDF5 root"
				ds = f[name][:]

				# Scalar expected for inputs; compare elementwise
				assert ds.shape[0] == n_combinations, (
					f"Input dataset '{name}' length {ds.shape[0]} != expected {n_combinations}"
				)
				assert pv.shape[0] == n_combinations, (
					f"Parameter values for '{name}' length {pv.shape[0]} != expected {n_combinations}"
				)

				for idx in range(n_combinations):
					a = float(ds[idx])
					b = float(pv[idx])
					assert np.isfinite(a) and np.isfinite(b), (
						f"Non-finite value for input '{name}' at index {idx}: root={a}, param={b}"
					)
					assert abs(a - b) <= tol, (
						f"Mismatch for input '{name}' at index {idx}: root={a} != param_values={b}"
					)

			# 2) Outputs: exist at root and are numeric/finite per combination
			for name in output_names:
				assert name in f, f"Expected output field '{name}' missing in HDF5 root"
				ds = f[name]
				# Skip non-dataset objects
				assert isinstance(ds, h5py.Dataset), f"Field '{name}' is not a dataset"
				if ds.dtype.kind in ('f', 'i', 'u'):
					arr = ds[:]
					# Iterate each element (supports scalar and vector fields)
					for idx, val in np.ndenumerate(arr):
						assert np.isfinite(val), f"Output '{name}' has non-finite value at {idx}: {val}"
				elif ds.dtype.kind == 'b':
					# Booleans: ensure they are boolean values (no NaNs)
					arr = ds[:]
					for idx, val in np.ndenumerate(arr):
						assert isinstance(val, (np.bool_, bool)), f"Output '{name}' element {idx} not boolean: {val}"
				else:
					# Strings or other types: skip value checks
					pass

			# 3) Aliases: if alias dataset exists, ensure equality with canonical
			params_schema = registry.parameters
			for field in all_fields:
				props = params_schema.get(field, {})
				aliases = props.get('aliases', [])
				if not aliases:
					continue
				# If canonical not present, skip
				if field not in f:
					continue
				canon = f[field][:]
				for alias in aliases:
					if alias in f:
						ali_ds = f[alias][:]
						assert canon.shape == ali_ds.shape, (
							f"Alias dataset '{alias}' shape {ali_ds.shape} != canonical '{field}' shape {canon.shape}"
						)
						# Numeric compare
						if getattr(canon, 'dtype', None) is not None and canon.dtype.kind in ('f', 'i', 'u'):
							assert np.allclose(canon, ali_ds, atol=tol, equal_nan=True), (
								f"Alias dataset '{alias}' does not match canonical '{field}'"
							)
						else:
							# For non-numeric, compare elementwise equality
							for idx, _ in np.ndenumerate(canon):
								assert canon[idx] == ali_ds[idx], (
									f"Alias '{alias}' value mismatch at {idx}: {ali_ds[idx]} != {canon[idx]}"
								)
