"""Integration test for multispecies analysis via src.main."""

from __future__ import annotations

import importlib
import sys

import h5py


def test_ddstartup_main_runs_multispecies_parametric(tmp_path, monkeypatch):
    params_file = tmp_path / "params_multispecies.yaml"
    config_file = tmp_path / "config_multispecies.yaml"
    output_dir = tmp_path / "outputs"

    params_file.write_text(
        """
parameters:
  V_plasma_field:
    type: scalar
    value: 600
    unit: m^3
  T_i_field:
    type: vector
    values: [14, 18]
    unit: keV
  n_tot_field:
    type: scalar
    value: 7.0e19
    unit: 1/m^3
  species_params:
    D:
      f_0: 0.6
      injection_mode: auto
    T:
      f_0: 0.0
      injection_mode: direct
    He3:
      f_0: 0.4
      injection_mode: auto
    He4:
      f_0: 0.0
      injection_mode: off
""".strip()
    )

    config_file.write_text(
        f"""
analysis_type: multispecies
method: parametric
vector_length: 40
max_simulation_time: 5.0e6
targets:
  - target_specie: T
    target_fraction_in_plasma: 0.1
enforce_constant_total_density: true
allow_negative_auto_injection: false
auto_injection_use_storage_limits: false
n_jobs: 1
chunk_size: 8
batch_size: 8
output_dir: {output_dir}
verbose: false
""".strip()
    )

    import src.main as mainmod

    importlib.reload(mainmod)

    monkeypatch.setattr(sys, "argv", ["ddstartup", str(params_file), str(config_file)])
    ret = mainmod.main()
    assert ret == 0

    h5_files = list(output_dir.glob("**/ddstartup_*.h5"))
    assert h5_files, "No ddstartup HDF5 file generated for multispecies run"
    latest = max(h5_files, key=lambda p: p.stat().st_mtime)

    with h5py.File(latest, "r") as f:
        assert f.attrs["analysis_type"] == "multispecies"
        assert f.attrs["method"] == "parametric"
        assert "n_D" in f
        assert "n_T" in f
        assert "N_ifc_T" in f
        assert "sum_dn_dt" in f
        assert f["n_T"].shape[1] == 40
