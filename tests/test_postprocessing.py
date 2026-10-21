"""Regression checks for loading legacy postprocessing inputs."""

import h5py
import numpy as np

from src.postprocessing.postprocess_functions import (
    PARAMETER_SCHEMA,
    load_h5_to_df,
    parse_filters_and_additional,
)


def test_legacy_inventory_passthrough_and_computed_cost(tmp_path, monkeypatch):
    # Parsing registers plotting metadata globally; restore it after this test.
    monkeypatch.setattr(
        "src.postprocessing.postprocess_functions.PARAMETER_SCHEMA",
        {name: dict(meta) for name, meta in PARAMETER_SCHEMA.items()},
    )
    config = {
        "additional_variables": {
            "I_target": {"expr": "", "unit": "kg"},
            "unrealized_profits_fp": {"expr": "unrealized_profits / capacity_factor"},
            "c_T": {"expr": "unrealized_profits_fp / I_target"},
        },
        "filters": ["c_T > 0", "c_T < 250"],
    }
    filters, additional, _, passthrough = parse_filters_and_additional(config)
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as f:
        for name, values in {
            "I_target": [1., 2., 1.],
            "unrealized_profits": [100., 300., 300.],
            "capacity_factor": [.5, .75, .5],
        }.items():
            f.create_dataset(name, data=values)

    df = load_h5_to_df(
        path,
        targets=["c_T"],
        filters_exprs=filters,
        additional_map=additional,
        passthrough_vars=passthrough,
        chunk_size=2,
        verbose=True,
    )

    np.testing.assert_allclose(df["I_target"], [1., 2., 1.])
    np.testing.assert_allclose(df["capacity_factor"], [.5, .75, .5])
    np.testing.assert_allclose(df["unrealized_profits_fp"], [200., 400., 600.])
    np.testing.assert_allclose(df["c_T"], [200., 200., 600.])
    # Failed filters retain rows for quartile probability accounting.
    assert df["_is_failed"].tolist() == [False, False, True]
