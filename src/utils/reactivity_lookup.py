"""
Reactivity lookup tables for parametric analyses.

Supports:
- legacy DDstartup channels (DD, DT, optional DHe3)
- multispecies placeholder channels (TT, He3He3, THe3)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.physics.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_He3He3_placeholder,
    sigmav_THe3_placeholder,
    sigmav_TT_placeholder,
)


class ReactivityLookupTable:
    """
    Pre-computed lookup table for fusion reactivities keyed by ion temperature.
    """

    def __init__(
        self,
        temperatures: np.ndarray,
        include_DHe3: bool = False,
        include_extra_channels: bool = False,
    ):
        self.temperatures = np.unique(np.asarray(temperatures, dtype=float))
        self.include_DHe3 = bool(include_DHe3)
        self.include_extra_channels = bool(include_extra_channels)

        self.sigmav_DD_p_lookup: Dict[float, float] = {}
        self.sigmav_DD_n_lookup: Dict[float, float] = {}
        self.sigmav_DT_lookup: Dict[float, float] = {}
        self.sigmav_DHe3_lookup: Dict[float, float] = {}
        self.sigmav_TT_lookup: Dict[float, float] = {}
        self.sigmav_He3He3_lookup: Dict[float, float] = {}
        self.sigmav_THe3_ch1_lookup: Dict[float, float] = {}
        self.sigmav_THe3_ch2_lookup: Dict[float, float] = {}
        self.sigmav_THe3_ch3_lookup: Dict[float, float] = {}

        self._build_lookup_table()

    def _build_lookup_table(self) -> None:
        _, sigmav_DD_n_arr, sigmav_DD_p_arr = sigmav_DD_BoschHale(self.temperatures)
        sigmav_DT_arr = sigmav_DT_BoschHale(self.temperatures)

        sigmav_DHe3_arr = None
        if self.include_DHe3:
            sigmav_DHe3_arr = sigmav_DHe3_BoschHale(self.temperatures)

        sigmav_TT_arr = None
        sigmav_He3He3_arr = None
        sigmav_THe3_ch1_arr = None
        sigmav_THe3_ch2_arr = None
        sigmav_THe3_ch3_arr = None
        if self.include_extra_channels:
            sigmav_TT_arr = sigmav_TT_placeholder(self.temperatures)
            sigmav_He3He3_arr = sigmav_He3He3_placeholder(self.temperatures)
            sigmav_THe3_ch1_arr, sigmav_THe3_ch2_arr, sigmav_THe3_ch3_arr = sigmav_THe3_placeholder(
                self.temperatures
            )

        for i, T_i in enumerate(self.temperatures):
            T_key = round(float(T_i) / 0.1) * 0.1
            self.sigmav_DD_p_lookup[T_key] = float(sigmav_DD_p_arr[i])
            self.sigmav_DD_n_lookup[T_key] = float(sigmav_DD_n_arr[i])
            self.sigmav_DT_lookup[T_key] = float(sigmav_DT_arr[i])

            if sigmav_DHe3_arr is not None:
                self.sigmav_DHe3_lookup[T_key] = float(sigmav_DHe3_arr[i])

            if sigmav_TT_arr is not None:
                self.sigmav_TT_lookup[T_key] = float(sigmav_TT_arr[i])
                self.sigmav_He3He3_lookup[T_key] = float(sigmav_He3He3_arr[i])
                self.sigmav_THe3_ch1_lookup[T_key] = float(sigmav_THe3_ch1_arr[i])
                self.sigmav_THe3_ch2_lookup[T_key] = float(sigmav_THe3_ch2_arr[i])
                self.sigmav_THe3_ch3_lookup[T_key] = float(sigmav_THe3_ch3_arr[i])

    def get_sigmav_DT(self, T_i: float) -> float:
        T_key = round(float(T_i) / 0.1) * 0.1
        return self.sigmav_DT_lookup[T_key]

    def to_dict(self) -> Dict[str, Dict[float, float]]:
        out: Dict[str, Dict[float, float] | bool] = {
            "sigmav_DD_p": self.sigmav_DD_p_lookup,
            "sigmav_DD_n": self.sigmav_DD_n_lookup,
            "sigmav_DT": self.sigmav_DT_lookup,
            "include_DHe3": self.include_DHe3,
            "include_extra_channels": self.include_extra_channels,
        }
        if self.include_DHe3:
            out["sigmav_DHe3"] = self.sigmav_DHe3_lookup
        if self.include_extra_channels:
            out["sigmav_TT"] = self.sigmav_TT_lookup
            out["sigmav_He3He3"] = self.sigmav_He3He3_lookup
            out["sigmav_THe3_ch1"] = self.sigmav_THe3_ch1_lookup
            out["sigmav_THe3_ch2"] = self.sigmav_THe3_ch2_lookup
            out["sigmav_THe3_ch3"] = self.sigmav_THe3_ch3_lookup
        return out  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: Dict) -> "ReactivityLookupTable":
        instance = cls.__new__(cls)
        instance.include_DHe3 = bool(data.get("include_DHe3", False))
        instance.include_extra_channels = bool(data.get("include_extra_channels", False))
        instance.sigmav_DD_p_lookup = data["sigmav_DD_p"]
        instance.sigmav_DD_n_lookup = data["sigmav_DD_n"]
        instance.sigmav_DT_lookup = data["sigmav_DT"]
        instance.sigmav_DHe3_lookup = data.get("sigmav_DHe3", {})
        instance.sigmav_TT_lookup = data.get("sigmav_TT", {})
        instance.sigmav_He3He3_lookup = data.get("sigmav_He3He3", {})
        instance.sigmav_THe3_ch1_lookup = data.get("sigmav_THe3_ch1", {})
        instance.sigmav_THe3_ch2_lookup = data.get("sigmav_THe3_ch2", {})
        instance.sigmav_THe3_ch3_lookup = data.get("sigmav_THe3_ch3", {})
        instance.temperatures = np.array(sorted(instance.sigmav_DT_lookup.keys()), dtype=float)
        return instance

    def __len__(self) -> int:
        return len(self.temperatures)
