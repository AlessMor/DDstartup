"""Multispecies time-dependent fuel-cycle solver (T-seeded style)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

from src.physics.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_He3He3_placeholder,
    sigmav_THe3_placeholder,
    sigmav_TT_placeholder,
)

SPECIES: Tuple[str, ...] = ("D", "T", "He3", "He4")
N_SPECIES = len(SPECIES)
_MODE_OFF = 0
_MODE_DIRECT = 1
_MODE_AUTO = 2


def _build_index(
    species_enabled: Mapping[str, bool],
    ofc_enabled: Mapping[str, bool],
) -> Dict[Tuple[str, str], int]:
    idx: Dict[Tuple[str, str], int] = {}
    cursor = 0
    for sp in SPECIES:
        if not species_enabled.get(sp, False):
            continue
        if ofc_enabled.get(sp, False):
            idx[(sp, "ofc")] = cursor
            cursor += 1
        for comp in ("ifc", "st", "n"):
            idx[(sp, comp)] = cursor
            cursor += 1
    return idx


@njit(cache=True, fastmath=True)
def _safe_outflow(N: float, tau: float) -> float:
    if (not np.isfinite(tau)) or (tau <= 0.0):
        return 0.0
    return N / tau


@njit(cache=True, fastmath=True)
def _allocate_auto_injection_numba(
    required_total: float,
    auto_mask: np.ndarray,
    weights: np.ndarray,
    caps: np.ndarray,
) -> tuple[np.ndarray, float]:
    alloc = np.zeros(N_SPECIES, dtype=np.float64)

    if required_total <= 0.0:
        residual = required_total
        if residual < 0.0:
            residual = 0.0
        return alloc, residual

    remaining = required_total
    active = np.zeros(N_SPECIES, dtype=np.bool_)
    for i in range(N_SPECIES):
        if auto_mask[i] and (caps[i] > 0.0):
            active[i] = True

    for _ in range(N_SPECIES + 2):
        active_count = 0
        for i in range(N_SPECIES):
            if active[i]:
                active_count += 1

        if (remaining <= 0.0) or (active_count == 0):
            break

        wsum = 0.0
        for i in range(N_SPECIES):
            if active[i]:
                wi = weights[i]
                if wi < 0.0:
                    wi = 0.0
                wsum += wi

        before = remaining

        if wsum <= 0.0:
            for i in range(N_SPECIES):
                if not active[i]:
                    continue
                share = remaining / active_count
                room = caps[i] - alloc[i]
                if room < 0.0:
                    room = 0.0
                add = share
                if add > room:
                    add = room
                alloc[i] += add
        else:
            for i in range(N_SPECIES):
                if not active[i]:
                    continue
                wi = weights[i]
                if wi < 0.0:
                    wi = 0.0
                share = remaining * wi / wsum
                room = caps[i] - alloc[i]
                if room < 0.0:
                    room = 0.0
                add = share
                if add > room:
                    add = room
                alloc[i] += add

        alloc_sum = 0.0
        for i in range(N_SPECIES):
            alloc_sum += alloc[i]
        remaining = required_total - alloc_sum

        for i in range(N_SPECIES):
            if active[i] and ((caps[i] - alloc[i]) <= 1e-16):
                active[i] = False

        if abs(before - remaining) < 1e-16:
            break

    residual = required_total
    for i in range(N_SPECIES):
        residual -= alloc[i]
    if residual < 0.0:
        residual = 0.0
    return alloc, residual


@njit(cache=True, fastmath=True)
def _compute_rhs_and_control_numba(
    y: np.ndarray,
    idx_ofc: np.ndarray,
    idx_ifc: np.ndarray,
    idx_st: np.ndarray,
    idx_n: np.ndarray,
    species_enabled_arr: np.ndarray,
    ofc_enabled_arr: np.ndarray,
    mode_codes: np.ndarray,
    tau_p_arr: np.ndarray,
    tau_ifc_arr: np.ndarray,
    tau_ofc_arr: np.ndarray,
    lambda_arr: np.ndarray,
    N_st_min_arr: np.ndarray,
    Ndot_max_arr: np.ndarray,
    auto_weights_arr: np.ndarray,
    V_plasma: float,
    TBR_DT: float,
    TBR_DDn: float,
    sigmav_DD_p: float,
    sigmav_DD_n: float,
    sigmav_DT: float,
    sigmav_DHe3: float,
    sigmav_TT: float,
    sigmav_He3He3: float,
    sigmav_THe3_ch1: float,
    sigmav_THe3_ch2: float,
    sigmav_THe3_ch3: float,
    route_THe3_ch3_to_He4: bool,
    enforce_constant_total_density: bool,
    allow_negative_auto_injection: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    dydt = np.zeros_like(y)
    n_clamped = np.zeros(N_SPECIES, dtype=np.float64)
    plasma_source = np.zeros(N_SPECIES, dtype=np.float64)
    q = np.zeros(N_SPECIES, dtype=np.float64)
    Ndot = np.zeros(N_SPECIES, dtype=np.float64)

    # Plasma densities and reaction rates.
    for i in range(N_SPECIES):
        if (not species_enabled_arr[i]) or (idx_n[i] < 0):
            continue
        n_i = y[idx_n[i]]
        if n_i < 0.0:
            n_i = 0.0
        n_clamped[i] = n_i

    n_D = n_clamped[0]
    n_T = n_clamped[1]
    n_He3 = n_clamped[2]

    R_DD_p = 0.5 * n_D * n_D * sigmav_DD_p
    R_DD_n = 0.5 * n_D * n_D * sigmav_DD_n
    R_DT = n_D * n_T * sigmav_DT
    R_DHe3 = n_D * n_He3 * sigmav_DHe3
    R_TT = 0.5 * n_T * n_T * sigmav_TT
    R_He3He3 = 0.5 * n_He3 * n_He3 * sigmav_He3He3
    R_THe3_ch1 = n_T * n_He3 * sigmav_THe3_ch1
    R_THe3_ch2 = n_T * n_He3 * sigmav_THe3_ch2
    R_THe3_ch3 = n_T * n_He3 * sigmav_THe3_ch3
    R_THe3_total = R_THe3_ch1 + R_THe3_ch2 + R_THe3_ch3

    if species_enabled_arr[1]:
        # T source/sink:
        # + from DDp, - from DT, -2 from TT, - from all THe3 branches.
        plasma_source[1] = R_DD_p - R_DT - 2.0 * R_TT - R_THe3_total
    if species_enabled_arr[0]:
        # D source/sink:
        # - from DDp/DDn/DT/DHe3, + from THe3 channel 2 (T+He3 -> He4 + D).
        plasma_source[0] = -R_DD_p - R_DD_n - R_DT - R_DHe3 + R_THe3_ch2
    if species_enabled_arr[2]:
        # He3 source/sink:
        # + from DDn, - from DHe3, -2 from He3He3, - from all THe3 branches.
        plasma_source[2] = R_DD_n - R_DHe3 - 2.0 * R_He3He3 - R_THe3_total
    if species_enabled_arr[3]:
        he4_from_the3_ch3 = 0.0
        if route_THe3_ch3_to_He4:
            he4_from_the3_ch3 = R_THe3_ch3
        # He4 source:
        # + from DT and DHe3; + from TT and He3He3;
        # + from THe3 branches 1 and 2; optionally + from branch 3 if 5He is
        # treated as instantaneously decaying to He4+n.
        plasma_source[3] = (
            R_DT + R_DHe3 + R_TT + R_He3He3 + R_THe3_ch1 + R_THe3_ch2 + he4_from_the3_ch3
        )

    # Injection control.
    fixed_sum = 0.0
    auto_mask = np.zeros(N_SPECIES, dtype=np.bool_)
    auto_caps = np.zeros(N_SPECIES, dtype=np.float64)

    for i in range(N_SPECIES):
        if not species_enabled_arr[i]:
            continue

        q[i] = plasma_source[i] - n_clamped[i] / tau_p_arr[i]

        mode = mode_codes[i]
        N_ifc = 0.0
        N_st = 0.0
        if idx_ifc[i] >= 0:
            N_ifc = y[idx_ifc[i]]
        if idx_st[i] >= 0:
            N_st = y[idx_st[i]]

        if mode == _MODE_OFF:
            Ndot_i = 0.0
            fixed_sum += Ndot_i
            Ndot[i] = Ndot_i
        elif mode == _MODE_DIRECT:
            inj = N_ifc / tau_ifc_arr[i] - lambda_arr[i] * N_st
            if N_st > N_st_min_arr[i]:
                if inj < 0.0:
                    inj = 0.0
                if inj > Ndot_max_arr[i]:
                    inj = Ndot_max_arr[i]
            else:
                inj = 0.0
            fixed_sum += inj
            Ndot[i] = inj
        elif mode == _MODE_AUTO:
            auto_mask[i] = True
            auto_caps[i] = Ndot_max_arr[i]
            Ndot[i] = 0.0

    required_auto_total = np.nan
    unmet_auto_total = np.nan

    if enforce_constant_total_density:
        q_sum = 0.0
        for i in range(N_SPECIES):
            q_sum += q[i]

        raw_required_auto_total = -V_plasma * q_sum - fixed_sum
        neg_unmet = 0.0
        if (not allow_negative_auto_injection) and (raw_required_auto_total < 0.0):
            required_auto_total = 0.0
            neg_unmet = -raw_required_auto_total
        else:
            required_auto_total = raw_required_auto_total

        alloc, pos_unmet = _allocate_auto_injection_numba(
            required_auto_total, auto_mask, auto_weights_arr, auto_caps
        )
        for i in range(N_SPECIES):
            if auto_mask[i]:
                Ndot[i] = alloc[i]
        unmet_auto_total = pos_unmet + neg_unmet
    else:
        for i in range(N_SPECIES):
            if auto_mask[i]:
                Ndot[i] = 0.0

    sum_dn_dt = 0.0
    for i in range(N_SPECIES):
        sum_dn_dt += Ndot[i] / V_plasma + q[i]

    # State derivatives.
    for i in range(N_SPECIES):
        if not species_enabled_arr[i]:
            continue

        N_ofc = 0.0
        N_ifc = 0.0
        N_st = 0.0
        if idx_ofc[i] >= 0:
            N_ofc = y[idx_ofc[i]]
        if idx_ifc[i] >= 0:
            N_ifc = y[idx_ifc[i]]
        if idx_st[i] >= 0:
            N_st = y[idx_st[i]]

        exhaust_rate = (n_clamped[i] / tau_p_arr[i]) * V_plasma
        ofc_out = 0.0
        if ofc_enabled_arr[i]:
            ofc_out = _safe_outflow(N_ofc, tau_ofc_arr[i])
        ifc_out = _safe_outflow(N_ifc, tau_ifc_arr[i])
        inj_rate = Ndot[i]
        lam = lambda_arr[i]

        breeding_source = 0.0
        if i == 1:
            breeding_source = V_plasma * (TBR_DDn * R_DD_n + TBR_DT * R_DT)

        if ofc_enabled_arr[i]:
            dN_ofc_dt = breeding_source - ofc_out - lam * N_ofc
            dN_ifc_dt = ofc_out + exhaust_rate - ifc_out - lam * N_ifc
        else:
            dN_ofc_dt = 0.0
            dN_ifc_dt = exhaust_rate + breeding_source - ifc_out - lam * N_ifc

        if mode_codes[i] == _MODE_AUTO:
            dN_st_dt = ifc_out - lam * N_st
        else:
            dN_st_dt = ifc_out - inj_rate - lam * N_st

        if idx_ofc[i] >= 0:
            dydt[idx_ofc[i]] = dN_ofc_dt
        if idx_ifc[i] >= 0:
            dydt[idx_ifc[i]] = dN_ifc_dt
        if idx_st[i] >= 0:
            dydt[idx_st[i]] = dN_st_dt
        if idx_n[i] >= 0:
            n_raw = y[idx_n[i]]
            dydt[idx_n[i]] = Ndot[i] / V_plasma + plasma_source[i] - n_raw / tau_p_arr[i]

    return dydt, Ndot, required_auto_total, unmet_auto_total, sum_dn_dt


def solve_multispecies_ode_system(
    *,
    V_plasma: float,
    T_i: float,
    n_tot: float,
    f0: Mapping[str, float],
    species_params: Mapping[str, Mapping[str, float]],
    injection_mode: Mapping[str, str],
    automatic_injection_weights: Mapping[str, float],
    TBR_DT: float,
    TBR_DDn: float,
    max_simulation_time: float,
    vector_length: int,
    targets: Optional[list[Mapping[str, Any]]] = None,
    enforce_constant_total_density: bool = True,
    allow_negative_auto_injection: bool = False,
    auto_injection_use_storage_limits: bool = False,
    route_THe3_ch3_to_He4: bool = False,
    solver_method: str = "BDF",
    solver_rtol: float = 1e-6,
    solver_atol: float = 1e-3,
    reactivities: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Solve multispecies fuel-cycle ODE system and return physics diagnostics."""

    for sp in SPECIES:
        if sp not in species_params:
            raise ValueError(f"Missing species_params entry for {sp!r}")

    # Species activation: when False, the species is removed from all equations.
    species_enabled = {sp: bool(species_params[sp].get("enable_plasma_channel", True)) for sp in SPECIES}
    if not any(species_enabled.values()):
        raise ValueError("At least one species must have enable_plasma_channel=True")

    # Targets:
    # - targets=None or [] -> no target-based stopping
    # - list of dicts -> integration stops when any provided condition is reached
    target_conditions: list[Dict[str, Any]] = []
    if targets is not None:
        if not isinstance(targets, (list, tuple)):
            raise ValueError("targets must be a list of dictionaries or None")

        for i, target in enumerate(targets):
            if not isinstance(target, Mapping):
                raise ValueError(f"targets[{i}] must be a mapping")

            sp = target.get("target_specie", None)
            if sp is None:
                raise ValueError(f"targets[{i}] must define 'target_specie'")
            sp = str(sp)
            if sp not in SPECIES:
                raise ValueError(f"targets[{i}]['target_specie'] must be one of {SPECIES}")
            if not species_enabled.get(sp, False):
                raise ValueError(f"targets[{i}] refers to disabled species {sp!r}")

            frac = target.get("target_fraction_in_plasma", None)
            if frac is not None:
                target_conditions.append(
                    {
                        "target_specie": sp,
                        "metric": "fraction",
                        "value": float(frac),
                        "label": f"{sp}:fraction>={float(frac):.6e}",
                    }
                )

            inv_ifc = target.get("target_inventory_ifc", None)
            if inv_ifc is not None:
                target_conditions.append(
                    {
                        "target_specie": sp,
                        "metric": "ifc",
                        "value": float(inv_ifc),
                        "label": f"{sp}:ifc>={float(inv_ifc):.6e}",
                    }
                )

            inv_ofc = target.get("target_inventory_ofc", None)
            if inv_ofc is not None:
                target_conditions.append(
                    {
                        "target_specie": sp,
                        "metric": "ofc",
                        "value": float(inv_ofc),
                        "label": f"{sp}:ofc>={float(inv_ofc):.6e}",
                    }
                )

            inv_st = target.get("target_inventory_storage", None)
            if inv_st is not None:
                target_conditions.append(
                    {
                        "target_specie": sp,
                        "metric": "st",
                        "value": float(inv_st),
                        "label": f"{sp}:storage>={float(inv_st):.6e}",
                    }
                )

    # OFC compartment is active only if tau_ofc is finite and positive; otherwise IFC-only routing is used.
    tau_ofc_map: Dict[str, float] = {}
    ofc_enabled: Dict[str, bool] = {}
    for sp in SPECIES:
        tau_ofc_sp = float(species_params[sp].get("tau_ofc", np.inf))
        tau_ofc_map[sp] = tau_ofc_sp
        ofc_enabled[sp] = species_enabled[sp] and np.isfinite(tau_ofc_sp) and (tau_ofc_sp > 0.0)

    f_local = {sp: float(f0.get(sp, 0.0)) for sp in SPECIES}
    for sp in SPECIES:
        if not species_enabled[sp]:
            f_local[sp] = 0.0
    f_sum = sum(f_local.values())
    if f_sum <= 0.0:
        raise ValueError("Initial fractions for enabled species must have a positive sum")
    if not np.isclose(f_sum, 1.0, atol=1e-12, rtol=0.0):
        for sp in SPECIES:
            f_local[sp] = f_local[sp] / f_sum

    if reactivities is None:
        _, sigmav_DD_n, sigmav_DD_p = sigmav_DD_BoschHale(float(T_i))
        sigmav_DT = sigmav_DT_BoschHale(float(T_i))
        sigmav_DHe3 = sigmav_DHe3_BoschHale(float(T_i))
        sigmav_TT = sigmav_TT_placeholder(float(T_i))
        sigmav_He3He3 = sigmav_He3He3_placeholder(float(T_i))
        sigmav_THe3_ch1, sigmav_THe3_ch2, sigmav_THe3_ch3 = sigmav_THe3_placeholder(float(T_i))
    else:
        sigmav_DD_p = float(reactivities.get("sigmav_DD_p", 0.0))
        sigmav_DD_n = float(reactivities.get("sigmav_DD_n", 0.0))
        sigmav_DT = float(reactivities.get("sigmav_DT", 0.0))
        sigmav_DHe3 = float(reactivities.get("sigmav_DHe3", 0.0))
        sigmav_TT = float(reactivities.get("sigmav_TT", 0.0))
        sigmav_He3He3 = float(reactivities.get("sigmav_He3He3", 0.0))
        sigmav_THe3_ch1 = float(reactivities.get("sigmav_THe3_ch1", 0.0))
        sigmav_THe3_ch2 = float(reactivities.get("sigmav_THe3_ch2", 0.0))
        sigmav_THe3_ch3 = float(reactivities.get("sigmav_THe3_ch3", 0.0))

    idx = _build_index(species_enabled, ofc_enabled)
    idx_ofc = {sp: idx.get((sp, "ofc"), -1) for sp in SPECIES}
    idx_ifc = {sp: idx.get((sp, "ifc"), -1) for sp in SPECIES}
    idx_st = {sp: idx.get((sp, "st"), -1) for sp in SPECIES}
    idx_n = {sp: idx.get((sp, "n"), -1) for sp in SPECIES}

    active_inventory_idx = np.array(
        [idx[(sp, comp)] for sp in SPECIES for comp in ("ofc", "ifc", "st") if (sp, comp) in idx],
        dtype=np.int64,
    )
    active_density_idx = np.array([idx_n[sp] for sp in SPECIES if idx_n[sp] >= 0], dtype=np.int64)

    n_state = len(idx)

    y0 = np.zeros(n_state, dtype=float)
    for sp in SPECIES:
        if species_enabled[sp]:
            if idx_ofc[sp] >= 0:
                y0[idx_ofc[sp]] = 100.0
            if idx_ifc[sp] >= 0:
                y0[idx_ifc[sp]] = 100.0
            if idx_st[sp] >= 0:
                y0[idx_st[sp]] = 100.0
            if idx_n[sp] >= 0:
                y0[idx_n[sp]] = n_tot * f_local[sp]

    valid_modes = {"auto", "direct", "off"}
    mode_to_code = {"off": _MODE_OFF, "direct": _MODE_DIRECT, "auto": _MODE_AUTO}
    for sp in SPECIES:
        mode = injection_mode.get(sp, "off")
        if mode not in valid_modes:
            raise ValueError(f"Invalid injection_mode for {sp}: {mode}")

    # Backward-compatible flag kept in API/config; auto injection is always
    # independent from storage in this solver.
    _ = auto_injection_use_storage_limits

    idx_ofc_arr = np.array([idx_ofc[sp] for sp in SPECIES], dtype=np.int64)
    idx_ifc_arr = np.array([idx_ifc[sp] for sp in SPECIES], dtype=np.int64)
    idx_st_arr = np.array([idx_st[sp] for sp in SPECIES], dtype=np.int64)
    idx_n_arr = np.array([idx_n[sp] for sp in SPECIES], dtype=np.int64)

    species_enabled_arr = np.array([species_enabled[sp] for sp in SPECIES], dtype=np.bool_)
    ofc_enabled_arr = np.array([ofc_enabled[sp] for sp in SPECIES], dtype=np.bool_)
    mode_codes = np.array([mode_to_code[injection_mode.get(sp, "off")] for sp in SPECIES], dtype=np.int64)

    tau_p_arr = np.array([float(species_params[sp]["tau_p"]) for sp in SPECIES], dtype=np.float64)
    tau_ifc_arr = np.array([float(species_params[sp]["tau_ifc"]) for sp in SPECIES], dtype=np.float64)
    tau_ofc_arr = np.array([tau_ofc_map[sp] for sp in SPECIES], dtype=np.float64)
    lambda_arr = np.array([float(species_params[sp]["lambda_decay"]) for sp in SPECIES], dtype=np.float64)
    N_st_min_arr = np.array([float(species_params[sp]["N_st_min"]) for sp in SPECIES], dtype=np.float64)
    Ndot_max_arr = np.array([float(species_params[sp]["Ndot_max"]) for sp in SPECIES], dtype=np.float64)
    auto_weights_arr = np.array([float(automatic_injection_weights.get(sp, 1.0)) for sp in SPECIES], dtype=np.float64)

    def _ode_system(t: float, y: np.ndarray) -> np.ndarray:
        dydt, _, _, _, _ = _compute_rhs_and_control_numba(
            y,
            idx_ofc_arr,
            idx_ifc_arr,
            idx_st_arr,
            idx_n_arr,
            species_enabled_arr,
            ofc_enabled_arr,
            mode_codes,
            tau_p_arr,
            tau_ifc_arr,
            tau_ofc_arr,
            lambda_arr,
            N_st_min_arr,
            Ndot_max_arr,
            auto_weights_arr,
            float(V_plasma),
            float(TBR_DT),
            float(TBR_DDn),
            float(sigmav_DD_p),
            float(sigmav_DD_n),
            float(sigmav_DT),
            float(sigmav_DHe3),
            float(sigmav_TT),
            float(sigmav_He3He3),
            float(sigmav_THe3_ch1),
            float(sigmav_THe3_ch2),
            float(sigmav_THe3_ch3),
            bool(route_THe3_ch3_to_He4),
            bool(enforce_constant_total_density),
            bool(allow_negative_auto_injection),
        )
        return dydt

    def _negative_event(t: float, y: np.ndarray) -> float:
        inv_min = np.min(y[active_inventory_idx] + 100.0) if active_inventory_idx.size else np.inf
        den_min = np.min(y[active_density_idx] + 1e5) if active_density_idx.size else np.inf
        return float(min(inv_min, den_min))

    _negative_event.terminal = True
    _negative_event.direction = -1

    target_event_functions = []
    for cond in target_conditions:
        sp = str(cond["target_specie"])
        metric = str(cond["metric"])
        value = float(cond["value"])

        if metric == "fraction":
            def _event(_, y, sp=sp, value=value):
                if idx_n[sp] < 0:
                    return -value
                return float(y[idx_n[sp]]) / n_tot - value
        elif metric == "ifc":
            def _event(_, y, sp=sp, value=value):
                if idx_ifc[sp] < 0:
                    return -value
                return float(y[idx_ifc[sp]]) - value
        elif metric == "ofc":
            def _event(_, y, sp=sp, value=value):
                if idx_ofc[sp] < 0:
                    return -value
                return float(y[idx_ofc[sp]]) - value
        elif metric == "st":
            def _event(_, y, sp=sp, value=value):
                if idx_st[sp] < 0:
                    return -value
                return float(y[idx_st[sp]]) - value
        else:
            raise ValueError(f"Unknown target metric: {metric}")

        _event.terminal = True
        _event.direction = 1
        target_event_functions.append(_event)

    events_to_solve = target_event_functions + [_negative_event]

    try:
        sol = solve_ivp(
            fun=_ode_system,
            t_span=(0.0, max_simulation_time),
            y0=y0,
            method=solver_method,
            dense_output=False,
            events=events_to_solve,
            rtol=solver_rtol,
            atol=solver_atol,
        )
    except Exception as e:
        return {
            "t_startup": np.inf,
            "sol_success": False,
            "error": f"Unexpected solver exception: {e}",
        }

    error: Any = None
    if not sol.success:
        error = f"ODE solver failed: {sol.message}"

    n_target_events = len(target_event_functions)
    negative_event_idx = n_target_events
    if len(sol.t_events) > negative_event_idx and sol.t_events[negative_event_idx].size > 0:
        t_fail = float(sol.t_events[negative_event_idx][0])
        error = f"Negative state event triggered at t={t_fail:.6e} s"

    target_hit = False
    t_startup = np.inf
    y_event = None
    if n_target_events > 0:
        for iev in range(n_target_events):
            if sol.t_events[iev].size > 0:
                t_hit = float(sol.t_events[iev][0])
                if t_hit < t_startup:
                    t_startup = t_hit
                    y_event = sol.y_events[iev][0]
                    target_hit = True

    if n_target_events > 0:
        sol_success = target_hit and error is None
        if (not target_hit) and (error is None):
            error = "No target condition reached within max_simulation_time"
    else:
        sol_success = bool(sol.success) and error is None

    # Interpolate to uniform timeline
    if (y_event is not None) and np.isfinite(t_startup):
        t_with_event = np.append(sol.t, t_startup)
        Y_with_event = np.column_stack([sol.y, y_event])

        sort_idx = np.argsort(t_with_event)
        t_sorted = t_with_event[sort_idx]
        Y_sorted = Y_with_event[:, sort_idx]

        t_array = np.linspace(0.0, t_startup, max(2, int(vector_length)))
        Y_interp = np.vstack([np.interp(t_array, t_sorted, Y_sorted[i, :]) for i in range(Y_sorted.shape[0])])
    else:
        if sol.t.size >= 2:
            t_end = float(sol.t[-1])
            t_array = np.linspace(0.0, max(t_end, 1e-9), max(2, int(vector_length)))
            Y_interp = np.vstack([np.interp(t_array, sol.t, sol.y[i, :]) for i in range(sol.y.shape[0])])
        elif sol.t.size == 1:
            t_array = np.linspace(0.0, max(float(sol.t[0]), 1e-9), max(2, int(vector_length)))
            Y_interp = np.vstack([np.full_like(t_array, sol.y[i, 0], dtype=float) for i in range(sol.y.shape[0])])
        else:
            t_array = np.linspace(0.0, 1.0, max(2, int(vector_length)))
            Y_interp = np.vstack([np.full_like(t_array, y0[i], dtype=float) for i in range(y0.size)])

    data: Dict[Tuple[str, str], np.ndarray] = {}
    for sp in SPECIES:
        for comp in ("ofc", "ifc", "st", "n"):
            pos = idx.get((sp, comp), None)
            if pos is None:
                data[(sp, comp)] = np.zeros_like(t_array)
            else:
                data[(sp, comp)] = Y_interp[pos, :]

    Ndot_history = {sp: np.zeros_like(t_array) for sp in SPECIES}
    sum_dn_dt_history = np.zeros_like(t_array)
    required_auto_total_history = np.full_like(t_array, np.nan, dtype=float)
    unmet_auto_total_history = np.full_like(t_array, np.nan, dtype=float)
    sum_abs_dn_terms_history = np.zeros_like(t_array)
    n_total_history = np.zeros_like(t_array)

    for i in range(len(t_array)):
        y_i = Y_interp[:, i]

        dydt_i, Ndot_i, required_i, unmet_i, sum_dn_i = _compute_rhs_and_control_numba(
            y_i,
            idx_ofc_arr,
            idx_ifc_arr,
            idx_st_arr,
            idx_n_arr,
            species_enabled_arr,
            ofc_enabled_arr,
            mode_codes,
            tau_p_arr,
            tau_ifc_arr,
            tau_ofc_arr,
            lambda_arr,
            N_st_min_arr,
            Ndot_max_arr,
            auto_weights_arr,
            float(V_plasma),
            float(TBR_DT),
            float(TBR_DDn),
            float(sigmav_DD_p),
            float(sigmav_DD_n),
            float(sigmav_DT),
            float(sigmav_DHe3),
            float(sigmav_TT),
            float(sigmav_He3He3),
            float(sigmav_THe3_ch1),
            float(sigmav_THe3_ch2),
            float(sigmav_THe3_ch3),
            bool(route_THe3_ch3_to_He4),
            bool(enforce_constant_total_density),
            bool(allow_negative_auto_injection),
        )

        for isp, sp in enumerate(SPECIES):
            Ndot_history[sp][i] = Ndot_i[isp]

        sum_abs_terms_i = 0.0
        n_total_i = 0.0
        for isp, sp in enumerate(SPECIES):
            if (not species_enabled[sp]) or idx_n_arr[isp] < 0:
                continue
            n_sp_i = float(y_i[idx_n_arr[isp]])
            dn_sp_dt_i = float(dydt_i[idx_n_arr[isp]])
            sum_abs_terms_i += abs(dn_sp_dt_i)
            n_total_i += n_sp_i

        sum_abs_dn_terms_history[i] = sum_abs_terms_i
        n_total_history[i] = n_total_i

        sum_dn_dt_history[i] = sum_dn_i
        required_auto_total_history[i] = required_i
        unmet_auto_total_history[i] = unmet_i

    n_total_drift_history = n_total_history - n_total_history[0]
    if n_total_history[0] != 0.0:
        n_total_rel_drift_history = n_total_drift_history / n_total_history[0]
    else:
        n_total_rel_drift_history = np.full_like(n_total_drift_history, np.nan)

    eps_res = np.finfo(float).eps
    normalized_residual_history = sum_dn_dt_history / np.maximum(sum_abs_dn_terms_history, eps_res)

    result: Dict[str, Any] = {
        "t": t_array,
        "t_startup": t_startup,
        "sol_success": bool(sol_success),
        "error": None if sol_success else error,
        "sum_dn_dt": sum_dn_dt_history,
        "required_auto_total": required_auto_total_history,
        "unmet_auto_total": unmet_auto_total_history,
        "normalized_residual": normalized_residual_history,
        "n_total_rel_drift": n_total_rel_drift_history,
    }

    for sp in SPECIES:
        result[f"n_{sp}"] = data[(sp, "n")]
        result[f"N_ofc_{sp}"] = data[(sp, "ofc")]
        result[f"N_ifc_{sp}"] = data[(sp, "ifc")]
        result[f"N_st_{sp}"] = data[(sp, "st")]
        result[f"Ndot_{sp}"] = Ndot_history[sp]

    return result
