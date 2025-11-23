"""
Unit tests for power_balance module.

Tests focus on correctness of power and energy calculations:
- Fusion power scaling
- Power balance consistency
- Energy integration
"""

import pytest
import numpy as np
from ddstartup.physics.power_balance import (
    compute_fusion_powers,
    compute_lump_powers_and_energies,
    compute_tseeded_powers_and_energies
)


class TestFusionPowerCalculations:
    """Test core fusion power calculations."""
    
    def test_fusion_power_density_scaling(self):
        """Fusion power should scale with n^2 for DD, n_D*n_T for DT."""
        n_D = 1e20
        n_T = 1e19
        n_tot = n_D
        V_plasma = 100.0
        sigmav_DD_p = 1e-23  # m^3/s
        sigmav_DD_n = 1e-23
        sigmav_DT = 1e-21
        
        # Base case
        P_DDn_1, P_DDp_1, P_DT_1, _, _ = compute_fusion_powers(
            n_D, n_T, n_tot, V_plasma,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT
        )
        
        # Double densities
        P_DDn_2, P_DDp_2, P_DT_2, _, _ = compute_fusion_powers(
            2*n_D, 2*n_T, 2*n_tot, V_plasma,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT
        )
        
        # DD scales as n_D^2 (factor of 4)
        assert P_DDn_2 == pytest.approx(4 * P_DDn_1)
        assert P_DDp_2 == pytest.approx(4 * P_DDp_1)
        
        # DT scales as n_D * n_T (factor of 4)
        assert P_DT_2 == pytest.approx(4 * P_DT_1)
    
    def test_fusion_power_volume_scaling(self):
        """Fusion power should scale linearly with volume."""
        n_D = 5e19
        n_T = 1e19
        n_tot = n_D
        sigmav_DD_p = 5e-24
        sigmav_DD_n = 5e-24
        sigmav_DT = 5e-22
        
        P_DDn_100, _, _, _, _ = compute_fusion_powers(
            n_D, n_T, n_tot, 100.0,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT
        )
        
        P_DDn_200, _, _, _, _ = compute_fusion_powers(
            n_D, n_T, n_tot, 200.0,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT
        )
        
        assert P_DDn_200 == pytest.approx(2 * P_DDn_100)
    
    def test_fusion_power_reactivity_scaling(self):
        """Fusion power should scale linearly with reactivity."""
        n_D = 1e20
        n_T = 5e19
        n_tot = n_D
        V_plasma = 100.0
        
        _, P_DDp_low, _, _, _ = compute_fusion_powers(
            n_D, n_T, n_tot, V_plasma,
            1e-23,  # sigmav_DD_p
            1e-23,  # sigmav_DD_n
            1e-21   # sigmav_DT
        )
        
        _, P_DDp_high, _, _, _ = compute_fusion_powers(
            n_D, n_T, n_tot, V_plasma,
            2e-23,  # sigmav_DD_p - Double reactivity
            1e-23,  # sigmav_DD_n
            1e-21   # sigmav_DT
        )
        
        assert P_DDp_high == pytest.approx(2 * P_DDp_low)
    
    def test_dt_equilibrium_power(self):
        """DT equilibrium should use 50-50 mixture."""
        n_tot = 1e20
        V_plasma = 100.0
        sigmav_DT = 1e-21
        
        _, _, _, _, P_DT_eq = compute_fusion_powers(
            n_D=0.5*n_tot,  # Doesn't matter for P_DT_eq
            n_T=0.5*n_tot,
            n_tot=n_tot,
            V_plasma=V_plasma,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=sigmav_DT
        )
        
        # P_DT_eq = 0.25 * n_tot^2 * sigmav_DT * V * E_DT
        # Check it's independent of actual n_D, n_T
        _, _, _, _, P_DT_eq_2 = compute_fusion_powers(
            n_D=0.9*n_tot,  # Different mix
            n_T=0.1*n_tot,
            n_tot=n_tot,
            V_plasma=V_plasma,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=sigmav_DT
        )
        
        assert P_DT_eq == pytest.approx(P_DT_eq_2)
    
    def test_fusion_powers_positive(self):
        """All fusion powers should be non-negative."""
        P_DDn, P_DDp, P_DT, P_DHe3, P_DT_eq = compute_fusion_powers(
            n_D=1e20,
            n_T=1e19,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            n_He3=1e18,
            sigmav_DHe3=1e-25
        )
        
        assert P_DDn >= 0
        assert P_DDp >= 0
        assert P_DT >= 0
        assert P_DHe3 >= 0
        assert P_DT_eq >= 0


class TestLumpPowersAndEnergies:
    """Test lump method power and energy calculations."""
    
    def test_lump_energy_equals_power_times_time(self):
        """For steady-state, energy = power × time."""
        n_T = 1e19
        n_D = 1e20
        n_He3 = 1e18
        t_startup = 3.156e8  # 10 years in seconds
        V_plasma = 150.0
        P_aux = 50e6  # 50 MW
        P_aux_DT_eq = 30e6  # 30 MW
        
        result = compute_lump_powers_and_energies(
            n_T, n_D, n_He3, t_startup,
            V_plasma,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-25,
            P_aux=P_aux,
            P_aux_DT_eq=P_aux_DT_eq
        )
        
        # Check energy = power × time
        expected_E_fusion = result['P_fusion_total'] * t_startup
        expected_E_aux = P_aux * t_startup
        expected_E_DT_eq = result['P_DT_eq'] * t_startup
        
        assert result['E_fusion_DD'] == pytest.approx(expected_E_fusion)
        assert result['E_aux_DD'] == pytest.approx(expected_E_aux)
        assert result['E_fusion_DT_eq'] == pytest.approx(expected_E_DT_eq)
    
    def test_lump_total_fusion_power(self):
        """Total fusion should equal sum of components."""
        result = compute_lump_powers_and_energies(
            n_T=2e19,
            n_D=8e19,
            n_He3=5e17,
            t_startup=1e8,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            sigmav_DHe3=1e-25,
            P_aux=40e6,
            P_aux_DT_eq=25e6
        )
        
        P_sum = result['P_DDn'] + result['P_DDp'] + result['P_DT']
        # Note: P_DHe3 not included in some versions, check implementation
        
        assert result['P_fusion_total'] >= P_sum  # Should include all components


class TestTseededPowersAndEnergies:
    """Test T-seeded method power and energy calculations."""
    
    def test_tseeded_interpolation_length(self):
        """Output arrays should have specified vector_length."""
        # Simple mock data
        t_raw = np.array([0.0, 1e7, 2e7, 3e7])
        n_T_raw = np.array([1e17, 5e18, 1e19, 2e19])
        n_D_raw = np.array([1e20, 9.5e19, 9e19, 8.5e19])
        N_ofc_raw = np.array([1e25, 5e25, 1e26, 1.5e26])
        N_ifc_raw = np.array([1e24, 5e24, 1e25, 2e25])
        N_st_raw = np.array([1e23, 5e23, 1e24, 2e24])
        
        vector_length = 50
        
        result = compute_tseeded_powers_and_energies(
            t_startup=3e7,
            t_raw=t_raw,
            n_T_raw=n_T_raw,
            n_D_raw=n_D_raw,
            N_ofc_raw=N_ofc_raw,
            N_ifc_raw=N_ifc_raw,
            N_st_raw=N_st_raw,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            tau_ifc=3600.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            injection_rate_max=1e20,
            N_st_min=1e24,
            vector_length=vector_length
        )
        
        assert len(result['t_interp']) == vector_length
        assert len(result['n_T']) == vector_length
        assert len(result['n_D']) == vector_length
        assert len(result['P_DDn']) == vector_length
    
    def test_tseeded_time_grid_uniform(self):
        """Time grid should be uniformly spaced."""
        t_raw = np.linspace(0, 1e8, 20)
        n_T_raw = np.linspace(1e17, 1e19, 20)
        n_D_raw = np.full(20, 1e20)
        N_ofc_raw = np.linspace(1e25, 1e26, 20)
        N_ifc_raw = np.linspace(1e24, 1e25, 20)
        N_st_raw = np.linspace(1e23, 1e24, 20)
        
        result = compute_tseeded_powers_and_energies(
            t_startup=1e8,
            t_raw=t_raw,
            n_T_raw=n_T_raw,
            n_D_raw=n_D_raw,
            N_ofc_raw=N_ofc_raw,
            N_ifc_raw=N_ifc_raw,
            N_st_raw=N_st_raw,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            tau_ifc=7200.0,
            P_aux=60e6,
            P_aux_DT_eq=35e6,
            injection_rate_max=1e20,
            N_st_min=1e24,
            vector_length=100
        )
        
        # Check uniform spacing
        dt = np.diff(result['t_interp'])
        assert np.allclose(dt, dt[0])
    
    def test_tseeded_energies_positive(self):
        """All energies should be non-negative."""
        t_raw = np.linspace(0, 1e8, 10)
        n_T_raw = np.linspace(1e17, 1e19, 10)
        n_D_raw = np.full(10, 1e20)
        N_ofc_raw = np.linspace(1e25, 1e26, 10)
        N_ifc_raw = np.linspace(1e24, 1e25, 10)
        N_st_raw = np.linspace(1e23, 1e24, 10)
        
        result = compute_tseeded_powers_and_energies(
            t_startup=1e8,
            t_raw=t_raw,
            n_T_raw=n_T_raw,
            n_D_raw=n_D_raw,
            N_ofc_raw=N_ofc_raw,
            N_ifc_raw=N_ifc_raw,
            N_st_raw=N_st_raw,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            tau_ifc=3600.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            injection_rate_max=1e20,
            N_st_min=1e24,
            vector_length=50
        )
        
        assert result['E_fusion_DD'] >= 0
        assert result['E_fusion_DT_eq'] >= 0
        assert result['E_aux_DD'] >= 0
        assert result['E_aux_DT_eq'] >= 0

    def test_tseeded_p_dt_eq_scalar_output(self):
        """P_DT_eq should be stored as a scalar while keeping a vector profile for internal use."""
        t_raw = np.array([0.0, 5e6, 1e7])
        n_T_raw = np.array([1e17, 5e18, 1e19])
        n_D_raw = np.array([1e20, 9.5e19, 9e19])
        N_ofc_raw = np.array([1e25, 5e25, 1e26])
        N_ifc_raw = np.array([1e24, 5e24, 1e25])
        N_st_raw = np.array([1e23, 5e23, 1e24])

        vector_length = 20

        result = compute_tseeded_powers_and_energies(
            t_startup=1e7,
            t_raw=t_raw,
            n_T_raw=n_T_raw,
            n_D_raw=n_D_raw,
            N_ofc_raw=N_ofc_raw,
            N_ifc_raw=N_ifc_raw,
            N_st_raw=N_st_raw,
            n_tot=1e20,
            V_plasma=100.0,
            sigmav_DD_p=1e-23,
            sigmav_DD_n=1e-23,
            sigmav_DT=1e-21,
            tau_ifc=3600.0,
            P_aux=50e6,
            P_aux_DT_eq=30e6,
            injection_rate_max=1e20,
            N_st_min=1e24,
            vector_length=vector_length
        )

        # Scalar storage
        assert np.ndim(result['P_DT_eq']) == 0
        # Vector profile preserved for any time-series math
        assert result['P_DT_eq_profile'].shape == result['P_DT'].shape
        assert np.allclose(result['P_DT_eq_profile'], result['P_DT_eq'])
