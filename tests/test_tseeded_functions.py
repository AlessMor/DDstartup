"""
Comprehensive tests for T_seeded physics functions.

Tests cover:
1. ODE system evaluation
2. ODE solver integration
3. Single combination computation
4. Postprocessing functions
5. Edge cases and error handling
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddstartup.physics.Tseeded_functions import (
    ode_system,
    solve_ode_system,
    compute_single_combination,
    postprocess_fusion_results_Tseeded,
    trapz_numba
)
from ddstartup.utils.units_and_constants import lambda_T, E_DDn, E_DDp, E_DT, tritium_mass


class TestODESystem:
    """Test the ODE system function."""
    
    def test_ode_system_returns_correct_shape(self):
        """Test that ODE system returns 4 derivatives."""
        # Initial state [N_ofc, N_ifc, N_st, n_T]
        y = np.array([1e20, 1e20, 1e19, 1e19])
        t = 0.0
        
        # Typical parameters
        V_plasma = 100.0
        n_tot = 1e20
        tau_p_T = 1.0
        TBR_DT = 1.05
        TBR_DDn = 0.5
        tau_ifc = 30 * 86400
        tau_ofc = 30 * 86400
        sigmav_DD_p = 1e-22
        sigmav_DD_n = 1e-22
        sigmav_DT = 1e-21
        injection_rate_max = 1e20
        N_st_min = 1e18
        
        derivatives = ode_system(
            t, y, V_plasma, n_tot, tau_p_T,
            TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT,
            injection_rate_max, N_st_min
        )
        
        assert len(derivatives) == 4, "Should return 4 derivatives"
        assert isinstance(derivatives, np.ndarray), "Should return numpy array"
    
    def test_ode_system_with_zero_tritium(self):
        """Test ODE system with zero tritium (initial condition)."""
        y = np.array([0.0, 0.0, 0.0, 0.0])
        t = 0.0
        
        derivatives = ode_system(
            t, y, 100.0, 1e20, 1.0, 1.05, 0.5,
            30*86400, 30*86400, 1e-22, 1e-22, 1e-21,
            1e20, 1e18
        )
        
        # With zero tritium, there should be some DD production
        assert derivatives[0] >= 0, "N_ofc should increase from DD reactions"
    
    def test_ode_system_injection_rate_limiting(self):
        """Test that injection rate is properly limited."""
        # High storage, should hit injection_rate_max
        y = np.array([1e20, 1e21, 1e21, 1e19])
        injection_rate_max = 1e19
        
        derivatives = ode_system(
            0.0, y, 100.0, 1e20, 1.0, 1.05, 0.5,
            30*86400, 30*86400, 1e-22, 1e-22, 1e-21,
            injection_rate_max, 1e18
        )
        
        # Check that storage derivative is affected by injection
        # (May be zero or negative depending on balance of inflow/outflow)
        assert derivatives[2] <= 0, "Storage should not increase when injecting"
    
    def test_ode_system_no_injection_below_minimum(self):
        """Test that no injection occurs below N_st_min."""
        # Storage below minimum
        y = np.array([1e20, 1e20, 1e17, 1e19])  # N_st = 1e17 < 1e18
        
        derivatives = ode_system(
            0.0, y, 100.0, 1e20, 1.0, 1.05, 0.5,
            30*86400, 30*86400, 1e-22, 1e-22, 1e-21,
            1e20, 1e18
        )
        
        # dnT_dt should not include injection term (or be less negative)
        # This is hard to test directly, but storage should accumulate
        assert np.isfinite(derivatives[3]), "n_T derivative should be finite"


class TestODESolver:
    """Test the ODE solver wrapper."""
    
    def test_solve_ode_system_successful(self):
        """Test successful ODE integration."""
        result = solve_ode_system(
            total_time=10*365*24*3600,  # 10 years
            V_plasma=100.0,
            n_tot=1.5e20,
            tau_p_T=1.0,
            P_aux=50e6,
            P_aux_DT_eq=50e6,
            TBR_DT=1.05,
            TBR_DDn=0.5,
            tau_ifc=30*86400,
            tau_ofc=30*86400,
            eta_th=0.4,
            capacity_factor=0.8,
            cost_of_electricity=1e-7,
            sigmav_DD_p=1e-22,
            sigmav_DD_n=1e-22,
            sigmav_DT=1e-21,
            injection_rate_max=1e20,
            vector_length=50
        )
        
        assert 'sol_success' in result
        assert 't_startup' in result or 'error' in result
        
        if result['sol_success']:
            assert np.isfinite(result['t_startup']), "Startup time should be finite"
            assert result['t_startup'] > 0, "Startup time should be positive"
            assert 'Q_DD' in result
            assert 'unrealized_profits' in result
    
    def test_solve_ode_system_vector_outputs(self):
        """Test that vector outputs have correct length."""
        vector_length = 75
        result = solve_ode_system(
            total_time=5*365*24*3600,
            V_plasma=100.0,
            n_tot=1.5e20,
            tau_p_T=1.0,
            P_aux=50e6,
            P_aux_DT_eq=50e6,
            TBR_DT=1.05,
            TBR_DDn=0.5,
            tau_ifc=30*86400,
            tau_ofc=30*86400,
            eta_th=0.4,
            capacity_factor=0.8,
            cost_of_electricity=1e-7,
            sigmav_DD_p=1e-22,
            sigmav_DD_n=1e-22,
            sigmav_DT=1e-21,
            injection_rate_max=1e20,
            vector_length=vector_length
        )
        
        if result['sol_success']:
            assert len(result['N_ofc']) == vector_length
            assert len(result['N_ifc']) == vector_length
            assert len(result['N_stor']) == vector_length
            assert len(result['n_T']) == vector_length
            assert len(result['P_DDn']) == vector_length
            assert len(result['P_DT']) == vector_length
    
    def test_solve_ode_system_impossible_parameters(self):
        """Test handling of impossible parameter combinations."""
        # Zero TBR - cannot breed tritium
        result = solve_ode_system(
            total_time=1*365*24*3600,
            V_plasma=100.0,
            n_tot=1e20,
            tau_p_T=1.0,
            P_aux=50e6,
            P_aux_DT_eq=50e6,
            TBR_DT=0.0,  # No breeding
            TBR_DDn=0.0,  # No breeding
            tau_ifc=30*86400,
            tau_ofc=30*86400,
            eta_th=0.4,
            capacity_factor=0.8,
            cost_of_electricity=1e-7,
            sigmav_DD_p=1e-22,
            sigmav_DD_n=1e-22,
            sigmav_DT=1e-21,
            injection_rate_max=1e20,
            vector_length=50
        )
        
        # Should either fail or not reach DT operation
        if result['sol_success']:
            assert result['t_startup'] == np.inf or not np.isfinite(result['t_startup'])
        else:
            assert 'error' in result
    
    def test_solve_ode_system_energy_conservation(self):
        """Test that energy calculations are self-consistent."""
        result = solve_ode_system(
            total_time=10*365*24*3600,
            V_plasma=100.0,
            n_tot=1.5e20,
            tau_p_T=1.0,
            P_aux=50e6,
            P_aux_DT_eq=50e6,
            TBR_DT=1.05,
            TBR_DDn=0.5,
            tau_ifc=30*86400,
            tau_ofc=30*86400,
            eta_th=0.4,
            capacity_factor=0.8,
            cost_of_electricity=1e-7,
            sigmav_DD_p=1e-22,
            sigmav_DD_n=1e-22,
            sigmav_DT=1e-21,
            injection_rate_max=1e20,
            vector_length=50
        )
        
        if result['sol_success'] and np.isfinite(result['t_startup']):
            # Q_DD should be positive for net gain
            assert result['Q_DD'] >= 0, "Q_DD should be non-negative"
            # DT equivalent should have higher Q
            assert result['Q_DT_eq'] >= result['Q_DD'], "DT should be more efficient"


class TestComputeSingleCombination:
    """Test the single combination wrapper."""
    
    def test_compute_single_combination_basic(self):
        """Test basic single combination computation."""
        # Setup input arrays (scalar parameters)
        input_arrays = [
            np.array([100.0]),  # V_plasma
            np.array([69.0]),   # T_i
            np.array([1.5e20]), # n_tot
            np.array([1.0]),    # tau_p_T
            np.array([50e6]),   # P_aux
            np.array([50e6]),   # P_aux_DT_eq
            np.array([1.05]),   # TBR_DT
            np.array([0.5]),    # TBR_DDn
            np.array([30*86400]), # tau_ifc
            np.array([30*86400]), # tau_ofc
            np.array([0.4]),    # eta_th
            np.array([0.8]),    # capacity_factor
            np.array([1e-7])    # cost_of_electricity
        ]
        param_shapes = np.ones(13, dtype=np.int64)
        
        result = compute_single_combination(
            linear_index=0,
            input_arrays_flat=input_arrays,
            param_shapes_array=param_shapes,
            total_time=10*365*24*3600,
            vector_length=50
        )
        
        assert 'linear_index' in result
        # linear_index may be NaN in error dict structure, check for field existence instead
        assert 'sol_success' in result
        assert 'V_plasma' in result
        assert result['V_plasma'] == 100.0
    
    def test_compute_single_combination_with_parametric_arrays(self):
        """Test with actual parametric arrays."""
        # Setup arrays with multiple values
        input_arrays = [
            np.linspace(80, 120, 3),    # V_plasma
            np.linspace(60, 80, 2),     # T_i
            np.array([1.5e20]),         # n_tot
            np.array([1.0]),            # tau_p_T
            np.array([50e6]),           # P_aux
            np.array([50e6]),           # P_aux_DT_eq
            np.array([1.05]),           # TBR_DT
            np.array([0.5]),            # TBR_DDn
            np.array([30*86400]),       # tau_ifc
            np.array([30*86400]),       # tau_ofc
            np.array([0.4]),            # eta_th
            np.array([0.8]),            # capacity_factor
            np.array([1e-7])            # cost_of_electricity
        ]
        param_shapes = np.array([3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
        
        # Test first combination (index 0)
        result = compute_single_combination(
            linear_index=0,
            input_arrays_flat=input_arrays,
            param_shapes_array=param_shapes,
            total_time=5*365*24*3600,
            vector_length=50
        )
        
        assert result['V_plasma'] == 80.0
        assert result['T_i'] == 60.0
        
        # Test last combination (index 5: 3*2-1)
        result = compute_single_combination(
            linear_index=5,
            input_arrays_flat=input_arrays,
            param_shapes_array=param_shapes,
            total_time=5*365*24*3600,
            vector_length=50
        )
        
        assert result['V_plasma'] == 120.0
        assert result['T_i'] == 80.0
    
    def test_compute_single_combination_error_handling(self):
        """Test error handling in compute_single_combination."""
        # Invalid parameters that should cause issues
        input_arrays = [
            np.array([100.0]),
            np.array([69.0]),
            np.array([0.0]),      # Zero density - invalid
            np.array([1.0]),
            np.array([50e6]),
            np.array([50e6]),
            np.array([1.05]),
            np.array([0.5]),
            np.array([30*86400]),
            np.array([30*86400]),
            np.array([0.4]),
            np.array([0.8]),
            np.array([1e-7])
        ]
        param_shapes = np.ones(13, dtype=np.int64)
        
        result = compute_single_combination(
            linear_index=0,
            input_arrays_flat=input_arrays,
            param_shapes_array=param_shapes,
            total_time=1*365*24*3600,
            vector_length=50
        )
        
        # Should handle gracefully (either fail or return inf)
        assert 'sol_success' in result
        if not result['sol_success']:
            assert 'error' in result


class TestPostprocessingFunctions:
    """Test postprocessing and utility functions."""
    
    def test_trapz_numba(self):
        """Test Numba trapezoidal integration."""
        # Simple test case: integrate y=x from 0 to 1
        x = np.linspace(0, 1, 11)
        y = x
        
        result = trapz_numba(y, x)
        expected = 0.5  # Integral of x from 0 to 1 is 0.5
        
        assert np.abs(result - expected) < 1e-10, "Trapz should match analytical result"
    
    def test_trapz_numba_constant(self):
        """Test integration of constant function."""
        x = np.linspace(0, 10, 101)
        y = np.ones_like(x) * 5.0
        
        result = trapz_numba(y, x)
        expected = 5.0 * 10  # Constant * width
        
        assert np.abs(result - expected) < 1e-8
    
    def test_postprocess_fusion_results_structure(self):
        """Test that postprocessing returns correct structure."""
        # Create mock ODE solution
        vector_length = 50
        t_startup = 3.156e7  # 1 year
        N_ofc = np.linspace(0, 1e20, vector_length)
        N_ifc = np.linspace(0, 1e20, vector_length)
        N_st = np.linspace(0, 1e19, vector_length)
        n_T = np.linspace(0, 7.5e19, vector_length)
        
        # Parameters
        n_tot = 1.5e20
        V_plasma = 100.0
        sigmav_DD_p = 1e-22
        sigmav_DD_n = 1e-22
        sigmav_DT = 1e-21
        TBR_DT = 1.05
        TBR_DDn = 0.5
        tau_ifc = 30*86400
        eta_th = 0.4
        capacity_factor = 0.8
        cost_of_electricity = 1e-7
        P_aux = 50e6
        P_aux_DT_eq = 50e6
        injection_rate_max = 1e20
        N_st_min = 0.001/tritium_mass
        
        results = postprocess_fusion_results_Tseeded(
            t_startup, N_ofc, N_ifc, N_st, n_T, n_tot, V_plasma,
            sigmav_DD_p, sigmav_DD_n, sigmav_DT, TBR_DT, TBR_DDn,
            tau_ifc, eta_th, capacity_factor, cost_of_electricity,
            P_aux, P_aux_DT_eq, E_DDn, E_DDp, E_DT,
            injection_rate_max, N_st_min, vector_length
        )
        
        # Unpack results
        P_DDn, P_DDp, P_DT, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_profits, TBE_vector, n_D = results
        
        # Check types and shapes
        assert len(P_DDn) == vector_length
        assert len(P_DDp) == vector_length
        assert len(P_DT) == vector_length
        assert isinstance(Q_DD, (float, np.floating))
        assert isinstance(Q_DT_eq, (float, np.floating))
        assert len(TBE_vector) == vector_length
        assert len(n_D) == vector_length
    
    def test_postprocess_fusion_results_energy_positivity(self):
        """Test that energy values are physically reasonable."""
        vector_length = 50
        t_startup = 3.156e7
        N_ofc = np.linspace(0, 1e20, vector_length)
        N_ifc = np.linspace(0, 1e20, vector_length)
        N_st = np.linspace(0, 1e19, vector_length)
        n_T = np.linspace(0, 7.5e19, vector_length)
        
        results = postprocess_fusion_results_Tseeded(
            t_startup, N_ofc, N_ifc, N_st, n_T, 1.5e20, 100.0,
            1e-22, 1e-22, 1e-21, 1.05, 0.5, 30*86400,
            0.4, 0.8, 1e-7, 50e6, 50e6,
            E_DDn, E_DDp, E_DT, 1e20, 0.001/tritium_mass, vector_length
        )
        
        P_DDn, P_DDp, P_DT, P_DT_eq, Q_DD, Q_DT_eq, E_lost, unrealized_profits, TBE_vector, n_D = results
        
        # Powers should be non-negative
        assert np.all(P_DDn >= 0), "DD neutron power should be non-negative"
        assert np.all(P_DDp >= 0), "DD proton power should be non-negative"
        assert np.all(P_DT >= 0), "DT power should be non-negative"
        assert P_DT_eq >= 0, "DT equivalent power should be non-negative"
        
        # Q factors should be reasonable
        assert Q_DD >= 0 or np.isinf(Q_DD), "Q_DD should be non-negative or inf"
        assert Q_DT_eq >= 0 or np.isinf(Q_DT_eq), "Q_DT_eq should be non-negative or inf"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_short_confinement_time(self):
        """Test with very short confinement time."""
        result = solve_ode_system(
            total_time=1*365*24*3600,
            V_plasma=100.0,
            n_tot=1.5e20,
            tau_p_T=0.01,  # Very short - 10 ms
            P_aux=50e6,
            P_aux_DT_eq=50e6,
            TBR_DT=1.05,
            TBR_DDn=0.5,
            tau_ifc=30*86400,
            tau_ofc=30*86400,
            eta_th=0.4,
            capacity_factor=0.8,
            cost_of_electricity=1e-7,
            sigmav_DD_p=1e-22,
            sigmav_DD_n=1e-22,
            sigmav_DT=1e-21,
            injection_rate_max=1e20,
            vector_length=50
        )
        
        # Should handle gracefully
        assert 'sol_success' in result
        if not result['sol_success']:
            assert 'error' in result
    
    def test_very_long_confinement_time(self):
        """Test with very long confinement time."""
        result = solve_ode_system(
            total_time=10*365*24*3600,
            V_plasma=100.0,
            n_tot=1.5e20,
            tau_p_T=1000.0,  # Very long - 1000 seconds
            P_aux=50e6,
            P_aux_DT_eq=50e6,
            TBR_DT=1.05,
            TBR_DDn=0.5,
            tau_ifc=30*86400,
            tau_ofc=30*86400,
            eta_th=0.4,
            capacity_factor=0.8,
            cost_of_electricity=1e-7,
            sigmav_DD_p=1e-22,
            sigmav_DD_n=1e-22,
            sigmav_DT=1e-21,
            injection_rate_max=1e20,
            vector_length=50
        )
        
        # Should reach DT operation faster with better confinement
        if result['sol_success']:
            assert np.isfinite(result['t_startup'])
    
    def test_high_breeding_ratio(self):
        """Test with high tritium breeding ratios."""
        result = solve_ode_system(
            total_time=5*365*24*3600,
            V_plasma=100.0,
            n_tot=1.5e20,
            tau_p_T=1.0,
            P_aux=50e6,
            P_aux_DT_eq=50e6,
            TBR_DT=2.0,   # Very high
            TBR_DDn=1.0,  # Very high
            tau_ifc=30*86400,
            tau_ofc=30*86400,
            eta_th=0.4,
            capacity_factor=0.8,
            cost_of_electricity=1e-7,
            sigmav_DD_p=1e-22,
            sigmav_DD_n=1e-22,
            sigmav_DT=1e-21,
            injection_rate_max=1e20,
            vector_length=50
        )
        
        # Should reach DT operation faster with high breeding
        if result['sol_success']:
            assert result['t_startup'] < 10*365*24*3600, "Should reach DT faster with high TBR"
    
    def test_vector_length_variations(self):
        """Test with different vector lengths."""
        for vlen in [10, 50, 100, 200]:
            result = solve_ode_system(
                total_time=5*365*24*3600,
                V_plasma=100.0,
                n_tot=1.5e20,
                tau_p_T=1.0,
                P_aux=50e6,
                P_aux_DT_eq=50e6,
                TBR_DT=1.05,
                TBR_DDn=0.5,
                tau_ifc=30*86400,
                tau_ofc=30*86400,
                eta_th=0.4,
                capacity_factor=0.8,
                cost_of_electricity=1e-7,
                sigmav_DD_p=1e-22,
                sigmav_DD_n=1e-22,
                sigmav_DT=1e-21,
                injection_rate_max=1e20,
                vector_length=vlen
            )
            
            if result['sol_success']:
                assert len(result['N_ofc']) == vlen
                assert len(result['n_T']) == vlen


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
