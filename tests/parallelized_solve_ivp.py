import sys
sys.path.append("..")  # to import utils from parent directory
from utils import *
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
import itertools
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import psutil

# OPTIMIZATION: Try to import efficient storage formats
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


# OPTIMIZATION: Try to import Numba for JIT compilation (optional)
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
    print("Numba available - will use JIT compilation for 5-10x speedup")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - consider installing with 'pip install numba' for major speedup")
    
    # Define dummy decorators if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit
import time
import platform

# Convert constants to magnitude values for speed (no units in calculations)
tritium_mass = tritium_mass.to('kg').magnitude  # Ensure tritium_mass is in kg for calculations
lambda_T = 1.78e-9  # tritium decay constant in 1/s
E_DDn = 2.45  # MeV, energy from DD->n reaction
E_DDp = 4.0   # MeV, energy from DD->p reaction  
E_DT = 17.6   # MeV, energy from DT reaction

# Get number of points from command line argument or use default
points = int(sys.argv[1]) if len(sys.argv) > 1 else 3

V_plasma_field = ParameterField(parametrization_type="normal", mean=150, std=15, unit=u.m**3, 
    param_points=points, name="plasma_volume")
T_i_field = ParameterField(parametrization_type="linear", min_val=14, max_val=20, unit=u.keV,
    param_points=points, name="T_i_field")
n_tot_field = ParameterField(parametrization_type="linear", min_val=1.3e20, max_val=2.1e20, unit=u.m**(-3),
    param_points=points, name="n_tot_field")
tau_p_T_field = ParameterField(parametrization_type="linear", min_val = 0.1, max_val=5, unit=u.s,
    param_points=points, name="tau_p_T")
tau_p_He3_field = ParameterField(parametrization_type="normal", mean=1, std=0.5, unit=u.s,
    param_points=points, name="tau_p_He3")
P_aux_field = ParameterField(parametrization_type="linear", min_val=20, max_val=100, unit=u.MW,
    param_points=points, name="P_aux")
P_lost_rad_field = ParameterField(parametrization_type="linear", min_val=0, max_val=20, unit=u.MW,
    param_points=points, name="P_lost_rad")
P_aux_all_DT_field = ParameterField(parametrization_type="linear", min_val=20, max_val=100, unit=u.MW,
    param_points=points, name="P_aux_all_DT")
P_lost_rad_all_DT_field = ParameterField(parametrization_type="linear", min_val=0, max_val=20, unit=u.MW,
    param_points=points, name="P_lost_rad_all_DT")
TBR_DT_field = ParameterField(parametrization_type="linear", min_val=1.05, max_val=1.15,
    param_points=points, name="TBR_DT")
TBR_DDn_field = ParameterField(parametrization_type="linear", min_val=0.5, max_val=0.9,
    param_points=points, name="TBR_DDn")
tau_ifc_field = ParameterField(parametrization_type="linear", min_val=1, max_val=12, unit=u.h,
    param_points=points, name="tau_ifc")
tau_ofc_field = ParameterField(parametrization_type="linear", min_val=1, max_val=24, unit=u.h,
    param_points=points, name="tau_ofc")
eta_th_field = ParameterField(parametrization_type="linear", min_val=0.3, max_val=0.4,
    param_points=points, name="eta_th")
plant_avail_field = ParameterField(parametrization_type="linear", min_val=0.5, max_val=0.9,
    param_points=points, name="plant_availability")
Cost_per_kWh_field = ParameterField(parametrization_type="normal", mean=0.25, std=0.15, unit=1/u.kWh,
    param_points=points, name="Cost_per_kWh")

print("Plasma parameter fields:")
print(f"V_plasma: {V_plasma_field}")
print(f"T_i_field: {T_i_field}")
print(f"n_tot_field: {n_tot_field}")
print(f"tau_p_T: {tau_p_T_field}")
print(f"tau_p_He3: {tau_p_He3_field}")
print(f"P_aux: {P_aux_field}")
print(f"P_aux_all_DT: {P_aux_all_DT_field}")
print(f"P_lost_rad: {P_lost_rad_field}")
print(f"P_lost_rad_all_DT: {P_lost_rad_all_DT_field}")
print("Breeding parameters:")
print(f"TBR_DT: {TBR_DT_field}")
print(f"TBR_DDn: {TBR_DDn_field}")
print(f"tau_ifc: {tau_ifc_field}")
print(f"tau_ofc: {tau_ofc_field}")
print("Economic parameters:")
print(f"eta_th: {eta_th_field}")
print(f"plant_avail: {plant_avail_field}")
print(f"Cost_per_kWh: {Cost_per_kWh_field}")

input_data = [
    V_plasma_field.data.to('m**3').magnitude,
    T_i_field.data.to('keV').magnitude,
    n_tot_field.data.to('m**(-3)').magnitude,
    tau_p_T_field.data.to('s').magnitude,
    tau_p_He3_field.data.to('s').magnitude,
    P_aux_field.data.to('W').magnitude,
    P_lost_rad_field.data.to('W').magnitude,
    P_aux_all_DT_field.data.to('W').magnitude,
    P_lost_rad_all_DT_field.data.to('W').magnitude,

    TBR_DT_field.data.to('dimensionless').magnitude,
    TBR_DDn_field.data.to('dimensionless').magnitude,
    tau_ifc_field.data.to('s').magnitude,
    tau_ofc_field.data.to('s').magnitude,

    eta_th_field.data.to('dimensionless').magnitude,
    plant_avail_field.data.to('dimensionless').magnitude,
    Cost_per_kWh_field.data.to('1/kWh').magnitude,
]

#########################################
# SYSTEM PROFILER AND OPTIMIZER
##########################################

class SystemProfiler:
    """Automatically profile system capabilities and optimize computation settings"""
    
    def __init__(self):
        self.profile = self._profile_system()
        self.optimization = self._optimize_settings()
    
    def _profile_system(self):
        """Profile system hardware and capabilities"""
        profile = {}
        
        # CPU Information
        profile['cpu_count'] = psutil.cpu_count(logical=True)
        profile['cpu_count_physical'] = psutil.cpu_count(logical=False)
        profile['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else None
        
        # Memory Information
        memory = psutil.virtual_memory()
        profile['memory_total_gb'] = memory.total / (1024**3)
        profile['memory_available_gb'] = memory.available / (1024**3)
        profile['memory_usage_percent'] = memory.percent
        
        # Platform Information
        profile['platform'] = platform.system()
        profile['machine'] = platform.machine()
        profile['processor'] = platform.processor()
        
        # Performance estimation
        profile['performance_class'] = self._classify_performance(profile)
        
        return profile
    
    def _classify_performance(self, profile):
        """Classify system performance level"""
        score = 0
        
        # CPU scoring
        if profile['cpu_count'] >= 16:
            score += 4
        elif profile['cpu_count'] >= 8:
            score += 3
        elif profile['cpu_count'] >= 4:
            score += 2
        else:
            score += 1
        
        # Memory scoring
        if profile['memory_available_gb'] >= 16:
            score += 4
        elif profile['memory_available_gb'] >= 8:
            score += 3
        elif profile['memory_available_gb'] >= 4:
            score += 2
        else:
            score += 1
        
        # CPU frequency scoring (if available)
        if profile['cpu_freq']:
            if profile['cpu_freq'] >= 3000:
                score += 2
            elif profile['cpu_freq'] >= 2000:
                score += 1
        
        # Classify based on total score
        if score >= 8:
            return "high_performance"
        elif score >= 6:
            return "medium_performance" 
        elif score >= 4:
            return "low_performance"
        else:
            return "minimal_performance"
    
    def _optimize_settings(self):
        """Optimize computation settings based on system profile"""
        settings = {}
        perf_class = self.profile['performance_class']
        
        if perf_class == "high_performance":
            # High-end workstation/server
            settings['n_jobs'] = min(self.profile['cpu_count'] - 2, 16)  # Leave 2 cores for system
            settings['chunk_size'] = 5000
            settings['batch_size'] = 500
            settings['memory_safety_factor'] = 0.8  # Can use more memory
            settings['backend'] = 'loky'  # Can handle process-based parallelism
            settings['use_parallel_chunks'] = True
            
        elif perf_class == "medium_performance":
            # Standard desktop/laptop
            settings['n_jobs'] = min(self.profile['cpu_count'] // 2, 8)
            settings['chunk_size'] = 2000
            settings['batch_size'] = 200
            settings['memory_safety_factor'] = 0.6
            settings['backend'] = 'threading'
            settings['use_parallel_chunks'] = True
            
        elif perf_class == "low_performance":
            # Older laptop/limited resources
            settings['n_jobs'] = min(self.profile['cpu_count'] // 2, 4)
            settings['chunk_size'] = 1000
            settings['batch_size'] = 100
            settings['memory_safety_factor'] = 0.4
            settings['backend'] = 'threading'
            settings['use_parallel_chunks'] = False  # Sequential chunks
            
        else:
            # Very limited resources
            settings['n_jobs'] = 1
            settings['chunk_size'] = 500
            settings['batch_size'] = 50
            settings['memory_safety_factor'] = 0.3
            settings['backend'] = 'threading'
            settings['use_parallel_chunks'] = False
        
        # Memory-based adjustments
        available_mem = self.profile['memory_available_gb']
        if available_mem < 2:
            settings['chunk_size'] = min(settings['chunk_size'], 500)
            settings['n_jobs'] = min(settings['n_jobs'], 2)
        elif available_mem < 4:
            settings['chunk_size'] = min(settings['chunk_size'], 1000)
            settings['n_jobs'] = min(settings['n_jobs'], 4)
        
        return settings
    
    def benchmark_single_computation(self, param_combo, input_data, total_time):
        """Benchmark a single computation to estimate performance"""
        start_time = time.time()
        try:
            result = solve_single_combination(param_combo, input_data, total_time)
            end_time = time.time()
            return end_time - start_time, True
        except Exception as e:
            end_time = time.time()
            return end_time - start_time, False
    
    def print_system_info(self):
        """Print comprehensive system information"""
        print("=" * 60)
        print("SYSTEM PERFORMANCE PROFILE")
        print("=" * 60)
        print(f"Platform: {self.profile['platform']} ({self.profile['machine']})")
        print(f"CPU Cores: {self.profile['cpu_count']} logical, {self.profile['cpu_count_physical']} physical")
        if self.profile['cpu_freq']:
            print(f"CPU Frequency: {self.profile['cpu_freq']:.0f} MHz")
        print(f"Memory: {self.profile['memory_total_gb']:.1f} GB total, {self.profile['memory_available_gb']:.1f} GB available")
        print(f"Memory Usage: {self.profile['memory_usage_percent']:.1f}%")
        print(f"Performance Class: {self.profile['performance_class'].replace('_', ' ').title()}")
        
        print("\nOPTIMIZED SETTINGS:")
        print(f"Parallel Jobs: {self.optimization['n_jobs']}")
        print(f"Chunk Size: {self.optimization['chunk_size']:,}")
        print(f"Batch Size: {self.optimization['batch_size']:,}")
        print(f"Backend: {self.optimization['backend']}")
        print(f"Parallel Chunks: {self.optimization['use_parallel_chunks']}")
        print(f"Memory Safety: {self.optimization['memory_safety_factor']*100:.0f}%")
        print("=" * 60)

def estimate_computation_time(total_combinations, benchmark_time, n_jobs, parallel_efficiency=0.8):
    """Estimate total computation time"""
    if benchmark_time <= 0:
        return None
    
    sequential_time = total_combinations * benchmark_time
    parallel_time = sequential_time / (n_jobs * parallel_efficiency)
    
    return parallel_time

def format_time_estimate(seconds):
    """Format time estimate in human-readable format"""
    if seconds is None:
        return "Unknown"
    
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"

#########################################
# CROSS-SECTION CACHING
##########################################

_sigmav_cache = {}

def get_cached_sigmav(T_i_keV):
    """Cache expensive Bosch-Hale cross-section calculations"""
    T_i_rounded = round(T_i_keV, 2)
    T_i_rounded = round(T_i_keV, 2)
    
    if T_i_rounded not in _sigmav_cache:
        sigmav_DD_results = sigmav_DD_BoschHale(T_i_rounded * u.keV)
        sigmav_DD_p = sigmav_DD_results[1].to('m^3/s').magnitude  
        sigmav_DD_n = sigmav_DD_results[2].to('m^3/s').magnitude
        sigmav_DT = sigmav_DT_BoschHale(T_i_rounded * u.keV).to('m^3/s').magnitude
        
        _sigmav_cache[T_i_rounded] = (sigmav_DD_p, sigmav_DD_n, sigmav_DT)
    
    return _sigmav_cache[T_i_rounded]

#########################################
# WORKING SCRIPT APPROACH - MAIN ODE FUNCTIONS
##########################################

_ode_globals = {}

def injection_rate_fun(t, N_ifc, N_st, n_T, N_st_min=None):
    """Injection rate function using global variables for numerical stability"""
    if N_st_min is None:
        N_st_min = _ode_globals['N_st_min']
    
    g = _ode_globals
    n_tot = g['n_tot'] * u.m**(-3)
    tau_p_T = g['tau_p_T'] * u.s
    V_plasma = g['V_plasma'] * u.m**3
    sigmav_DT = g['sigmav_DT'] * u.m**3/u.s
    sigmav_DD_p = g['sigmav_DD_p'] * u.m**3/u.s
    tau_ifc = g['tau_ifc'] * u.s
    lambda_T = g['lambda_T'] * u.s**(-1)
    
    # Calculate injection_rate_max like working script
    injection_rate_max = (n_tot/2/tau_p_T*V_plasma + 0.25*n_tot**2*sigmav_DT*V_plasma - 0.25/2*n_tot**2*sigmav_DD_p*V_plasma).to('1/s')
    
    if N_st < N_st_min:
        return 0 * u.s**(-1)
    else:
        return min((N_ifc/tau_ifc - lambda_T*N_st), injection_rate_max).to('1/s')

def tritium_inventory_odes_global(t, y):
    """ODE function using global variables for numerical stability"""
    N_ofc, N_ifc, N_st = y[0], y[1], y[2]
    n_T = y[3]/u.m**3
    
    g = _ode_globals
    n_tot = g['n_tot'] * u.m**(-3)
    V_plasma = g['V_plasma'] * u.m**3
    tau_p_T = g['tau_p_T'] * u.s
    tau_ifc = g['tau_ifc'] * u.s
    tau_ofc = g['tau_ofc'] * u.s
    sigmav_DD_p = g['sigmav_DD_p'] * u.m**3/u.s
    sigmav_DD_n = g['sigmav_DD_n'] * u.m**3/u.s
    sigmav_DT = g['sigmav_DT'] * u.m**3/u.s
    TBR_DDn = g['TBR_DDn']
    TBR_DT = g['TBR_DT']
    lambda_T = g['lambda_T'] * u.s**(-1)
    
    n_D = n_tot - n_T
    
    injection_rate = injection_rate_fun(t, N_ifc, N_st, n_T)    
    
    Tdot_DDn = (TBR_DDn*0.5*n_D**2*sigmav_DD_n*V_plasma).to('1/s')
    Tdot_DDp = (0.5*n_D**2*sigmav_DD_p*V_plasma).to('1/s')
    Tdot_DT = (TBR_DT*n_D*n_T*sigmav_DT*V_plasma).to('1/s')
    Tdot_burn = (n_D*n_T*sigmav_DT*V_plasma).to('1/s')

    dN_ofc_dt = (Tdot_DT + Tdot_DDn - N_ofc / tau_ofc - N_ofc*lambda_T).to('1/s')
    dN_ifc_dt = (N_ofc / tau_ofc - N_ifc / tau_ifc  - lambda_T * N_ifc + n_T/tau_p_T*V_plasma).to('1/s')
    dN_stor_dt = (N_ifc / tau_ifc - lambda_T * N_st - injection_rate).to('1/s')
    dnT_dt = (injection_rate/V_plasma + Tdot_DDp/V_plasma - n_T/tau_p_T - Tdot_burn/V_plasma).to('1/s/m^3')

    return [float(dN_ofc_dt.to('1/s').magnitude), float(dN_ifc_dt.to('1/s').magnitude), float(dN_stor_dt.to('1/s').magnitude), float(dnT_dt.to('1/s/m^3').magnitude)]

def DT_reached_event_global(t, y):
    """Event function for DT condition using global variables"""
    n_T = y[3]
    n_tot = _ode_globals['n_tot']
    return n_T - 0.5 * n_tot

def negative_event_global(t, y):
    """Event function for negative states"""
    return min(y[0]+1e-10, y[1]+1e-10, y[2]+1e-10, y[3]+1e-10)

#########################################
# LEGACY ODE FUNCTION (FALLBACK)
##########################################

def tritium_inventory_odes(t, y, args):
    """
    Legacy ODE function for tritium inventory system (fallback method)
    """
    N_ofc, N_ifc, N_st, n_T = y
    (n_tot, sigmav_DD_p, sigmav_DD_n, sigmav_DT, V_plasma, 
     TBR_DDn, TBR_DT, tau_p_T, tau_ifc, tau_ofc, 
     injection_rate_max, N_st_min, lambda_T) = args
    
    # Convert back to proper physics units for calculations
    n_T_density = max(n_T, 0.0)  # Tritium density [m^-3]
    n_D_density = max(n_tot - n_T_density, 0.0)  # Deuterium density [m^-3]
    
    # Calculate reaction rates
    Tdot_DDn = max(TBR_DDn * 0.5 * n_D_density * n_D_density * sigmav_DD_n * V_plasma, 0.0)
    Tdot_DDp = max(0.5 * n_D_density * n_D_density * sigmav_DD_p * V_plasma, 0.0)
    Tdot_DT = max(TBR_DT * n_D_density * n_T_density * sigmav_DT * V_plasma, 0.0)
    Tdot_burn = max(n_D_density * n_T_density * sigmav_DT * V_plasma, 0.0)

    # Injection rate logic
    if N_st > N_st_min:
        injection_rate = min(N_st / tau_ifc, injection_rate_max)
    else:
        injection_rate = 0.0
   
    # ODE system
    dN_ofc_dt = Tdot_DT + Tdot_DDn - N_ofc / tau_ofc - N_ofc * lambda_T
    dN_ifc_dt = N_ofc / tau_ofc - N_ifc / tau_ifc - lambda_T * N_ifc + n_T_density / tau_p_T * V_plasma
    dN_stor_dt = N_ifc / tau_ifc - lambda_T * N_st - injection_rate
    dnT_dt = injection_rate / V_plasma + Tdot_DDp / V_plasma - n_T_density / tau_p_T - Tdot_burn / V_plasma
    
    return [dN_ofc_dt, dN_ifc_dt, dN_stor_dt, dnT_dt]

def DT_reached_event(t, y, args):
    """Event function to detect when DT condition is reached"""
    n_T = y[3]
    n_tot = args[0]
    # Stop when tritium reaches 50% of total density
    return n_T - 0.5 * n_tot

def negative_event(t, y, args):
    """Event function to detect negative states"""
    # Use same logic as working script with small offset
    return min(y[0] + 1e-10, y[1] + 1e-10, y[2] + 1e-10, y[3] + 1e-10)

# Set event properties
DT_reached_event.terminal = True
DT_reached_event.direction = 1  # Only trigger when crossing from below

negative_event.terminal = True  
negative_event.direction = -1  # Only trigger when crossing from positive to negative

#########################################
# SINGLE SOLVE_IVP FUNCTION (TO BE CALLED IN PARALLEL)
##########################################

def solve_single_combination(param_combo, input_data, total_time):
    """
    Solve ODE system for a single parameter combination
    Uses working script approach for numerical stability
    """
    # Extract data for each parameter
    extracted_data = [input_data[i][param_idx] for i, param_idx in enumerate(param_combo)]
        
    # Unpack the extracted data (all unit-stripped values)
    (V_plasma, T_i, n_tot,
     tau_p_T, tau_p_He3, 
     P_aux, P_lost_rad, P_aux_all_DT, P_lost_rad_all_DT,
     
     TBR_DT, TBR_DDn, tau_ifc, tau_ofc,
     
     eta_th, plant_avail, Cost_per_kWh,
     ) = extracted_data

    # OPTIMIZATION: Quick validity check to avoid expensive computations
    if (T_i <= 0 or n_tot <= 0 or V_plasma <= 0 or 
        tau_p_T <= 0 or tau_p_He3 <= 0 or P_aux <= 0):
        # Return failure case quickly
        return list(extracted_data) + [np.inf, 0, np.inf, np.inf, 0, 0, 0, 0]

    # Use cached cross-sections for better performance
    sigmav_DD_p, sigmav_DD_n, sigmav_DT = get_cached_sigmav(T_i)
    
    # Precompute repeated constants for speed
    half_V_plasma = 0.5 * V_plasma
    V_plasma_over_tau_p_T = V_plasma / tau_p_T
    quarter_V_plasma = 0.25 * V_plasma
    n_tot_squared = n_tot * n_tot
    
    # Calculate injection_rate_max with proper physics
    
    injection_rate_max = (
        n_tot/2/tau_p_T*V_plasma + 
        0.25*n_tot*n_tot*sigmav_DT*V_plasma - 
        0.25/2*n_tot*n_tot*sigmav_DD_p*V_plasma
    )
    
    # Calculate N_st_min
    # tritium_mass is already unit-stripped in global scope (kg)
    N_st_min = 0.001/tritium_mass
    
    # Use proper initial conditions
    y0 = np.zeros(4)  # [N_ofc, N_ifc, N_st, n_T] all start at zero
    
    # Time span (total_time already in seconds)
    t_span = (0, total_time)
    
    # Protect against zero division in stiffness ratio
    time_scales = [tau_p_T, tau_ifc, tau_ofc]
    min_time_scale = max(min(time_scales), 1e-12)  # Prevent zero
    max_time_scale = max(time_scales)
    stiffness_ratio = max_time_scale / min_time_scale
    
    # Enhanced solver settings selection
    characteristic_time = min_time_scale
    problem_scale = total_time / characteristic_time
    
    # Use same solver approach as working script
    
    method = 'BDF'           # Same as working script
    rtol = 1e-6              # Reasonable tolerance
    atol = 1e-9              # Reasonable tolerance  
    max_step = None          # Let solver choose (like working script)
    first_step = None        # Let solver choose (like working script)
    
    # Prepare args for top-level ODE function
    ode_args = (n_tot, sigmav_DD_p, sigmav_DD_n, sigmav_DT, V_plasma,
                TBR_DDn, TBR_DT, tau_p_T, tau_ifc, tau_ofc,
                injection_rate_max, N_st_min, lambda_T)
    
    # Prepare events with args - FIXED VERSION
    def DT_event_wrapper(t, y, *args):
        return DT_reached_event(t, y, args[0])  # args[0] contains ode_args
    
    def negative_event_wrapper(t, y, *args):
        return negative_event(t, y, args[0])  # args[0] contains ode_args
    
    # Set event properties
    DT_event_wrapper.terminal = True
    DT_event_wrapper.direction = 1
    negative_event_wrapper.terminal = True
    negative_event_wrapper.direction = -1
    
    try:
        # FIXED: Match working script solve_ivp call exactly
        solve_kwargs = {
            'fun': tritium_inventory_odes,
            't_span': t_span,
            'y0': y0,
            'method': method,
            'dense_output': False,  # Same as working script
            't_eval': None,         # Working script uses t_eval but we'll use dense_output
            'events': [DT_event_wrapper, negative_event_wrapper],
            'rtol': rtol,
            'atol': atol,
            'args': (ode_args,)
        }
        
        # Only add optional parameters if they're not None
        if max_step is not None:
            solve_kwargs['max_step'] = max_step
        if first_step is not None:
            solve_kwargs['first_step'] = first_step
        
        # TRY WORKING SCRIPT APPROACH FIRST
        try:
            
            # Set up global variables for ODE function
            global _ode_globals
            _ode_globals = {
                'n_tot': n_tot,
                'V_plasma': V_plasma,
                'tau_p_T': tau_p_T,
                'tau_ifc': tau_ifc,
                'tau_ofc': tau_ofc,
                'sigmav_DD_p': sigmav_DD_p,
                'sigmav_DD_n': sigmav_DD_n,
                'sigmav_DT': sigmav_DT,
                'TBR_DDn': TBR_DDn,
                'TBR_DT': TBR_DT,
                'lambda_T': lambda_T,
                'N_st_min': N_st_min
            }
            
            # Set up events for global approach
            DT_reached_event_global.terminal = True
            DT_reached_event_global.direction = 1
            negative_event_global.terminal = True
            negative_event_global.direction = -1
            
            # Solve with working script approach
            sol_working = solve_ivp(
                fun=tritium_inventory_odes_global,
                t_span=t_span,
                y0=y0,
                method='BDF',
                dense_output=False,
                events=[DT_reached_event_global, negative_event_global],
                rtol=rtol,
                atol=atol
            )
            
            if sol_working.success:
                sol = sol_working
            else:
                raise Exception("Working script approach failed, trying legacy")
                
        except Exception:
            # Fallback: Use original argument-based approach
            
            # Solve ODE with working-script-like settings
            sol = solve_ivp(**solve_kwargs)
            
            # Solve ODE with working-script-like settings
            sol = solve_ivp(**solve_kwargs)
        
        # Process results - check events first
        if not sol.success:
            # Integration failed
            t_startup = np.inf
            return create_result_row(param_combo, extracted_data, t_startup, None, sigmav_DT, sigmav_DD_n, sigmav_DD_p)
        
        if len(sol.t_events) > 1 and sol.t_events[1].size > 0:
            # Negative event occurred (second event)
            t_startup = np.inf
            return create_result_row(param_combo, extracted_data, t_startup, None, sigmav_DT, sigmav_DD_n, sigmav_DD_p)
        elif len(sol.t_events) > 0 and sol.t_events[0].size > 0:
            # DT condition reached (first event)
            t_startup = sol.t_events[0][0]
        else:
            # Time limit reached without DT condition
            t_startup = np.inf
        
        # For successful cases, we can evaluate the dense solution at specific points if needed
        # This is much more efficient than using t_eval
        return create_result_row(param_combo, extracted_data, t_startup, sol, sigmav_DT, sigmav_DD_n, sigmav_DD_p)
    
    except Exception as e:
        # Handle any integration errors silently during parallel execution
        return create_result_row(param_combo, extracted_data, np.inf, None, 
                                sigmav_DD_p, sigmav_DD_n, sigmav_DT)  # Fixed order

def create_result_row(param_combo, extracted_data, t_startup, sol, sigmav_DD_p, sigmav_DD_n, sigmav_DT):
    """
    Create a result row from the solution
    Optimized to return minimal objects for memory efficiency
    """
    # Unpack extracted_data again
    (V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad, 
     P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc, 
     tau_ofc, eta_th, plant_avail, Cost_per_kWh) = extracted_data
    
    if sol is None or t_startup == np.inf or not sol.success:
        # Failed case - return all input parameters and fill outputs with np.nan or np.inf
        return [
            V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad,
            P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc,
            tau_ofc, eta_th, plant_avail, Cost_per_kWh,
            np.inf,  # t_startup
            np.nan,  # P_fusion_startup
            np.nan,  # E_lost
            np.nan,  # Dollar_Lost
            np.nan,  # P_fusion_DD_startup
            np.nan,  # P_fusion_DT_startup
            np.nan,  # avg_n_T_startup
            np.nan,  # avg_n_D_startup
        ]
    
    # Calculate physics results using dense solution (more efficient)
    # Evaluate at startup time and a few intermediate points for averaging
    t_points = np.array([0, t_startup * 0.5, t_startup]) if t_startup < np.inf else np.array([0, sol.t[-1]])
    
    try:
        y_eval = sol.sol(t_points)  # Use dense output
        n_T_vals = y_eval[3]  # Tritium density values
        n_D_vals = n_tot - n_T_vals  # Deuterium density values
        
        # Use final values for startup calculations
        n_T_final = n_T_vals[-1]
        n_D_final = n_D_vals[-1]
        
    except Exception:
        # If dense evaluation fails, use last time point
        n_T_final = sol.y[3][-1] if len(sol.y[3]) > 0 else 0
        n_D_final = max(n_tot - n_T_final, 0)
        n_T_vals = np.array([n_T_final])
        n_D_vals = np.array([n_D_final])
    
    # Calculate fusion power at startup using precomputed cross-sections
    # Use scalar math for efficiency
    P_fusion_DD_startup = 0.5 * n_D_final * n_D_final * (sigmav_DD_n + sigmav_DD_p) * V_plasma * E_DDn * 1.602e-13
    P_fusion_DT_startup = n_D_final * n_T_final * sigmav_DT * V_plasma * E_DT * 1.602e-13
    P_fusion_startup = P_fusion_DD_startup + P_fusion_DT_startup
    
    # Simplified energy loss calculation (avoid expensive integration for screening)
    if t_startup < np.inf:
        # Rough estimate of energy lost during startup
        avg_n_T = np.mean(n_T_vals) if len(n_T_vals) > 1 else n_T_final
        avg_n_D = np.mean(n_D_vals) if len(n_D_vals) > 1 else n_D_final
        
        # Approximate average fusion power during startup
        avg_P_fusion = (0.5 * avg_n_D * avg_n_D * (sigmav_DD_n + sigmav_DD_p) * V_plasma * E_DDn + 
                       avg_n_D * avg_n_T * sigmav_DT * V_plasma * E_DT) * 1.602e-13
        
        # Energy lost is roughly (P_aux - avg_P_fusion) * t_startup
        P_net_loss = max(P_aux - avg_P_fusion - P_lost_rad, 0)
        E_lost = P_net_loss * t_startup
        
        # Economic calculation
        E_lost_kWh = E_lost / 3.6e6  # Convert J to kWh
        Dollar_Lost = E_lost_kWh * Cost_per_kWh
    else:
        avg_n_T = np.nan
        avg_n_D = np.nan
        E_lost = np.nan
        Dollar_Lost = np.nan
    
    return [
        V_plasma, T_i, n_tot, tau_p_T, tau_p_He3, P_aux, P_lost_rad,
        P_aux_all_DT, P_lost_rad_all_DT, TBR_DT, TBR_DDn, tau_ifc,
        tau_ofc, eta_th, plant_avail, Cost_per_kWh,
        t_startup,
        P_fusion_startup,
        E_lost,
        Dollar_Lost,
        P_fusion_DD_startup,
        P_fusion_DT_startup,
        avg_n_T,
        avg_n_D,
    ]


def choose_output_format(n_combinations):
    """
    Optimized output format selection: Parquet only for best performance
    
    Returns:
        tuple: (format_name, file_extension, save_function, load_example)
    """
    
    if not PARQUET_AVAILABLE:
        print("⚠️  Warning: pyarrow not available - install with 'conda install -c conda-forge pyarrow'")
        print("   Falling back to CSV (much slower for large datasets)")
        return ('CSV', '.csv',
                lambda df, filename: df.to_csv(filename, index=False),
                "pd.read_csv('filename.csv')")
    
    # Always use Parquet for optimal performance
    if n_combinations < 10000:
        # Small datasets: Parquet with fast compression
        return ('Parquet', '.parquet',
                lambda df, filename: df.to_parquet(filename, index=False, compression='snappy'),
                "pd.read_parquet('filename.parquet')")
    else:
        # Large datasets: Parquet with high compression
        return ('Parquet', '.parquet',
                lambda df, filename: df.to_parquet(filename, index=False, compression='gzip'),
                "pd.read_parquet('filename.parquet')")


#########################################
# MAIN PARALLELIZATION EXECUTION
##########################################

def get_memory_info():
    """Get available memory in GB"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_available = None
        for line in lines:
            if 'MemAvailable:' in line:
                mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
                break
        if mem_available is None:
            # Fallback if MemAvailable not found
            for line in lines:
                if 'MemFree:' in line:
                    mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
                    break
        return mem_available / (1024**3)  # Convert to GB
    except:
        return 4.0  # Default fallback

def calculate_optimal_jobs(total_combinations, available_memory_gb):
    """Calculate optimal number of jobs based on memory and combinations"""
    # For very large parameter spaces, use minimal jobs to prevent memory issues
    if total_combinations > 1000000:  # > 1M combinations
        return 2  # Very conservative
    elif total_combinations > 100000:  # > 100K combinations  
        return 3
    elif total_combinations > 10000:   # > 10K combinations
        return 4
    else:
        # Estimate memory per job (conservative estimate)
        memory_per_job_gb = 0.1  # ~100MB per job
        
        # Calculate max jobs based on memory (use only 50% of available memory)
        max_jobs_memory = max(1, int(available_memory_gb * 0.5 / memory_per_job_gb))
        
        # Calculate max jobs based on CPU cores
        max_jobs_cpu = min(4, os.cpu_count())  # Cap at 4 for stability
        
        # Use the minimum of memory and CPU constraints
        optimal_jobs = min(max_jobs_memory, max_jobs_cpu)
        
        return max(1, optimal_jobs)

if __name__ == "__main__":
    # Initialize system profiler
    profiler = SystemProfiler()
    profiler.print_system_info()
    
    # Create parameter combinations
    param_ranges = [range(data.shape[0]) for data in input_data]
    total_combinations = np.prod([len(r) for r in param_ranges])
    
    print(f"\nPARAMETER SPACE:")
    print(f"Total combinations: {total_combinations:,}")
    
    # CRITICAL: Warn about large parameter spaces but allow execution
    if total_combinations > 1000000:  # > 1M combinations 
        print(f"🚨 WARNING: {total_combinations:,} combinations is very large!")
        print(f"This may take significant time but will search for viable combinations.")
        print(f"Your parameter space has {total_combinations/1000000:.1f} million combinations!")
        print("\n� ANALYSIS GOAL:")
        print("   Searching for parameter combinations that achieve startup < 5 years")
        print("   Even if most combinations fail, we need to find the viable ones!")
        print("\n⏱️  ESTIMATED TIME:")
        estimated_hours = total_combinations * 0.002  # Optimistic 2ms per combination
        if estimated_hours > 24:
            print(f"   Expected runtime: {estimated_hours/24:.1f} days")
        else:
            print(f"   Expected runtime: {estimated_hours:.1f} hours")
        
        # Ask user but don't force exit
        response = input(f"\nProceed with {total_combinations:,} combinations to find viable ones? (y/n): ")
        if response.lower() != 'y':
            print("Exiting by user choice.")
            exit()
        else:
            print("🚀 PROCEEDING: Searching for successful startup combinations...")
        
    elif total_combinations > 100000:  # > 100K combinations - warn but allow
        print(f"🚨 WARNING: {total_combinations:,} combinations is large!")
        print(f"Searching for parameter combinations that achieve startup < 5 years")
        
        response = input(f"\nContinue with {total_combinations:,} combinations? (y/n): ")
        if response.lower() != 'y':
            print("Exiting by user choice.")
            exit()
        else:
            print("🚀 PROCEEDING: Analyzing parameter space...")
    
    # Benchmark single computation for time estimation
    print("\nRunning performance benchmark...")
    sample_combo = tuple([0] * len(param_ranges))  # First parameter combination
    total_time = 5 * 365.25 * 24 * 3600  # 5 years in seconds
    
    benchmark_time, benchmark_success = profiler.benchmark_single_computation(
        sample_combo, input_data, total_time
    )
    
    if benchmark_success:
        estimated_total_time = estimate_computation_time(
            total_combinations, 
            benchmark_time, 
            profiler.optimization['n_jobs']
        )
        print(f"Benchmark: {benchmark_time:.2f}s per combination")
        print(f"Estimated total time: {format_time_estimate(estimated_total_time)}")
    else:
        print("Benchmark failed - proceeding with default estimates")
    
    # Use optimized settings with dynamic tuning
    n_jobs = profiler.optimization['n_jobs']
    chunk_size = profiler.optimization['chunk_size']
    batch_size = profiler.optimization['batch_size']
    backend = profiler.optimization['backend']
    use_parallel_chunks = profiler.optimization['use_parallel_chunks']
    
    # OPTIMIZATION 9: Dynamic chunk size tuning based on benchmark results
    if benchmark_time is not None:
        if benchmark_time < 0.01:  # Very fast computations
            chunk_size = min(50000, chunk_size * 5)  # Larger chunks
            buffer_size_multiplier = 0.5  # Smaller buffer
        elif benchmark_time > 0.5:  # Slow computations  
            chunk_size = max(100, chunk_size // 2)  # Smaller chunks
            buffer_size_multiplier = 2.0  # Larger buffer
        else:
            buffer_size_multiplier = 1.0
            
        print(f"Auto-tuned chunk size: {chunk_size} (based on {benchmark_time:.3f}s benchmark)")
    else:
        buffer_size_multiplier = 1.0
    
    print(f"\nStarting optimized parallel computation...")
    print(f"Using {n_jobs} jobs, {chunk_size} chunk size, {backend} backend")
    
    # Add diagnostic information
    param_iter_test = itertools.product(*param_ranges)
    first_few = list(itertools.islice(param_iter_test, 3))
    print(f"First few parameter combinations to test:")
    # Show first few combinations only for small datasets
    if total_combinations <= 1000:
        for i, combo in enumerate(first_few):
            print(f"  {i+1}: {combo}")
    
    # Quick test of solve function (silent for large datasets)
    if total_combinations <= 100000:
        print("Testing solve function with first combination...")
    try:
        test_result = solve_single_combination(first_few[0], input_data, total_time)
        if total_combinations <= 100000:
            print(f"Test successful! Result length: {len(test_result)}")
            print(f"Sample values: t_startup={test_result[-8]:.2f}, P_fusion={test_result[-7]:.2e}")
        
    except Exception as e:
        if total_combinations <= 100000:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            print("Switching to sequential mode for debugging...")
        use_parallel_chunks = False
        n_jobs = 1
    
    # Adaptive processing based on parameter space size
    if total_combinations > 10000:  # > 10K combinations
        print("Using chunked processing...")
        
        # Better saving approach: Use faster binary formats
        output_filename = f"parametric_analysis_results_{points}x{points}_points"
        
        # Use HDF5 for fast saving and loading (much faster than CSV)
        h5_filename = output_filename + ".h5"
        
        # Define column names
        column_names = [
            'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_lost_rad',
            'P_aux_all_DT', 'P_lost_rad_all_DT', 'TBR_DT', 'TBR_DDn', 'tau_ifc',
            'tau_ofc', 'eta_th', 'plant_avail', 'Cost_per_kWh',
            't_startup', 'P_fusion_startup', 'E_lost', 'Dollar_Lost',
            'P_fusion_DD_startup', 'P_fusion_DT_startup', 'avg_n_T_startup', 'avg_n_D_startup'
        ]
        
        # Initialize empty list for fast in-memory accumulation
        # We'll save in larger chunks to minimize I/O
        results_buffer = []
        buffer_size = int(10000 * buffer_size_multiplier)  # Dynamic buffer size
        
        print(f"Results will be saved to {h5_filename} (HDF5 format - much faster)")
        print(f"Optimized buffer size: {buffer_size} results per save operation")
        
        # RESUME FUNCTIONALITY: Check if partial results exist
        resume_from = 0
        if os.path.exists(h5_filename):
            try:
                import h5py
                with h5py.File(h5_filename, 'r') as f:
                    if 'results' in f:
                        existing_rows = f['results'].shape[0]
                        response = input(f"Found existing results ({existing_rows:,} combinations). Resume? (y/n): ")
                        if response.lower() == 'y':
                            resume_from = existing_rows
                            print(f"Resuming from combination {resume_from:,}")
                        else:
                            print("Starting fresh (will overwrite existing file)")
            except Exception as e:
                print(f"Could not read existing file: {e}")
                print("Starting fresh")
        
        if resume_from > 0:
            print(f"Skipping first {resume_from:,} combinations...")
        
        # Statistics tracking (memory-efficient)
        stats = {
            'successful_count': 0,
            'failed_count': 0,
            't_startup_sum': 0.0,
            't_startup_min': float('inf'),
            't_startup_max': 0.0,
            'e_lost_sum': 0.0,
            'dollar_lost_sum': 0.0
        }
        
        processed = 0
        
        # OPTIMIZATION 7: Create chunks of parameter combinations with memory optimization
        # Use generator to avoid creating entire parameter space in memory
        def optimized_param_generator():
            """Memory-efficient parameter combination generator"""
            for combo in itertools.product(*param_ranges):
                yield combo
        
        param_iter = optimized_param_generator()
        
        # Skip combinations if resuming
        if resume_from > 0:
            for _ in range(resume_from):
                next(param_iter, None)
            processed = resume_from
        else:
            processed = 0
        
        # Main progress bar with ETA and rate information
        main_pbar = tqdm(
            total=total_combinations, 
            desc="Overall Progress", 
            unit="combo",
            unit_scale=True,
            dynamic_ncols=True,
            position=0,
            initial=processed,  # Start progress bar from resume point
            smoothing=0.1,      # Smooth the rate calculation
            miniters=1000       # Update every 1000 items minimum
        )
        
        while processed < total_combinations:
            # Get next chunk
            chunk = list(itertools.islice(param_iter, chunk_size))
            if not chunk:
                break
                
            chunk_num = processed // chunk_size + 1
            
            if use_parallel_chunks:
                # OPTIMIZATION 5: Enhanced parallel processing with dynamic batch sizing
                # Adjust batch size based on chunk size for optimal performance
                dynamic_batch_size = max(1, min(batch_size, len(chunk) // n_jobs))
                
                chunk_results = Parallel(
                    n_jobs=n_jobs, 
                    verbose=0, 
                    backend=backend,
                    batch_size=dynamic_batch_size,
                    prefer="threads",
                    pre_dispatch='2*n_jobs',  # Control memory usage
                    temp_folder=None,  # Use memory instead of disk for temporary results
                    max_nbytes=None  # No limit on array size for pickling
                )(
                    delayed(solve_single_combination)(param_combo, input_data, total_time) 
                    for param_combo in tqdm(chunk, desc=f"Chunk {chunk_num}", leave=False, position=1)
                )
            else:
                # Sequential processing within chunk (for low-resource systems)
                chunk_results = []
                chunk_pbar = tqdm(
                    chunk,
                    desc=f"Chunk {chunk_num}",
                    leave=False,
                    position=1
                )
                
                for param_combo in chunk_pbar:
                    result = solve_single_combination(param_combo, input_data, total_time)
                    chunk_results.append(result)
            
            # OPTIMIZATION 8: Vectorized result processing and statistics
            # Convert to numpy array immediately for better memory efficiency
            chunk_array = np.array(chunk_results, dtype=np.float64)
            results_buffer.append(chunk_array)
            
            # Vectorized statistics update (much faster than loops)
            t_startup_col = chunk_array[:, -8]  # t_startup column
            e_lost_col = chunk_array[:, -6]     # E_lost column  
            dollar_lost_col = chunk_array[:, -5] # Dollar_Lost column
            
            # Boolean masks for vectorized operations
            finite_mask = np.isfinite(t_startup_col)
            successful_count = np.sum(finite_mask)
            failed_count = len(chunk_array) - successful_count
            
            # Update statistics vectorized way
            stats['successful_count'] += successful_count
            stats['failed_count'] += failed_count
            
            if successful_count > 0:
                successful_t_startup = t_startup_col[finite_mask]
                stats['t_startup_sum'] += np.sum(successful_t_startup)
                stats['t_startup_min'] = min(stats['t_startup_min'], np.min(successful_t_startup))
                stats['t_startup_max'] = max(stats['t_startup_max'], np.max(successful_t_startup))
                
                # Handle E_lost and Dollar_Lost (check for NaN)
                successful_e_lost = e_lost_col[finite_mask]
                successful_dollar_lost = dollar_lost_col[finite_mask]
                
                valid_e_lost = successful_e_lost[~np.isnan(successful_e_lost)]
                valid_dollar_lost = successful_dollar_lost[~np.isnan(successful_dollar_lost)]
                
                if len(valid_e_lost) > 0:
                    stats['e_lost_sum'] += np.sum(valid_e_lost)
                if len(valid_dollar_lost) > 0:
                    stats['dollar_lost_sum'] += np.sum(valid_dollar_lost)
            
            processed += len(chunk)
            main_pbar.update(len(chunk))
            
            # SAFETY: Monitor memory usage to prevent crashes
            current_memory_percent = psutil.virtual_memory().percent
            if current_memory_percent > 85:  # High memory usage
                tqdm.write(f"⚠️  HIGH MEMORY USAGE: {current_memory_percent:.1f}% - forcing save and cleanup")
                # Force save regardless of buffer size
                if results_buffer:
                    # Concatenate all arrays in buffer
                    if len(results_buffer) > 1:
                        results_array = np.vstack(results_buffer)
                    else:
                        results_array = results_buffer[0]
                    
                    # Save to HDF5
                    import h5py
                    mode = 'a' if processed > len(chunk) else 'w'
                    with h5py.File(h5_filename, mode) as f:
                        if 'results' not in f:
                            f.create_dataset('results', data=results_array, maxshape=(None, len(column_names)))
                            f['results'].attrs['columns'] = [col.encode('utf-8') for col in column_names]
                        else:
                            current_size = f['results'].shape[0]
                            new_size = current_size + len(results_array)
                            f['results'].resize((new_size, len(column_names)))
                            f['results'][current_size:] = results_array
                    
                    tqdm.write(f"🚨 Emergency save: {len(results_array)} results (Memory: {current_memory_percent:.1f}%)")
                    results_buffer.clear()
                    
                    # Aggressive cleanup
                    import gc
                    gc.collect()
                    
            # Save buffer to disk when it gets large enough (much more efficient)
            current_buffer_size = sum(len(arr) for arr in results_buffer)
            if current_buffer_size >= buffer_size or processed >= total_combinations:
                # Concatenate all arrays in buffer
                if len(results_buffer) > 1:
                    results_array = np.vstack(results_buffer)
                else:
                    results_array = results_buffer[0]
                
                # Save to HDF5 (much faster than CSV)
                import h5py
                mode = 'a' if processed > len(chunk) else 'w'  # append if not first save
                with h5py.File(h5_filename, mode) as f:
                    if 'results' not in f:
                        # Create dataset on first write
                        f.create_dataset('results', data=results_array, maxshape=(None, len(column_names)))
                        # Save column names as attributes
                        f['results'].attrs['columns'] = [col.encode('utf-8') for col in column_names]
                    else:
                        # Append to existing dataset
                        current_size = f['results'].shape[0]
                        new_size = current_size + len(results_array)
                        f['results'].resize((new_size, len(column_names)))
                        f['results'][current_size:] = results_array
                
                results_buffer.clear()  # Clear buffer to free memory
                
                # OPTIMIZATION 10: Force garbage collection to free memory
                import gc
                gc.collect()
        
        main_pbar.close()
        
        print(f"\n🎉 COMPUTATION COMPLETED!")
        print(f"📊 Final Stats:")
        print(f"   • Total processed: {processed:,} combinations")
        print(f"   • Successful startups: {stats['successful_count']:,}")
        print(f"   • Failed simulations: {stats['failed_count']:,}")
        
        print(f"Completed {processed:,} computations using chunked processing")
        
    else:
        # Non-chunked processing for smaller datasets
        print("Using direct processing for smaller parameter space...")
        
        # For smaller datasets, just use in-memory processing
        output_filename = f"parametric_analysis_results_{points}x{points}_points"
        
        # Choose optimal output format based on dataset size
        format_name, file_ext, save_func, load_example = choose_output_format(total_combinations)
        output_filename_full = output_filename + file_ext
        
        print(f"Using {format_name} format for optimal performance")
        
        # Define column names
        column_names = [
            'V_plasma', 'T_i', 'n_tot', 'tau_p_T', 'tau_p_He3', 'P_aux', 'P_lost_rad',
            'P_aux_all_DT', 'P_lost_rad_all_DT', 'TBR_DT', 'TBR_DDn', 'tau_ifc',
            'tau_ofc', 'eta_th', 'plant_avail', 'Cost_per_kWh',
            't_startup', 'P_fusion_startup', 'E_lost', 'Dollar_Lost',
            'P_fusion_DD_startup', 'P_fusion_DT_startup', 'avg_n_T_startup', 'avg_n_D_startup'
        ]
        
        # For small datasets, keep results in memory
        all_results = []
        
        print(f"Results will be saved to {output_filename_full} at the end")
        
        # Statistics tracking (memory-efficient)
        stats = {
            'successful_count': 0,
            'failed_count': 0,
            't_startup_sum': 0.0,
            't_startup_min': float('inf'),
            't_startup_max': 0.0,
            'e_lost_sum': 0.0,
            'dollar_lost_sum': 0.0
        }
        
        # Use generator to avoid loading all combinations into memory at once
        def param_combinations_generator():
            """Generator that yields parameter combinations one by one"""
            for combo in itertools.product(*param_ranges):
                yield combo
        
        # Create progress bar
        pbar = tqdm(
            total=total_combinations,
            desc="Processing combinations",
            unit="combo",
            unit_scale=True,
            dynamic_ncols=True
        )
        
        # Process combinations in smaller batches to show progress
        batch_size = 100  # Process 100 at a time to update progress more frequently
        param_iter = param_combinations_generator()
        processed = 0
        
        while processed < total_combinations:
            # Get next batch
            batch = list(itertools.islice(param_iter, batch_size))
            if not batch:
                break
            
            # Process this batch
            batch_results = Parallel(
                n_jobs=n_jobs, 
                verbose=0, 
                backend='threading',
                batch_size=1,
                prefer="threads"
            )(
                delayed(solve_single_combination)(param_combo, input_data, total_time) 
                for param_combo in batch
            )
            
            # Add to results list (in-memory for small datasets)
            all_results.extend(batch_results)
            
            # Update statistics
            for result in batch_results:
                t_startup = result[-8]  # t_startup is at index -8
                if np.isfinite(t_startup):
                    stats['successful_count'] += 1
                    stats['t_startup_sum'] += t_startup
                    stats['t_startup_min'] = min(stats['t_startup_min'], t_startup)
                    stats['t_startup_max'] = max(stats['t_startup_max'], t_startup)
                    
                    e_lost = result[-6]  # E_lost is at index -6
                    dollar_lost = result[-5]  # Dollar_Lost is at index -5
                    if not np.isnan(e_lost):
                        stats['e_lost_sum'] += e_lost
                    if not np.isnan(dollar_lost):
                        stats['dollar_lost_sum'] += dollar_lost
                else:
                    stats['failed_count'] += 1
            
            processed += len(batch_results)
            pbar.update(len(batch_results))
        
        pbar.close()
        
        # Save all results at once (fast for small datasets)
        df_results = pd.DataFrame(all_results, columns=column_names)
        
        # Choose efficient output format based on dataset size
        format_name, file_ext, save_func, load_example = choose_output_format(len(all_results))
        output_filename_full = output_filename + file_ext
        
        # Save in the selected format
        save_func(df_results, output_filename_full)
        
        print(f"Saved {len(all_results)} results to {output_filename_full}")
        print(f"Completed {processed:,} computations using direct processing")

    print(f"\nAll computations completed")

    # Print summary statistics from tracked values
    print("\nSummary Statistics:")
    print(f"Successful startups: {stats['successful_count']}")
    print(f"Failed simulations: {stats['failed_count']}")
    
    if stats['successful_count'] > 0:
        avg_startup_time = stats['t_startup_sum'] / stats['successful_count']
        print(f"Average startup time: {avg_startup_time/(365.25*24*3600):.2f} years")
        print(f"Min startup time: {stats['t_startup_min']/(365.25*24*3600):.2f} years")
        print(f"Max startup time: {stats['t_startup_max']/(365.25*24*3600):.2f} years")
        
        if stats['e_lost_sum'] > 0:
            avg_e_lost = stats['e_lost_sum'] / stats['successful_count']
            print(f"Average energy lost during startup: {avg_e_lost/1e9:.2f} GJ")
            avg_dollar_lost = stats['dollar_lost_sum'] / stats['successful_count']
            print(f"Average cost during startup: ${avg_dollar_lost/1e6:.2f} M")
    else:
        print("No successful startups found in the parameter space.")
    
    # Provide information about the output file format
    format_name, file_ext, save_func, load_example = choose_output_format(total_combinations)
    output_filename_final = output_filename + file_ext
    print(f"\n📁 Results saved to: {output_filename_final}")
    print(f"   • Format: {format_name}")
    print("   • Reading in Python:")
    print(f"     import pandas as pd")
    
    # Extract just the filename without path
    filename_only = output_filename_final.split('/')[-1]
    
    # Create proper load example
    if format_name == 'Parquet':
        print(f"     df = pd.read_parquet('{filename_only}')")
    else:
        print(f"     df = pd.read_csv('{filename_only}')")