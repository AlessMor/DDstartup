import numpy as np
from numpy import float64
from numpy.typing import NDArray
from itertools import product
from .sigmav_functions import *
from .units_and_constants import u

def calculate_reaction_rates(
    n_tot,  # [m^-3] density of D+T, can be scalar or array (spatial distribution)
    T_i,  # [keV] ion temperature, can be scalar or array (spatial distribution)
    V,  # [m^3], plasma volume
    tau_p_T=None,  # [s], can be scalar or array
    tau_p_He3=None,  # [s], can be scalar or array
    tau_p_D=None,  # [s], can be scalar or array
    f_T=[0.5],  # [-], fraction of Tritium in the plasma, can be scalar or 1D array (time distribution)
):
    """
    Calculate reaction rates for DD fusion with time-dependent tritium fraction.
    
    Parameters:
    -----------
    n_tot : scalar or 1D array
        D+T density [m^-3]. If array, represents spatial distribution.
    T_i : scalar or 1D array  
        Ion temperature [keV]. If array, represents spatial distribution.
    V : scalar
        Plasma volume [m^3]
    tau_p_T, tau_p_He3, tau_p_D : scalar
        Particle confinement times [s]
    f_T : scalar or 1D array
        Tritium fraction [-]. If array, represents time evolution.
        
    Returns:
    --------
    dictionary : dict
        Reaction rates and densities. If f_T is an array (time-dependent),
        each rate will be a 1D array with length equal to len(f_T).
        If n_e/T_e are arrays (spatial), integration is performed over space.
    """
    n_time = len(f_T) if np.ndim(f_T) > 0 else 1
    n_spatial = max(len(n_tot[0]), len(T_i[0]))
    
    # Define the integral function (working with pint quantities)
    def integral_func(n1, n2, sigmav, V):
        if n_spatial == 1:
            # Single point calculation - no spatial dependence
            return V * n1[0] * n2[0] * sigmav[0]
        else:
            # Array integration over spatial profile
            # Integral: V * int_0^1 n1(x) * n2(x) * sigmav(x) dx
            x = np.linspace(0, 1, n_spatial)
            integrand = n1 * n2 * sigmav
            integral = V * np.trapz(integrand, x)
            integral = integral[0][0]
            return integral

    # Get cross-sections
    sigmav_DD_p = sigmav_DD_BoschHale(T_i)[1]  # [m^3/s]
    sigmav_DD_n = sigmav_DD_BoschHale(T_i)[2]  # [m^3/s]
    sigmav_DT = sigmav_DT_BoschHale(T_i)       # [m^3/s]
    sigmav_DHe3 = sigmav_DHe3_BoschHale(T_i)   # [m^3/s]
    
    # Initialize output arrays
    R_DDp_values = np.zeros(n_time)
    R_DDn_values = np.zeros(n_time)
    R_DT_values = np.zeros(n_time)
    R_DD_DT_values = np.zeros(n_time)
    R_DHe3_values = np.zeros(n_time)
    n_D_avg_values = np.zeros(n_time)
    n_T_avg_values = np.zeros(n_time)
    n_T_prod_avg_values = np.zeros(n_time)
    n_He3_prod_avg_values = np.zeros(n_time)
    N_D = np.zeros(n_time)
    N_T = np.zeros(n_time)
    
    # Calculate reaction rates for each time point
    for i in range(n_time):
        # Calculate densities at this time point (spatial profiles)
        n_D = n_tot * (1 - f_T[i])  # [m^-3] - D density (spatial profile)
        n_T = n_tot * f_T[i]  # [m^-3] - T density (spatial profile)
        # Secondary products (spatial profiles)
        n_T_prod = (0.5 * n_D**2 * sigmav_DD_p) / (n_D * sigmav_DT + 1 / tau_p_T)
        n_He3_prod = (0.5 * n_D**2 * sigmav_DD_n) / (n_D * sigmav_DHe3 + 1 / tau_p_He3)

        n_T_inj = np.maximum(n_T - n_T_prod,0) # [m^-3] - T injected (spatial profile)
    
        # Use the integral function for all reaction rates
        R_DDp_i = 0.5 * integral_func(n_D, n_D, sigmav_DD_p, V)
        R_DDn_i = 0.5 * integral_func(n_D, n_D, sigmav_DD_n, V)
        R_DT_i = integral_func(n_D, n_T, sigmav_DT, V)
        R_DD_DT_i = integral_func(n_D, n_T_prod, sigmav_DT, V)
        R_DT_inj_i = integral_func(n_D, n_T_inj, sigmav_DT, V)
        R_DHe3_i = integral_func(n_D, n_He3_prod, sigmav_DHe3, V)
        
        #check if DT reaction rate is correct
        if R_DT_i != R_DT_inj_i+ R_DD_DT_i:
            raise ValueError(f"R_DT_i = {R_DT_i}, R_DT_inj_i = {R_DT_inj_i}, R_DD_DT_i = {R_DD_DT_i} do not match")
        
        
        # Store the magnitude values (pint quantities can't be stored directly in numpy arrays)
        R_DDp_values[i] = R_DDp_i.magnitude
        R_DDn_values[i] = R_DDn_i.magnitude
        R_DT_values[i] = R_DT_i.magnitude
        R_DD_DT_values[i] = R_DD_DT_i.magnitude
        R_DHe3_values[i] = R_DHe3_i.magnitude
        
        # Store volume-averaged densities for this time point
        if n_spatial > 1:
            x = np.linspace(0, 1, n_spatial)
            # Volume-weighted average
            n_D_avg_values[i] = (np.trapz(n_D * x, x) / np.trapz(x, x)).magnitude
            n_T_avg_values[i] = (np.trapz(n_T * x, x) / np.trapz(x, x)).magnitude
            n_T_prod_avg_values[i] = (np.trapz(n_T_prod * x, x) / np.trapz(x, x)).magnitude
            n_He3_prod_avg_values[i] = (np.trapz(n_He3_prod * x, x) / np.trapz(x, x)).magnitude
            N_D[i] = (np.trapz(n_D * x, x)).magnitude
            N_T[i] = (np.trapz(n_T * x, x)).magnitude
        else:
            n_D_avg_values[i] = n_D[0].magnitude
            n_T_avg_values[i] = n_T[0].magnitude
            n_T_prod_avg_values[i] = n_T_prod[0].magnitude
            n_He3_prod_avg_values[i] = n_He3_prod[0].magnitude
            
        
    
    # Apply units back to results
    R_DDp = R_DDp_values * (1/u.s)
    R_DDn = R_DDn_values * (1/u.s)
    R_DT = R_DT_values * (1/u.s)
    R_DD_DT = R_DD_DT_values * (1/u.s)
    R_DHe3 = R_DHe3_values * (1/u.s)
    
    n_D_avg = n_D_avg_values * n_tot.units
    n_T_avg = n_T_avg_values * n_tot.units + n_T_prod_avg_values * n_tot.units  # Add T production to T average
    n_T_prod_avg = n_T_prod_avg_values * n_tot.units
    n_He3_prod_avg = n_He3_prod_avg_values * n_tot.units
    
    # Calculate total reaction rate and probabilities
    R_tot = R_DDp + R_DDn + R_DT + R_DHe3
    
    # Handle division by zero for probability calculations
    R_tot_mag = R_tot.magnitude
    prob_DDp = np.divide(R_DDp.magnitude, R_tot_mag, out=np.zeros_like(R_DDp.magnitude), where=R_tot_mag!=0)
    prob_DDn = np.divide(R_DDn.magnitude, R_tot_mag, out=np.zeros_like(R_DDn.magnitude), where=R_tot_mag!=0)
    prob_DT_prod = np.divide(R_DD_DT.magnitude, R_tot_mag, out=np.zeros_like(R_DD_DT.magnitude), where=R_tot_mag!=0)
    prob_DHe3 = np.divide(R_DHe3.magnitude, R_tot_mag, out=np.zeros_like(R_DHe3.magnitude), where=R_tot_mag!=0)
    prob_DT = np.divide(R_DT.magnitude, R_tot_mag, out=np.zeros_like(R_DT.magnitude), where=R_tot_mag!=0)
    
    prob_tot = prob_DDp + prob_DDn + prob_DT + prob_DHe3 + prob_DT_prod
    
    # Check probability conservation
    if np.any(np.abs(prob_tot - 1) > 0.01):
        bad_indices = np.where(np.abs(prob_tot - 1) > 0.01)[0]
        raise ValueError(f"prob_tot != 1 at time indices {bad_indices}: {prob_tot[bad_indices]}")
    
    # Build dictionary with results
    dictionary = {
        "R_DT": R_DT,           # [1/s] - time vector
        "R_DDp": R_DDp,         # [1/s] - time vector
        "R_DDn": R_DDn,         # [1/s] - time vector
        "R_DD_DT": R_DD_DT,     # [1/s] - time vector
        "R_DHe3": R_DHe3,       # [1/s] - time vector
        "R_tot": R_tot,         # [1/s] - time vector
        
        "n_D_avg": n_D_avg,         # [m^-3] - time vector (volume-averaged)
        "n_T_avg": n_T_avg,         # [m^-3] - time vector (volume-averaged)
        "n_T_prod_avg": n_T_prod_avg,   # [m^-3] - time vector (volume-averaged)
        "n_He3_prod_avg": n_He3_prod_avg,  # [m^-3] - time vector (volume-averaged)
        "N_D": N_D,            
        "N_T": N_T,             
        
        "prob_DDp": prob_DDp,   # [-] - time vector
        "prob_DDn": prob_DDn,   # [-] - time vector
        "prob_DT_prod": prob_DT_prod,  # [-] - time vector
        "prob_DHe3": prob_DHe3, # [-] - time vector
        "prob_DT": prob_DT,     # [-] - time vector
        "prob_tot": prob_tot,   # [-] - time vector
        
        "f_T": f_T              # [-] - time vector (input)
    }
    
    # If f_T was scalar, return scalar results
    if n_time == 1:
        for key in dictionary:
            if isinstance(dictionary[key], np.ndarray) and dictionary[key].size == 1:
                dictionary[key] = dictionary[key].item()
            elif hasattr(dictionary[key], 'magnitude') and dictionary[key].magnitude.size == 1:
                dictionary[key] = dictionary[key].magnitude.item() * dictionary[key].units
    
    return dictionary

def calculate_reaction_rates_test(
    n_tot,  # [m^-3] density of D+T, can be scalar or array (spatial distribution)
    T_i,  # [keV] ion temperature, can be scalar or array (spatial distribution)
    V_plasma,  # [m^3], plasma volume
    ):
    """
    Test function for calculate_reaction_rates.
    """
    
    n_spatial = max(len(n_tot[0]), len(T_i[0]))
    
    # Define the integral function (working with pint quantities)
    def integral_func(n1, n2, sigmav, V):
        if n_spatial == 1:
            # Single point calculation - no spatial dependence
            return V * n1[0] * n2[0] * sigmav[0]
        else:
            # Array integration over spatial profile
            # Integral: V * int_0^1 n1(x) * n2(x) * sigmav(x) dx
            x = np.linspace(0, 1, n_spatial)
            integrand = n1 * n2 * sigmav
            integral = V * np.trapz(integrand, x)
            integral = integral[0][0]
            return integral
        

    # Get cross-sections
    sigmav_DD_p = sigmav_DD_BoschHale(T_i)[1]  # [m^3/s]
    sigmav_DD_n = sigmav_DD_BoschHale(T_i)[2]  # [m^3/s]
    sigmav_DT = sigmav_DT_BoschHale(T_i)       # [m^3/s]
    sigmav_DHe3 = sigmav_DHe3_BoschHale(T_i)   # [m^3/s]


    RrDDn = 0.5 * integral_func(n_tot, n_tot, sigmav_DD_n, V_plasma)
    RrDDp = 0.5 * integral_func(n_tot, n_tot, sigmav_DD_p, V_plasma)
    RrDT = integral_func(n_tot, n_tot, sigmav_DT, V_plasma)
    
    return {
        "RrDDn": RrDDn.magnitude,  # [1/s]
        "RrDDp": RrDDp.magnitude,  # [1/s]
        "RrDT": RrDT.magnitude,    # [1/s]
    }