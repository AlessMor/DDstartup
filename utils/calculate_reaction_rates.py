import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .sigmav_functions import *
from .units_and_constants import unit


def calculate_reaction_rates_DD(
    n_e_DD,  # [m^-3], can be scalar or array
    T_e,  # [keV], can be scalar or array
    V,  # [m^3], optional, only needed for profile/integral
    tau_p_T,  # [s], can be scalar or array
    tau_p_He3,  # [s], can be scalar or array
    points=10000  # number of points for the integral (for profiles)
):
    def integral_func(n1, n2, sigmav, V):
        x = np.linspace(0, 1, points)
        dx = x[1] - x[0]  # uniform spacing

        # total_reaction_rate = 2V*int(n1*n2*sigma*x*dx)_in [0,1]
        # n1 and n2 are the density profiles, sigma is the cross section, V is the volume
        # return the integral of the reaction rate
        return 2 * V * np.trapz(n1 * n2 * sigmav*x, dx=1/points)  # [m^3/s]

    sigmav_DD_tot = sigmav_DD_BoschHale(T_e)[0]
    sigmav_DD_p = sigmav_DD_BoschHale(T_e)[1]
    sigmav_DD_n = sigmav_DD_BoschHale(T_e)[2]
    sigmav_DT = sigmav_DT_BoschHale(T_e)
    sigmav_DHe3 = sigmav_DHe3_BoschHale(T_e)
    
    # estimate volumetric reaction rates for the secondary reactions    
    n_T= (0.5 * n_e_DD**2 * sigmav_DD_p) / (n_e_DD * sigmav_DT + 1 / tau_p_T)  # [m^-3]
    n_He3 = (0.5 * n_e_DD**2 * sigmav_DD_n) / (n_e_DD * sigmav_DHe3 + 1 / tau_p_He3)  # [m^-3]
    
    R_DDp = 0.5*integral_func(n_e_DD, n_e_DD, sigmav_DD_p, V)  # [1/s]
    R_DDn = 0.5*integral_func(n_e_DD, n_e_DD, sigmav_DD_n, V)  # [1/s]
    R_DD_DT = integral_func(n_e_DD, n_T, sigmav_DT, V)  # [1/s]
    R_DHe3 = integral_func(n_e_DD, n_He3, sigmav_DHe3, V)  # [1/s]
    
    # calculate the probabilities associated to the reactions
    R_tot = R_DDp + R_DDn + R_DD_DT + R_DHe3 # [1/m^3/s]
    prob_DDp = R_DDp / R_tot
    prob_DDn = R_DDn / R_tot
    prob_DT = R_DD_DT / R_tot
    prob_DHe3 = R_DHe3 / R_tot
    prob_tot = prob_DDp + prob_DDn + prob_DT + prob_DHe3
    if prob_tot != 1:
        if abs(prob_tot - 1) > 0.01:
            raise ValueError(f"prob_tot = {prob_tot} != 1")
    
    # build a dictionary with the results
    dictionary = {
        "R_DDp": R_DDp, # [1/m^3/s]
        "R_DDn": R_DDn, # [1/m^3/s]
        "R_DT": R_DD_DT, # [1/m^3/s]
        "R_DHe3": R_DHe3, # [1/m^3/s]
        "R_tot": R_tot,   # [1/m^3/s]
        "density_T": n_T, # [m^-3]
        "density_He3": n_He3, # [m^-3]
        "prob_DDp": prob_DDp,   # [-]
        "prob_DDn": prob_DDn,   # [-]
        "prob_DT": prob_DT,  # [-]
        "prob_DHe3": prob_DHe3, # [-]
        "prob_tot": prob_tot # [-]
    }
    return dictionary



def calculate_reaction_rates_DT(
    n_e,  # [m^-3], can be scalar or array
    T_e,  # [keV], can be scalar or array
    V,  # [m^3], optional, only needed for profile/integral
    points=1000  # number of points for the integral (for profiles)
):
    def integral_func(n1, n2, sigmav, V):
        x = np.linspace(0, 1, points)
        dx = x[1] - x[0]  # uniform spacing

        # total_reaction_rate = 2V*int(n1*n2*sigma*x*dx)_in [0,1]
        # n1 and n2 are the density profiles, sigma is the cross section, V is the volume
        # return the integral of the reaction rate
        return 2 * V * np.trapz(n1 * n2 * sigmav*x, dx=1/points)  # [m^3/s]

    sigmav_DD_tot = sigmav_DD_BoschHale(T_e)[0]
    sigmav_DD_p = sigmav_DD_BoschHale(T_e)[1]
    sigmav_DD_n = sigmav_DD_BoschHale(T_e)[2]
    sigmav_DT = sigmav_DT_BoschHale(T_e)
    
    n_D = n_e/2  # [m^-3]
    
    # check if n_T and n_He3 are scalar or array
    if np.isscalar(n_e) and np.isscalar(T_e):
        R_DDp = 0.5*n_D**2 * sigmav_DD_p *  V  # [1/s]
        R_DDn = 0.5*n_D**2 * sigmav_DD_n * V # [1/s]
        R_DT = sigmav_DT * (n_e/2)**2 * V  # [1/s]
    else:
        R_DDp = integral_func(n_e/2, n_e/2, sigmav_DD_p, V)  # [1/s]
        R_DDn = integral_func(n_e/2, n_e/2, sigmav_DD_n, V)  # [1/s]
        R_DT = integral_func(n_e/2, n_e/2, sigmav_DT, V)  # [1/s]
    # calculate the probabilities associated to the reactions
    R_tot = R_DDp + R_DDn + R_DT # [1/m^3/s]
    prob_DDp = R_DDp / R_tot
    prob_DDn = R_DDn / R_tot
    prob_DT = R_DT / R_tot
    prob_tot = prob_DDp + prob_DDn + prob_DT
    if prob_tot != 1:
        if abs(prob_tot - 1) > 0.01:
            raise ValueError(f"prob_tot = {prob_tot} != 1")
        else:
            print(f"Warning: prob_tot = {prob_tot} != 1")
    
    # build a dictionary with the results
    dictionary = {
        "R_DDp": R_DDp, # [1/s]
        "R_DDn": R_DDn, # [1/s]
        "R_DT": R_DT, # [1/s]
        "R_tot": R_tot,   # [1/s]
        "density_T": n_e/2, # [m^-3]
        "density_D": n_e/2, # [m^-3]
        "prob_DDp": prob_DDp,   # [-]
        "prob_DDn": prob_DDn,   # [-]
        "prob_DT": prob_DT,  # [-]
        "prob_tot": prob_tot # [-]
    }
    return dictionary

    
    
def pedestal_profile(x, value_center=1, value_ped=0.5, value_edge=0, transition_ratio=0.95):
    """
    Generate a position-dependent profile for a tokamak (e.g., density or temperature).

    Parameters:
    - x: Position array (e.g., along the minor radius, normalized 0 to 1).
    - value_center: Value at the center (x=0).
    - value_ped: Value at the pedestal/transition point.
    - value_edge: Value at the edge (x=1).
    - transition_ratio: Fraction of the minor radius where the transition occurs (0 < transition_ratio < 1).

    Returns:
    - Profile as a numpy array.
    """
    transition_point = transition_ratio * np.max(x)
    profile = np.zeros_like(x)* value_center

    # Parabolic region (x <= transition_point)
    parabola_mask = x <= transition_point
    profile[parabola_mask] = value_center - (value_center - value_ped) * (x[parabola_mask] / transition_point) ** 2

    # Linear region (x > transition_point)
    linear_mask = x > transition_point
    profile[linear_mask] = value_ped + (value_edge - value_ped) * (x[linear_mask] - transition_point) / (np.max(x) - transition_point)

    # Compute the volume-averaged value of the profile
    numerator = np.trapz(profile * x, x)
    denominator = np.trapz(x, x)
    profile_avg = numerator / denominator

    return profile, profile_avg