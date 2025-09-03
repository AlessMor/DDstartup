import numpy as np
from typing import Union, List, Optional, Any
from .units_and_constants import u
from pint import UnitRegistry
from scipy.stats import norm

class Parameter:
    def __init__(
        self,
        value: Union[float, int],
        unit: Optional[UnitRegistry] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        num_samples: Optional[int] = None,  # NEW
    ):
        self.value = value
        self.name = name
        self.unit = unit
        self.description = description
        self.num_samples = num_samples  # NEW

    def __repr__(self):
        name_str = f"name={self.name}, " if self.name else ""
        unit_str = f"unit={self.unit}, " if self.unit else ""
        description_str = f"description={self.description}, " if self.description else ""
        num_samples_str = f"num_samples={self.num_samples}, " if self.num_samples is not None else ""
        return f"Parameter {name_str}{num_samples_str}value={self.value} {unit_str})\n     {description_str}"

class SingleParameter(Parameter):
    def __init__(self, value, unit=None, name=None, description=None, num_samples=None):
        super().__init__(value, unit, name, description, num_samples)

    def get_single_value(self):
        return self.value if self.unit is None else self.value * self.unit

    def get_values(self, num: Optional[int] = None) -> List[float]:
        n = num if num is not None else self.num_samples if self.num_samples is not None else 1
        return [self.get_single_value()] * n

    def get_magnitudes(self, num: Optional[int] = None) -> List[float]:
        values = self.get_values(num)
        return [v.magnitude if hasattr(v, "magnitude") else v for v in values]

    def get_mean_value(self):
        return self.get_single_value()
    def get_min(self):
        return self.get_single_value()
    def get_max(self):
        return self.get_single_value()
    def __repr__(self):
        return super().__repr__()

class LinSpaceParameter(Parameter):
    def __init__(self, start, stop, unit=None, name=None, description=None, num_samples=None):
        super().__init__(start, unit, name, description, num_samples)
        self.start = start
        self.stop = stop

    def get_values(self, num: Optional[int] = None) -> List:
        n = num if num is not None else self.num_samples if self.num_samples is not None else 5
        if n == 1:
            values = [self.get_mean_value() * self.unit] if self.unit is not None else [self.get_mean_value()]
        else:
            values = np.linspace(self.start, self.stop, n)
            if self.unit is not None:
                return [v * self.unit for v in values]
        return values

    def get_magnitudes(self, num: Optional[int] = None) -> List[float]:
        values = self.get_values(num)
        return [v.magnitude if hasattr(v, "magnitude") else v for v in values]

    def get_mean_value(self):
        return (self.start + self.stop) / 2
    def get_min(self):
        return self.start * u.second if self.unit is None else self.start * self.unit
    def get_max(self):
        return self.stop * u.second if self.unit is None else self.stop * self.unit
    def __repr__(self):
        return super().__repr__()

class NormSpaceParameter(Parameter):
    def __init__(self, mean, std_dev, unit=None, name=None, description=None, num_samples=None):
        super().__init__(mean, unit, name, description, num_samples)
        self.mean = mean
        self.std_dev = std_dev

    def get_values(self, num: Optional[int] = None) -> List[float]:
        n = num if num is not None else self.num_samples if self.num_samples is not None else 5
        if n == 1:
            values = [self.mean]
        else:
            quantiles = np.linspace(0.01, 0.99, n)
            values = norm.ppf(quantiles, loc=self.mean, scale=self.std_dev)
        if self.unit is not None:
            return [v * self.unit for v in values]
        return values

    def get_magnitudes(self, num: Optional[int] = None) -> List[float]:
        values = self.get_values(num)
        return [v.magnitude if hasattr(v, "magnitude") else v for v in values]

    def get_mean_value(self):
        return self.mean
    def get_min(self, num=5):
        vals = self.get_values(num)
        return min(vals)
    def get_max(self, num=5):
        vals = self.get_values(num)
        return max(vals)
    def __repr__(self):
        return super().__repr__()

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from itertools import product
from .sigmav_functions import *
from .units_and_constants import u


def calculate_reaction_rates_DD(
    n_e_DD,  # [m^-3], can be scalar or array
    T_e,  # [keV], can be scalar or array
    V,  # [m^3], optional, only needed for profile/integral
    tau_p_T,  # [s], can be scalar or array
    tau_p_He3,  # [s], can be scalar or array
):
    
    points = len(n_e_DD) if isinstance(n_e_DD, (list, np.ndarray)) else 1000
    
    def integral_func(n1, n2, sigmav, V):
        # If all inputs are scalars, just return the analytic result
        if (np.isscalar(n1.magnitude) and np.isscalar(n2.magnitude) and np.isscalar(sigmav.magnitude)) or (len(n1)==1 and len(n2)==1 and len(sigmav)==1):
            return V * n1 * n2 * sigmav  # The 0.5 comes from integrating x from 0 to 1

        # Otherwise, do the array-based integration as before
        for arr, name in zip([n1, n2, sigmav], ["n1", "n2", "sigmav"]):
            if not hasattr(arr, "__len__"):
                raise TypeError(f"{name} must be an array-like object, got {type(arr)}")

        arr_lens = [len(n1), len(n2), len(sigmav)]
        if len(set(arr_lens)) != 1:
            raise ValueError(
                f"All input arrays must have the same length, got: "
                f"len(n1)={len(n1)}, len(n2)={len(n2)}, len(sigmav)={len(sigmav)}"
            )
        points = arr_lens[0]
        x = np.linspace(0, 1, points)
        if len(x) != points:
            raise ValueError(f"x must have length {points}, got {len(x)}")

        try:
            result = 2 * V * np.trapz(n1 * n2 * sigmav * x, dx=1/points)
        except Exception as e:
            raise RuntimeError(f"Error during integration: {e}")

        return result  # [m^3/s]

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

    
    
"""def pedestal_profile(x, value_center=1, value_ped=0.5, value_edge=0, transition_ratio=0.95):
int = transition_ratio * np.max(x)
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

    return profile, profile_avg"""



def pedestal_profile(value_center=1*u.keV, value_ped=0.5*u.keV, value_edge=0*u.keV, transition_ratio=0.95, n=1000):
    """
    Generate one or more position-dependent profiles for a tokamak (e.g., density or temperature).
    All value_* parameters can be pint.Quantity or floats.
    Returns:
    - profiles: list of pint.Quantity arrays (or a single Quantity array if all parameters are scalars)
    - profile_avgs: list of pint.Quantity (or a single Quantity if all parameters are scalars)
    """

    x = np.linspace(0, 1, n)

    # Helper to extract magnitude and unit
    def get_mag_unit(val):
        try:
            return val.magnitude, val.units
        except AttributeError:
            return val, 1

    # Convert all to lists if not already
    def to_list(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            return list(val)
        else:
            return [val]

    centers = to_list(value_center)
    peds = to_list(value_ped)
    edges = to_list(value_edge)
    transitions = to_list(transition_ratio)

    combos = list(product(centers, peds, edges, transitions))
    profiles = []
    avgs = []

    for c, p, e, t in combos:
        c_mag, c_unit = get_mag_unit(c)
        p_mag, p_unit = get_mag_unit(p)
        e_mag, e_unit = get_mag_unit(e)
        # Use the unit of the center value (or fallback to p or e)
        unit_out = c_unit if c_unit != 1 else (p_unit if p_unit != 1 else e_unit)
        if n == 1:
            # Only one point: profile is just the center value
            profile_mag = np.array([c_mag])
            profile_avg_mag = c_mag
        else:
            transition_point = t * np.max(x)
            profile_mag = np.zeros_like(x, dtype=float) * c_mag
            parabola_mask = x <= transition_point
            profile_mag[parabola_mask] = c_mag - (c_mag - p_mag) * (x[parabola_mask] / transition_point) ** 2
            linear_mask = x > transition_point
            profile_mag[linear_mask] = p_mag + (e_mag - p_mag) * (x[linear_mask] - transition_point) / (np.max(x) - transition_point)
            numerator = np.trapz(profile_mag * x, x)
            denominator = np.trapz(x, x)
            profile_avg_mag = numerator / denominator


        # Attach unit to profile and average
        profiles.append(profile_mag * unit_out)
        avgs.append(profile_avg_mag * unit_out)
    
    return profiles, avgs



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
    
    # Calculate reaction rates for each time point
    for i in range(n_time):
        # Calculate densities at this time point (spatial profiles)
        n_D = n_tot * (1 - f_T[i])  # [m^-3] - D density (spatial profile)
        
        # Secondary products (spatial profiles)
        n_T_prod = (0.5 * n_D**2 * sigmav_DD_p) / (n_D * sigmav_DT + 1 / tau_p_T)
        n_He3_prod = (0.5 * n_D**2 * sigmav_DD_n) / (n_D * sigmav_DHe3 + 1 / tau_p_He3)

        n_T = np.maximum(n_tot * f_T[i], n_T_prod) # [m^-3] - T density (spatial profile)
        n_T_inj = np.maximum(n_T - n_T_prod,0) # [m^-3] - T injected (spatial profile)
    
        # Use the integral function for all reaction rates
        R_DDp_i = 0.5 * integral_func(n_D, n_D, sigmav_DD_p, V)
        R_DDn_i = 0.5 * integral_func(n_D, n_D, sigmav_DD_n, V)
        R_DT_i = integral_func(n_D, n_T, sigmav_DT, V)
        R_DD_DT_i = integral_func(n_D, n_T_prod, sigmav_DT, V)
        R_DHe3_i = integral_func(n_D, n_He3_prod, sigmav_DHe3, V)
        
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
    R_tot = R_DDp + R_DDn + R_DT + R_DHe3 + R_DD_DT
    
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