import numpy as np
from numpy import float64
from numpy.typing import NDArray

from sigmav_functions import *
from units_and_constants import unit


def volumetric_reaction_rates_DDfuel(
    density_D: NDArray[float64], # [m^-3]
    temperature: NDArray[float64], # [keV]
    tau_p_T: NDArray[float64], # [s] needed to estimate the reaction rates of the secondary reaction D-T
    tau_p_He3: NDArray[float64] # [s] needed to estimate the reaction rates of the secondary reaction D-He3
):
    
    DEBUG = False
    r"""Calculate volumetric reaction rates for D-D main reactions,
        as well as D-T and D-He3 secondary reactions.

    Args:
        density_D: D density [m^-3]
        temperature: ion temperature profile [keV]
        tau_p_T: tritium confinement time [s]           
        tau_p_He3: He3 confinement time [s]
    Returns:
        dictionary: dictionary with the following keys:
            - rr_DDp: volumetric reaction rate for D-D -> D-T + p [1/m^3/s]
            - rr_DDn: volumetric reaction rate for D-D -> D-He3 + n [1/m^3/s]
            - rr_DT: volumetric reaction rate for D-T -> He4 + n [1/m^3/s]
            - rr_DHe3: volumetric reaction rate for D-He3 -> He4 + p [1/m^3/s]
            - rr_tot: total volumetric reaction rate [1/m^3/s]
            - density_T: tritium density [m^-3]
            - density_He3: He3 density [m^-3]
            - prob_DDp: probability of D-D -> D-T + p [-]
            - prob_DDn: probability of D-D -> D-He3 + n [-]
            - prob_DT: probability of D-T -> He4 + n [-]
            - prob_DHe3: probability of D-He3 -> He4 + p [-]
            - prob_tot: total probability [-]
    """
    
    
    """ recalling that
            |----> He3(1.01 MeV) + n(2.45 MeV)
    D + D ->|
            |----> T(1.01 MeV) + p(3.02 MeV)
            
    AND
    Following the calculations done by Pedretti, E., Rollet, S., 
    "COMPARISON OF NEUTRON PRODUCTIONS DURING OPERATION OF THE ARIES-III SECOND STABILITY D3 He TOKAMAK REACTOR WITHOUT AND WITH TRITIUM-ASSISTED STARTUP" 
    (1994):

    Production of deuteric T = (n_D/2)^2 * < sigmav >_DDp  

    Losses of deuteric T = n_D * n_T,D * < sigmav >_DT + n_T,D / tau_p
    (where n_T,D is the tritium produced bby DDp reactions)

    At steady-state the two equilibrate:  
    (1/2) * n_D^2 * < sigmav >_DDp = n_D * n_T,D * < sigmav >_DT + n_T,D / tau_p  

    Then:  
    n_T,D =( (1/2) * n_D^2 * < sigmav >_DDp ) / ( n_D * < sigmav >_DT + 1 / tau_p )
    
    A similar discussion can be made for the He3 production
    """
    
    
    # calculate the reactivities for all the main and secondary reactions
    # use BoschHale fit exclusively


    # Call sigmav functions with raw numerical values
    sigmav_DD_tot = sigmav_DD_BoschHale(temperature)[0]
    sigmav_DD_p = sigmav_DD_BoschHale(temperature)[1] 
    sigmav_DD_n = sigmav_DD_BoschHale(temperature)[2] 
    sigmav_DT = sigmav_DT_BoschHale(temperature) 
    sigmav_DHe3 = sigmav_DHe3_BoschHale(temperature)

    # calculate volumetric reaction rates for the two main reactions (the 0.5 factor takes into account fusion from identical particles)
    rr_DDp = 0.5 * sigmav_DD_p * density_D**2 # [1/m^3/s]
    rr_DDn = 0.5 * sigmav_DD_n * density_D**2 # [1/m^3/s]
    
    # estimate volumetric reaction rates for the secondary reactions    
    density_T = (0.5 * density_D**2 * sigmav_DD_p)/(density_D * sigmav_DT + 1/tau_p_T) # [m^-3]
    density_He3 = (0.5 * density_D**2 * sigmav_DD_n)/(density_D * sigmav_DHe3 + 1/tau_p_He3) # [m^-3]
    
    rr_DT = sigmav_DT * density_D * density_T # [1/m^3/s]
    rr_DHe3 = sigmav_DHe3 * density_D * density_He3 # [1/m^3/s]
    
    # calculate the probabilities associated to the reactions
    rr_tot = rr_DDp + rr_DDn + rr_DT + rr_DHe3 # [1/m^3/s]
    prob_DDp = rr_DDp / rr_tot
    prob_DDn = rr_DDn / rr_tot
    prob_DT = rr_DT / rr_tot
    prob_DHe3 = rr_DHe3 / rr_tot
    
    # build a dictionary with the results
    dictionary = {
        "rr_DDp": rr_DDp, # [1/m^3/s]
        "rr_DDn": rr_DDn, # [1/m^3/s]
        "rr_DT": rr_DT, # [1/m^3/s]
        "rr_DHe3": rr_DHe3, # [1/m^3/s]
        "rr_tot": rr_tot,   # [1/m^3/s]
        "density_T": density_T, # [m^-3]
        "density_He3": density_He3, # [m^-3]
        "prob_DDp": prob_DDp,   # [-]
        "prob_DDn": prob_DDn,   # [-]
        "prob_DT": prob_DT,  # [-]
        "prob_DHe3": prob_DHe3, # [-]
        "prob_tot": prob_DDp + prob_DDn + prob_DT + prob_DHe3 # [-]
    }
    if DEBUG == True:
        print(f"reaction rates in [1/m^3/s]:\n",
            f"    rr_DDp: {rr_DDp}, rr_DDn: {rr_DDn}, rr_DT: {rr_DT}, rr_DHe3: {rr_DHe3}, rr_tot: {rr_tot}")
        print(f"density in [m^-3]:\n",
            f"    density_T: {density_T}, density_He3: {density_He3}")
        print(f"probabilities:\n",
            f"    prob_DDp: {prob_DDp}, prob_DDn: {prob_DDn}, prob_DT: {prob_DT}, prob_DHe3: {prob_DHe3}, prob_tot: {prob_DDp + prob_DDn + prob_DT + prob_DHe3}")
        
    return dictionary
    