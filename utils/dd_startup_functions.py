from .sigmav_functions import sigmav_DT_BoschHale
import numpy as np
from utils.units_and_constants import *
from scipy.integrate import solve_ivp

def calculate_P_e_net_Q(Pf:float, Q:float, eta_th:float, P_rad = 0) -> float:
    r"""Calculate the net electrical power produced by the reactor.

    Args:
        Pf: Fusion power [W]
        Q: Auxiliary heating power [W]
        eta_th: Thermal efficiency of the reactor [-]

    Returns:
        P_e_net: Net electrical power produced by the reactor [W]
    """
    # Q = (Pf - Paux)/Paux then Paux = Pf/(Q+1)
    P_aux = Pf / (Q + 1) # [W] is the auxiliary heating power needed to maintain the plasma temperature
    P_e_net = eta_th * (Pf -P_rad)- P_aux # [W] is the net electrical power produced by the reactor
    return P_e_net, P_aux

def calculate_P_e_net_Paux(Pf:float, P_aux:float, eta_th:float, P_rad = 0) -> float:
    r"""Calculate the net electrical power produced by the reactor.

    Args:
        Pf: Fusion power [W]
        P_aux: Auxiliary heating power [W]
        eta_th: Thermal efficiency of the reactor [-]

    Returns:
        P_e_net: Net electrical power produced by the reactor [W]
        Q: Physical Energy gain factor [-]
    """
    Q = (Pf - P_aux)/P_aux 
    P_e_net = eta_th * (Pf-P_rad) - P_aux # [W] is the net electrical power produced by the reactor
    return P_e_net, Q


def fusion_power_50D50T(n_e_avg, T_e_avg, E_DT, V_plasma):
    """
    Calculate the fusion power for a 50% Deuterium (D) and 50% Tritium (T) plasma.

    Args:
        n_e_avg: Average electron density [1/m^3]
        T_e_avg: Average electron temperature [keV]
        E_DT: Energy released by DT reactions [J]
        V_plasma: Plasma volume [m^3]
        sigmav_DT_BoschHale: Function to calculate <sigmav> for DT reactions [m^3/s]

    Returns:
        Pf_DT: Fusion power for DT reactions [W]
    """

    sigmav_DT = sigmav_DT_BoschHale(T_e_avg) # Reactivity for DT reactions [m^3/s]
    Pf_DT = (n_e_avg / 2)**2 * sigmav_DT * E_DT * V_plasma  # Fusion power [W]
    return Pf_DT

def fusion_power_DT(n_tot_profile, T_profile, V_plasma, f_DT=0.5):
    """
    Calculate the fusion power for DT reactions.

    Args:
        n_tot_profile: Total density profile [1/m^3]
        T_profile: Temperature profile [keV]
        V_plasma: Plasma volume [m^3]
        f_DT: fraction of D over T in the plasma, default is 0.5 (50% D, 50% T)
    Returns:
        Pf_DT: Fusion power for DT reactions [W]
    """
    sigmav_DT = sigmav_DT_BoschHale(T_profile)  # [m^3/s], array or scalar

    # Convert to arrays for consistent handling
    n_arr = np.atleast_1d(n_tot_profile)
    T_arr = np.atleast_1d(T_profile)
    sigmav_arr = np.atleast_1d(sigmav_DT)

    if n_arr.size == 1:
        # Analytic formula for uniform plasma
        Pf_DT = (n_arr[0] * f_DT)*(n_arr[0]*(1-f_DT)) * sigmav_arr[0] * E_DT * V_plasma
    else:
        points = n_arr.size
        r = np.linspace(0, 1, points)
        integrand = n_arr * f_DT * n_arr * f_DT * sigmav_arr * E_DT * r
        integral = np.trapz(integrand, r)
        Pf_DT = 2 * V_plasma * integral
    return Pf_DT

def compute_tritium_production(DD_reaction_rates, TBR, TBR_DDn, V_plasma, tau_p_T):
    """Calculate tritium production rates."""
    Tdot_fusion = DD_reaction_rates["R_DDp"] - DD_reaction_rates["R_DT"] # [1/s] rate of tritium production due to DDp fusions, considering the losses due to DT neutrons (NB. It is assumed that all the Tritium that is not burnt in DT fusios can be extracted from the system)
    Tdot_breedingDT = TBR * DD_reaction_rates["R_DT"] # [1/s] is the rate of tritium production due to DT neutrons interacting with the breeding blanket
    Tdot_breedingDD = TBR_DDn * DD_reaction_rates["R_DDn"] # [1/s] is the rate of tritium production due to DD neutrons interacting with the Li6 in the breeding blanket
    #Tdot_diff = V_plasma * DD_reaction_rates["density_T"] / tau_p_T # [1/s] is the rate of tritium production due to diffusion of tritium in the breeding blanket
    Tdot_tot = Tdot_fusion + Tdot_breedingDT + Tdot_breedingDD # [1/s] is the total rate of tritium production in the system
    
    return Tdot_fusion, Tdot_breedingDT, Tdot_breedingDD, Tdot_tot

def compute_startup_time(I_ST, Tdot_tot, molecular_weight_T, plant_availability=1):
    """Calculate the startup time for tritium inventory."""
    N_ST = I_ST / molecular_weight_T.to("kg/mol") * N_A
    lambda_T = np.log(2) / ((12.32 * u.year).to('s'))
    effective_Tdot = Tdot_tot * plant_availability
    ratio = N_ST * lambda_T / Tdot_tot
    if ratio >= 1:
        return np.inf * u.s
    else:
        return - (1/lambda_T) * np.log(1 - (ratio)) 

def compute_fusion_power(DD_reaction_rates, n_e_avg, T_e_avg, V_plasma):
    """Calculate fusion power for DD and DT."""
    Pf_DD = DD_reaction_rates["R_DDp"]*E_DDp + DD_reaction_rates["R_DDn"]*E_DDn
    Pf_DD_DT = DD_reaction_rates["R_DT"]*E_DT
    Pf_DD_DHe3 = DD_reaction_rates["R_DHe3"]*E_DHe3
    Pf_DD_tot = Pf_DD.to('MW') + Pf_DD_DT.to('MW') + Pf_DD_DHe3.to('MW')
    Pf_DT = fusion_power_DT(n_e_avg, T_e_avg, V_plasma, f_DT=0.5)  # Assuming 50% D and 50% T
    return Pf_DD, Pf_DD_DT, Pf_DD_DHe3, Pf_DD_tot, Pf_DT

def compute_net_power(Pf_DD, Pf_DT, P_aux, eta_th, startup_time, Cost_per_kWh, P_rad=0):
    """Calculate net electrical power, Q, energy lost, and dollar lost."""
    P_e_net_DD, Q_DD = calculate_P_e_net_Paux(Pf_DD, P_aux, eta_th, P_rad)
    P_e_net_DT, Q_DT = calculate_P_e_net_Paux(Pf_DT, P_aux, eta_th, P_rad)
    E_lost = (P_e_net_DT - P_e_net_DD) * startup_time
    Dollar_lost = Cost_per_kWh.to('1/J') * E_lost
    return P_e_net_DD, Q_DD, P_e_net_DT, Q_DT, E_lost, Dollar_lost


def compute_startup_inventory(N_T_burn, tau_ifc, tau_ofc, TBR, TBE):
    """Calculate the startup inventory given the fuel cycle characteristic timescales.

    Args:
        N_T_burn (float): Burn rate of tritium.
        tau_ifc (float): Characteristic timescale for the in-fusion cycle.
        tau_ofc (float): Characteristic timescale for the out-fusion cycle.
        TBR (float): Tritium breeding ratio.
        TBE (float): Tritium breeding efficiency.

    Returns:
        float: Startup inventory, calculated as the initial inventory minus the minimum inventory during the time span.
    """
    # System of ODEs
    def tritium_inventory_odes(t, y):
        I_ofc, I_ifc, I_st = y
        dI_ofc_dt = N_T_burn * TBR - I_ofc / tau_ofc
        dI_ifc_dt = I_ofc / tau_ofc - I_ifc / tau_ifc + N_T_burn * (1 - TBE) / TBE
        dI_st_dt  = I_ifc / tau_ifc - N_T_burn / TBE
        return [dI_ofc_dt, dI_ifc_dt, dI_st_dt]

    # Initial conditions
    y0 = [0, 0, 1.5]  # assuming inventories start from zero

    # Time span
    t_span = (0, 5 * tau_ofc)  # 5 days
    t_eval = np.linspace(*t_span, 1000)

    # Solve ODE
    sol = solve_ivp(tritium_inventory_odes, t_span, y0, t_eval=t_eval)
    I_st = sol.y[2]
    min_index = np.argmin(I_st)  # Find the index of the minimum inventory
    min_time = sol.t[min_index]  # Time at which the minimum occurs
    I_st_min = I_st[min_index]  # Minimum inventory value
    I_st_initial = y0[2]  # Initial inventory
    I_startup = I_st_initial - I_st_min  # Initial inventory minus inventory at the minimum point
    # Initial inventory minus inventory at the minimum point
    return I_startup  # time and all inventories


####################################################################################################################################################

def calculate_P_e_net(P_fus_tot, P_aux=0, P_rad=0, eta_th=1.0, plant_avail=1.0):
    r"""Calculate the net electrical power produced by the reactor.

    Args:
        P_fus_tot: Total fusion power [W]
        P_aux: Auxiliary heating power [W]
        eta_th: Thermal efficiency of the reactor [-]
        plant_avail: Plant availability factor [-]

    Returns:
        P_e_net: Net electrical power produced by the reactor [W]
    """
    Q = (P_fus_tot - P_aux)/P_aux if P_aux > 0 else np.inf
    P_e_net = plant_avail * (eta_th * (P_fus_tot - P_rad) - P_aux)  # [W] is the net electrical power produced by the reactor
    return P_e_net, Q


def injection_rate_fun(N_ifc, N_st, tau_ifc=12*u.h, N_st_min = 0.001*tritium_mass.to('kg').magnitude, injection_rate_max=1e20/u.s):
    if N_st < N_st_min:
        # if the inventory in the storage is below the minimum, do not inject
        return 0 * u.s**(-1)
    else:
        # tries to inject all the T that enters the storage, limit to injection_rate_max
        return min((N_ifc/tau_ifc - lambda_T*N_st), injection_rate_max).to('1/s')

