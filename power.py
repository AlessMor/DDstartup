from sigmav_functions import sigmav_DT_BoschHale

def calculate_P_e_net_Q(Pf:float, Q:float, eta_th:float) -> float:
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
    P_e_net = eta_th * Pf - P_aux # [W] is the net electrical power produced by the reactor
    return P_e_net, P_aux

def calculate_P_e_net_Paux(Pf:float, P_aux:float, eta_th:float) -> float:
    r"""Calculate the net electrical power produced by the reactor.

    Args:
        Pf: Fusion power [W]
        Q: Auxiliary heating power [W]
        eta_th: Thermal efficiency of the reactor [-]

    Returns:
        P_e_net: Net electrical power produced by the reactor [W]
    """
    Q = (Pf - P_aux)/P_aux 
    P_e_net = eta_th * Pf - P_aux # [W] is the net electrical power produced by the reactor
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