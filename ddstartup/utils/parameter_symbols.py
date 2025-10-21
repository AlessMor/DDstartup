"""
Parameter symbol mappings for plotting.

This module provides LaTeX-formatted symbols for parameters to use in plot labels
instead of variable names, improving readability and conforming to scientific conventions.
"""

# Mapping from parameter names to LaTeX symbols
PARAM_SYMBOLS = {
    # Plasma parameters
    'V_plasma': r'$V_{\mathrm{plasma}}$',
    'T_i': r'$T_i$',
    'n_tot': r'$n_{\mathrm{tot}}$',
    'n_T': r'$n_T$',
    'n_D': r'$n_D$',
    'n_He3': r'$n_{\mathrm{He3}}$',
    
    # Confinement times
    'tau_p_T': r'$\tau_{p,T}$',
    'tau_p_He3': r'$\tau_{p,\mathrm{He3}}$',
    'tau_ifc': r'$\tau_{\mathrm{ifc}}$',
    'tau_ofc': r'$\tau_{\mathrm{ofc}}$',
    
    # Powers
    'P_aux': r'$P_{\mathrm{aux}}$',
    'P_aux_DT_eq': r'$P_{\mathrm{aux,DT}}$',
    'P_DDn': r'$P_{\mathrm{DD}_n}$',
    'P_DDp': r'$P_{\mathrm{DD}_p}$',
    'P_DT': r'$P_{\mathrm{DT}}$',
    'P_DT_eq': r'$P_{\mathrm{DT,eq}}$',
    
    # Tritium breeding ratios
    'TBR_DT': r'$\mathrm{TBR}_{\mathrm{DT}}$',
    'TBR_DDn': r'$\mathrm{TBR}_{\mathrm{DD}_n}$',
    
    # Inventory
    'I_target': r'$I_{\mathrm{target}}$',
    'N_ofc': r'$N_{\mathrm{ofc}}$',
    'N_ifc': r'$N_{\mathrm{ifc}}$',
    'N_stor': r'$N_{\mathrm{stor}}$',
    
    # Efficiencies and factors
    'eta_th': r'$\eta_{\mathrm{th}}$',
    'capacity_factor': r'$C_{\mathrm{f}}$',
    'cost_of_electricity': r'$C_{\mathrm{kWh}}$',
    
    # Q factors
    'Q_DD': r'$Q_{\mathrm{DD}}$',
    'Q_DT_eq': r'$Q_{\mathrm{DT,eq}}$',
    
    # Time and energy
    't_startup': r'$t_{\mathrm{startup}}$',
    'E_lost': r'$E_{\mathrm{lost}}$',
    'unrealized_profits': r'$G$',
    'unrealized_gains': r'$G$',
    
    # Tritium burn-up efficiency
    'TBE': r'$\mathrm{TBE}$',
}


def get_param_label(param_name, unit=None, use_symbol=True):
    """
    Get formatted parameter label for plotting.
    
    Parameters
    ----------
    param_name : str
        Parameter name (e.g., 'V_plasma', 't_startup')
    unit : str, optional
        Unit string (e.g., 'm³', 'keV', 's')
    use_symbol : bool, optional
        If True, use LaTeX symbol; if False, use parameter name
    
    Returns
    -------
    str
        Formatted label for use in plots
    
    Examples
    --------
    >>> get_param_label('V_plasma', 'm³')
    '$V_{\\mathrm{plasma}}$ [m³]'
    
    >>> get_param_label('T_i', 'keV', use_symbol=False)
    'T_i [keV]'
    
    >>> get_param_label('t_startup', 's')
    '$t_{\\mathrm{startup}}$ [s]'
    """
    if use_symbol and param_name in PARAM_SYMBOLS:
        label = PARAM_SYMBOLS[param_name]
    else:
        label = param_name
    
    if unit:
        label = f"{label} [{unit}]"
    
    return label


def get_param_symbol(param_name):
    """
    Get LaTeX symbol for a parameter.
    
    Parameters
    ----------
    param_name : str
        Parameter name
    
    Returns
    -------
    str
        LaTeX symbol or original name if not found
    """
    return PARAM_SYMBOLS.get(param_name, param_name)
