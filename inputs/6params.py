from utils.units_and_constants import *
from utils.custom_classes import ParameterField


variable_param = 10

V_plasma_field = ParameterField(
    parametrization_type="scalar", mean=141, unit=u.m**3,
    param_points=1, name="plasma_volume"
)

T_i_field = ParameterField(
    parametrization_type="scalar", mean=14, unit=u.keV,
    param_points=1, name="T_i_field"
)

n_tot_field = ParameterField(
    parametrization_type="linear", min_val=1.3e20, max_val=2.1e20, unit=u.m**(-3),
    param_points=variable_param, name="n_tot_field"
)

tau_p_T_field = ParameterField(
    parametrization_type="linear", min_val=0.1, max_val=5, unit=u.s,
    param_points=variable_param, name="tau_p_T"
)

tau_p_He3_field = ParameterField(
    parametrization_type="scalar", mean=1, unit=u.s,
    param_points=1, name="tau_p_He3"
)

P_aux_field = ParameterField(
    parametrization_type="linear", min_val=20, max_val=100, unit=u.MW,
    param_points=variable_param, name="P_aux"
)

P_aux_DT_eq_field = ParameterField(
    parametrization_type="linear", min_val=20, max_val=100, unit=u.MW,
    param_points=variable_param, name="P_aux_DT_eq"
)
TBR_DT_field = ParameterField(
    parametrization_type="linear", min_val=1.05, max_val=1.15,
    param_points=variable_param, name="TBR_DT"
)

TBR_DDn_field = ParameterField(
    parametrization_type="scalar", mean=1,
    param_points=1, name="TBR_DDn"
)

tau_ifc_field = ParameterField(
    parametrization_type="scalar", mean=6, unit=u.h,
    param_points=1, name="tau_ifc"
)

tau_ofc_field = ParameterField(
    parametrization_type="scalar", mean=4, unit=u.h,
    param_points=1, name="tau_ofc"
)

# Economic parameters
eta_th_field = ParameterField(
    parametrization_type="scalar", mean=0.35,
    param_points=1, name="eta_th"
)

capacity_factor_field = ParameterField(
    parametrization_type="scalar", mean=0.7,
    param_points=1, name="capacity_factor"
)

cost_of_electricity_field = ParameterField(
    parametrization_type="linear", min_val=0.1, max_val=0.4, unit=1/u.kWh,
    param_points=variable_param, name="cost_of_electricity"
)

I_target_field = ParameterField(
    parametrization_type="linear", min_val=0.1, max_val=5, unit=u.kg,
    param_points=variable_param, name="I_target"
)