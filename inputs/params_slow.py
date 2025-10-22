from utils.units_and_constants import *
from utils.custom_classes import ParameterField

V_plasma_field = ParameterField(
    parametrization_type="linear", min_val = 100, max_val = 2000, unit=u.m**3,
    param_points=10, name="plasma_volume"
)

T_i_field = ParameterField(
    parametrization_type="vector", vector=[10, 15, 20], unit=u.keV,
    param_points=3, name="T_i_field"
)

n_tot_field = ParameterField(
    parametrization_type="linear", min_val=0.7e20, max_val=1.5e20, unit=u.m**(-3),
    param_points=10, name="n_tot_field"
)

tau_p_T_field = ParameterField(
    parametrization_type="vector", vector=[0.1, 1, 5], unit=u.s,
    param_points=3, name="tau_p_T"
)

tau_p_He3_field = ParameterField(
    parametrization_type="scalar", mean=1.0, unit=u.s,
    param_points=1, name="tau_p_He3"
)

P_aux_field = ParameterField(
    parametrization_type="linear", min_val=20, max_val=50, unit=u.MW,
    param_points=2, name="P_aux"
)

P_aux_DT_eq_field = ParameterField(
    parametrization_type="linear", min_val=20, max_val=50, unit=u.MW,
    param_points=2, name="P_aux_DT_eq"
)
TBR_DT_field = ParameterField(
    parametrization_type="linear", min_val=1.05, max_val=1.3,
    param_points=10, name="TBR_DT"
)

TBR_DDn_field = ParameterField(
    parametrization_type="linear", min_val=0.5, max_val=1,
    param_points=5, name="TBR_DDn"
)

tau_ifc_field = ParameterField(
    parametrization_type="linear", min_val=1, max_val=110, unit=u.h,
    param_points=10, name="tau_ifc"
)

tau_ofc_field = ParameterField(
    parametrization_type="vector", vector=[1, 12, 24], unit=u.h,
    param_points=3, name="tau_ofc"
)

# Economic parameters
eta_th_field = ParameterField(
    parametrization_type="linear", min_val=0.3, max_val=0.4,
    param_points=2, name="eta_th"
)

capacity_factor_field = ParameterField(
    parametrization_type="linear", min_val=0.5, max_val=0.9,
    param_points=3, name="capacity_factor"
)

cost_of_electricity_field = ParameterField(
    parametrization_type="linear", min_val=0.1, max_val=0.5, unit=1/u.kWh,
    param_points=10, name="cost_of_electricity"
)

I_target_field = ParameterField(
    parametrization_type="linear", min_val=0.1, max_val=5, unit=u.kg,
    param_points=10, name="I_target"
)