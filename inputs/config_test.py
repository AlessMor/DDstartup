from utils.units_and_constants import *
from utils.custom_classes import ParameterField

total_time = 10*365*24*3600  # total operation time [s] - 10 years

param_points = 2

V_plasma_field = ParameterField(
    parametrization_type="normal", mean=150, std=15, unit=u.m**3, 
    param_points=1, name="plasma_volume"
)

T_i_field = ParameterField(
    parametrization_type="linear", min_val=14, max_val=20, unit=u.keV,
    param_points=1, name="T_i_field"
)

n_tot_field = ParameterField(
    parametrization_type="linear", min_val=1.3e20, max_val=2.1e20, unit=u.m**(-3),
    param_points=1, name="n_tot_field"
)

tau_p_T_field = ParameterField(
    parametrization_type="linear", min_val = 0.1, max_val=1, unit=u.s,
    param_points=param_points, name="tau_p_T"
)

tau_p_He3_field = ParameterField(
    parametrization_type="normal", mean=1, std=0.5, unit=u.s,
    param_points=param_points, name="tau_p_He3"
)

P_aux_field = ParameterField(
    parametrization_type="linear", min_val=20, max_val=100, unit=u.MW,
    param_points=1, name="P_aux"
)

P_lost_rad_field = ParameterField(
    parametrization_type="linear", min_val=0, max_val=20, unit=u.MW,
    param_points=1, name="P_lost_rad"
)

P_aux_all_DT_field = ParameterField(
    parametrization_type="linear", min_val=20, max_val=100, unit=u.MW,
    param_points=1, name="P_aux_all_DT"
)

P_lost_rad_all_DT_field = ParameterField(
    parametrization_type="linear", min_val=0, max_val=20, unit=u.MW,
    param_points=1, name="P_lost_rad_all_DT"
)

TBR_DT_field = ParameterField(
    parametrization_type="linear", min_val=1.05, max_val=1.15,
    param_points=param_points, name="TBR_DT"
)

TBR_DDn_field = ParameterField(
    parametrization_type="linear", min_val=0.5, max_val=0.9,
    param_points=param_points, name="TBR_DDn"
)

tau_ifc_field = ParameterField(
    parametrization_type="linear", min_val=1, max_val=12, unit=u.h,
    param_points=param_points, name="tau_ifc"
)

tau_ofc_field = ParameterField(
    parametrization_type="linear", min_val=1, max_val=24, unit=u.h,
    param_points=param_points, name="tau_ofc"
)

# Economic parameters
eta_th_field = ParameterField(
    parametrization_type="linear", min_val=0.3, max_val=0.4,
    param_points=param_points, name="eta_th"
)

plant_avail_field = ParameterField(
    parametrization_type="linear", min_val=0.5, max_val=0.9,
    param_points=param_points, name="plant_availability"
)

Cost_per_kWh_field = ParameterField(
    parametrization_type="normal", mean=0.25, std=0.1, unit=1/u.kWh,
    param_points=param_points, name="Cost_per_kWh"
)

I_target_field = ParameterField(
    parametrization_type="linear", min_val=0.5, max_val=2, unit=u.kg,
    param_points=param_points, name="I_target"
)