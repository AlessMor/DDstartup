"""
Test configuration and fixtures for DD Startup analysis tests.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
# tests/conftest.py -> dd_startup/ (one level up)
dd_startup_root = Path(__file__).parent.parent
sys.path.insert(0, str(dd_startup_root))

import pytest
import tempfile
import shutil
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_yaml_config():
    """Sample YAML configuration dictionary"""
    return {
        'analysis_type': 'T_seeded',
        'method': 'parametric',
        'vector_length': 50,
        'total_time': 31536000,  # 1 year
        'verbose': True,
        'output_dir': 'outputs',
        'n_jobs': 2,
        'chunk_size': 1000,
        'batch_size': 100
    }


@pytest.fixture
def sample_yaml_file(temp_dir, sample_yaml_config):
    """Create a sample YAML configuration file"""
    yaml_path = temp_dir / "test_config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(sample_yaml_config, f)
    return yaml_path


@pytest.fixture
def sample_yaml_file_missing_fields(temp_dir):
    """Create a YAML file with missing required fields"""
    yaml_path = temp_dir / "bad_config.yaml"
    config = {
        'vector_length': 50,
        'verbose': True
        # Missing 'analysis_type' and 'method'
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    return yaml_path


@pytest.fixture
def sample_param_module(temp_dir):
    """Create a sample parameter module"""
    param_file = temp_dir / "test_params.py"
    content = '''
"""Test parameter configuration"""
from ddstartup.utils.custom_classes import ParameterField
from ddstartup.utils.units_and_constants import u
import numpy as np

# Simple scalar fields for testing
V_plasma_field = ParameterField(
    unit=u.m**3,
    name="Plasma Volume",
    parametrization_type="scalar",
    mean=100.0,
    param_points=1
)

T_i_field = ParameterField(
    unit=u.keV,
    name="Ion Temperature",
    parametrization_type="scalar",
    mean=69.0,
    param_points=1
)

n_tot_field = ParameterField(
    unit=u.cm**-3,
    name="Total Density",
    parametrization_type="scalar",
    mean=1e14,
    param_points=1
)

tau_p_T_field = ParameterField(
    unit=u.s,
    name="Tritium Confinement Time",
    parametrization_type="scalar",
    mean=10.0,
    param_points=1
)

tau_p_He3_field = ParameterField(
    unit=u.s,
    name="Helium-3 Confinement Time",
    parametrization_type="scalar",
    mean=10.0,
    param_points=1
)

P_aux_field = ParameterField(
    unit=u.MW,
    name="Auxiliary Power",
    parametrization_type="scalar",
    mean=50.0,
    param_points=1
)

P_aux_DT_eq_field = ParameterField(
    unit=u.MW,
    name="DT Equilibrium Auxiliary Power",
    parametrization_type="scalar",
    mean=50.0,
    param_points=1
)

TBR_DT_field = ParameterField(
    unit=u.dimensionless,
    name="DT Tritium Breeding Ratio",
    parametrization_type="scalar",
    mean=1.05,
    param_points=1
)

TBR_DDn_field = ParameterField(
    unit=u.dimensionless,
    name="DD-n Tritium Breeding Ratio",
    parametrization_type="scalar",
    mean=0.5,
    param_points=1
)

tau_ifc_field = ParameterField(
    unit=u.day,
    name="In-Fuel-Cycle Time",
    parametrization_type="scalar",
    mean=30.0,
    param_points=1
)

tau_ofc_field = ParameterField(
    unit=u.day,
    name="Out-Fuel-Cycle Time",
    parametrization_type="scalar",
    mean=30.0,
    param_points=1
)

eta_th_field = ParameterField(
    unit=u.dimensionless,
    name="Thermal Efficiency",
    parametrization_type="scalar",
    mean=0.4,
    param_points=1
)

capacity_factor_field = ParameterField(
    unit=u.dimensionless,
    name="Capacity Factor",
    parametrization_type="scalar",
    mean=0.8,
    param_points=1
)

cost_of_electricity_field = ParameterField(
    unit=u.dimensionless / u.J,
    name="Cost of Electricity",
    parametrization_type="scalar",
    mean=1e-7,
    param_points=1
)

I_target_field = ParameterField(
    unit=u.kg,
    name="Target Inventory",
    parametrization_type="scalar",
    mean=10.0,
    param_points=1
)

total_time = 365 * 24 * 3600  # 1 year in seconds
'''
    with open(param_file, 'w') as f:
        f.write(content)
    return param_file


@pytest.fixture
def inputs_dir(temp_dir):
    """Create an inputs directory structure"""
    inputs = temp_dir / "inputs"
    inputs.mkdir()
    return inputs


# ============================================================================
# POSTPROCESSING FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """
    Create a sample DataFrame for postprocessing tests.
    Contains typical input and output parameters.
    """
    import pandas as pd
    import numpy as np
    
    return pd.DataFrame({
        'V_plasma': [100, 120, 140, 160, 180],
        'n_tot': [1e14, 1.5e14, 2e14, 2.5e14, 3e14],
        'T_i': [50, 60, 70, 80, 90],
        'tau_p_T': [5, 7, 9, 11, 13],
        'P_aux': [30, 40, 50, 60, 70],
        't_startup': [1e6, 1.5e6, 2e6, 2.5e6, 3e6],
        'unrealized_profits': [1e8, 1.5e8, 2e8, 2.5e8, 3e8]
    })


@pytest.fixture
def sample_dataframe_with_nan():
    """
    Create a sample DataFrame with NaN values for testing filtering.
    """
    import pandas as pd
    import numpy as np
    
    return pd.DataFrame({
        'V_plasma': [100, 120, 140, 160, 180],
        'n_tot': [1e14, 1.5e14, np.nan, 2.5e14, 3e14],
        'T_i': [50, 60, 70, 80, 90],
        't_startup': [1e6, np.nan, 2e6, 2.5e6, np.inf],
        'unrealized_profits': [1e8, 1.5e8, 2e8, np.nan, 3e8]
    })


@pytest.fixture
def sample_h5_file(temp_dir):
    """
    Create a sample HDF5 file for testing data loading.
    """
    import h5py
    import numpy as np
    
    h5_path = temp_dir / "sample_data.h5"
    
    with h5py.File(h5_path, 'w') as f:
        # 1D datasets
        f.create_dataset('V_plasma', data=[100, 120, 140, 160, 180])
        f.create_dataset('n_tot', data=[1e14, 1.5e14, 2e14, 2.5e14, 3e14])
        f.create_dataset('T_i', data=[50, 60, 70, 80, 90])
        f.create_dataset('t_startup', data=[1e6, 1.5e6, 2e6, 2.5e6, 3e6])
        f.create_dataset('unrealized_profits', data=[1e8, 1.5e8, 2e8, 2.5e8, 3e8])
        
        # 2D dataset (vector field)
        vector_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        f.create_dataset('time_vector', data=vector_data)
        
        # Parameter fields group
        param_group = f.create_group('parameter_fields')
        param_group.create_dataset('tau_p_T', data=[5, 7, 9, 11, 13])
        param_group.create_dataset('P_aux', data=[30, 40, 50, 60, 70])
    
    return h5_path


@pytest.fixture
def postprocess_config_yaml(temp_dir):
    """
    Create a sample postprocessing configuration YAML file.
    """
    import yaml
    
    config = {
        'files': ['output_1.h5', 'output_2.h5'],
        'target_variables': ['t_startup', 'unrealized_profits'],
        'input_filters': {
            'V_plasma': {'min': 100, 'max': 200},
            'n_tot': {'min': 1e14, 'max': 5e14}
        },
        'output_filters': {
            't_startup': {'min': 0, 'max': 1e8},
            'unrealized_profits': {'min': 0, 'max': 1e10}
        },
        'plot_types': ['kde', 'parcoords', 'pdf'],
        'output_dir': 'plots'
    }
    
    config_path = temp_dir / "postprocess_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path
