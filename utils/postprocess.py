import h5py
import numpy as np
import pandas as pd

def load_h5_to_dataframe(h5_path):
    data = {}
    expected_length = None
    with h5py.File(h5_path, 'r') as f:
        # Load all datasets (1D and 2D)
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                arr = f[key][:]
                if arr.ndim == 1:
                    if expected_length is None:
                        expected_length = len(arr)
                    if len(arr) == expected_length:
                        data[key] = arr
                elif arr.ndim == 2 and arr.shape[0] == expected_length:
                    # Store each vector as a list in the DataFrame
                    data[key] = [arr[i, :] for i in range(arr.shape[0])]
        # Load parameter_fields group if present
        if 'parameter_fields' in f:
            param_group = f['parameter_fields']
            for subkey in param_group.keys():
                arr = param_group[subkey][:]
                if arr.ndim == 1 and len(arr) == expected_length:
                    data[subkey] = arr
    return pd.DataFrame(data)

def filter_finite(df, target_variable, filter_dict=None):
    mask = np.isfinite(df[target_variable])
    if filter_dict is not None:
        if filter_dict.get('min') is not None:
            mask &= df[target_variable] >= filter_dict['min']
        if filter_dict.get('max') is not None:
            mask &= df[target_variable] <= filter_dict['max']
    return df[mask].copy()

def get_input_parameters(df, target_variable, filename=None):
    DESIRED_ORDER = [
        'V_plasma','n_tot', 'T_i', 'tau_p_T','tau_p_He3','P_aux', 'P_aux_DT_eq', 'tau_ifc', 'tau_ofc', 'TBR_DT', 'TBR_DDn', 'eta_th', 'capacity_factor', 'cost_of_electricity',  'I_target',
    ]
    base_inputs = [col for col in df.columns if col in DESIRED_ORDER and col != target_variable]
    input_parameters = [p for p in DESIRED_ORDER if p in base_inputs] + [p for p in base_inputs if p not in DESIRED_ORDER]
    REMOVE_PARAMS = []
    if filename:
        if 'T_seeded' in filename:
            REMOVE_PARAMS = ['tau_p_He3', 'I_target']
        elif 'lump' in filename:
            REMOVE_PARAMS = ['tau_ifc', 'tau_ofc']
    if target_variable == 't_startup':
        REMOVE_PARAMS += ['eta_th', 'capacity_factor', 'cost_of_electricity','P_aux', 'P_aux_DT_eq']
    input_parameters = [p for p in input_parameters if p not in REMOVE_PARAMS]
    return input_parameters

def scale_target(df, target_variable):
    if target_variable == 'unrealized_gains':
        df[target_variable] = df[target_variable] / 1e6
        return df, 'M$'
    if target_variable != 't_startup':
        return df, 's'
    max_val = df[target_variable].max()
    scale = 1
    unit = "s"
    if max_val > 30*24*3600:
        scale = 1/(24*3600)
        unit = "days"
    elif max_val > 24*3600:
        scale = 1/3600
        unit = "hours"
    df[target_variable] = df[target_variable] * scale
    return df, unit

def get_discrete_colorscale(n_chunks):
    """
    Returns a Plotly-compatible discrete colorscale from green to red.
    """
    # Vibrant green to red
    base_colors = [
        '#00FF00',  # bright green
        '#7FFF00',  # chartreuse
        '#FFFF00',  # yellow
        '#FFD700',  # gold
        '#FF9900',  # orange
        '#FF4500',  # orange-red
        '#FF0000',  # bright red
    ]
    if n_chunks > len(base_colors):
        # Use matplotlib for more colors if needed
        from matplotlib import cm
        cmap = cm.get_cmap('RdYlGn_r', n_chunks)
        color_list = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r,g,b,_ in cmap(np.linspace(0,1,n_chunks))]
    else:
        color_list = base_colors[:n_chunks]
    # Build Plotly colorscale: each color repeated for its interval
    colorscale = []
    for i, color in enumerate(color_list):
        frac0 = i / n_chunks
        frac1 = (i + 1) / n_chunks
        colorscale.append([frac0, color])
        colorscale.append([frac1, color])
    return colorscale