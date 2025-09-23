import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import chaospy as cp
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.config import *
from utils.physics import sigmav_DT_BoschHale, sigmav_DD_BoschHale

# Constants
tritium_mass = 3.016049268e-3 / 6.022e23  # kg per tritium atom
lambda_T = np.log(2) / (12.32 * 365*24*3600)  # [1/s]
MeV_to_J = 1.6022e-13
E_DDp = 4.03*MeV_to_J  # [J] energy released by DDp reactions
E_DDn = 3.46*MeV_to_J  # [J] energy released by DDn reactions
E_DT = 17.6*MeV_to_J   # [J] energy released by DT reactions

def solve_for_sample(sample_idx, sample):
    # Unpack sample parameters
    temp_kev, tau_p_T, TBRDT, TBRDD, tau_ifc, tau_ofc, c_dollar_per_kWh = sample
    
    # Fixed parameters
    Vp = 150
    ntot = 1.7e20  # total density
    eta_th = 0.3
    P_aux = 100e6  # W (100 MW)
    # tau_ofc = 3600*24  # seconds (1 day)
    # tau_ifc = 3600*12  # seconds (12 hours)
    C_f = 0.7  # capacity factor

    # Calculate cross sections
    sigmavDT = sigmav_DT_BoschHale(np.array([temp_kev]))[0]
    sigmavDD_results = sigmav_DD_BoschHale(np.array([temp_kev]))
    sigmavDDp = sigmavDD_results[1][0]
    sigmavDDn = sigmavDD_results[2][0]

    # Calculate maximum injection rate
    injection_rate_max = (ntot/2/tau_p_T*Vp + 0.25*ntot**2*sigmavDT*Vp - 0.25/2*ntot**2*sigmavDDp*Vp)

    def tritium_inventory_odes(t, y):
        N_ofc, N_ifc, N_st, nT = y
        if N_st < 0.001/tritium_mass:
            injection_rate = 0.0
        else:
            injection_rate = min((N_ifc/tau_ifc - lambda_T*N_st), injection_rate_max)
        
        nD = (ntot - nT)
        dN_ofc_dt = TBRDT*nD*nT*sigmavDT*Vp + TBRDD*0.5*nD**2*sigmavDDn*Vp - N_ofc / tau_ofc - N_ofc*lambda_T
        dN_ifc_dt = N_ofc / tau_ofc - N_ifc / tau_ifc - lambda_T * N_ifc + nT / tau_p_T * Vp
        dN_stor_dt = N_ifc / tau_ifc - lambda_T * N_st - max(injection_rate,0)
        dnT_dt = injection_rate/Vp + 0.5*nD**2*sigmavDDp - nT/tau_p_T - nD*nT*sigmavDT
        return [dN_ofc_dt, dN_ifc_dt, dN_stor_dt, dnT_dt]

    y0 = [0.0, 0.0, 0.0, 0.0] 
    years_to_seconds = 365 * 24 * 3600
    t_span = (0, 10*years_to_seconds)

    def DT_reached(t, y):
        return y[3] - (0.5 * ntot)
    def NEGATIVE(t, y):
        return min(y[0]+100, y[1]+100, y[2]+100, y[3]+100)
    DT_reached.terminal = True
    NEGATIVE.terminal = True

    try:
        sol = solve_ivp(tritium_inventory_odes, t_span, y0, method='BDF', 
                        events=[DT_reached, NEGATIVE], dense_output=False)
        
        if len(sol.t_events) > 0 and sol.t_events[0].size > 0:
            startup_time = sol.t_events[0][0]
        else:
            startup_time = np.nan
        
        # Check for negative event
        if len(sol.t_events) > 1 and sol.t_events[1].size > 0:
            return {
                'sample_idx': sample_idx,
                'success': False,
                'startup_time': np.nan,
                'Dollar_lost': np.nan
            }
            
        n_D = ntot - sol.y[3][-1]  # deuterium density
        n_T = sol.y[3][-1]
        
        # Calculate power and energy
        rate_DT = n_D * n_T * sigmavDT  # [1/s]
        rate_DDp = 0.5 * n_D**2 * sigmavDDp  # [1/s]
        rate_DDn = 0.5 * n_D**2 * sigmavDDn  # [1/s]
        
        P_fusion_DT = rate_DT * E_DT  # W
        P_fusion_DD = rate_DDp * E_DDp + rate_DDn * E_DDn  # W
        
        P_e_net_DT = eta_th * P_fusion_DT - P_aux  # W
        P_e_net_DD = eta_th * P_fusion_DD - P_aux  # W
        
        E_lost = (P_e_net_DT - P_e_net_DD) * startup_time * C_f  # J
        Ws_into_kWh = 1e-3 / 3600  # convert Ws to kWh
        E_lost_kWh = E_lost * Ws_into_kWh  # kWh
        Dollar_lost = c_dollar_per_kWh * E_lost_kWh  # unrealized gain in dollars
        
        return {
            'sample_idx': sample_idx,
            'success': sol.success,
            'startup_time': startup_time,
            'Dollar_lost': Dollar_lost
        }
    
    except Exception as e:
        return {
            'sample_idx': sample_idx,
            'success': False,
            'startup_time': np.nan,
            'Dollar_lost': np.nan
        }

def main():
    # Define parameter distributions
    input_data = [
        # n_tot_field.data.to('1/m^3').magnitude,
        T_i_field.data.to('keV').magnitude,
        tau_p_T_field.data.to('s').magnitude,
        TBR_DT_field.data.to_base_units().magnitude,
        TBR_DDn_field.data.to_base_units().magnitude,
        tau_ifc_field.data.to('s').magnitude,
        tau_ofc_field.data.to('s').magnitude,
        Cost_per_kWh_field.data.to('1/kWh').magnitude
    ]
    
    distr_params = {
        "T_i": cp.Uniform(input_data[0][0], input_data[0][-1]),
        "tau_p_T": cp.Uniform(input_data[1][0], input_data[1][-1]),
        "TBR_DT": cp.Uniform(input_data[2][0], input_data[2][-1]),
        "TBR_DDn": cp.Uniform(input_data[3][0], input_data[3][-1]),
        "tau_ifc": cp.Uniform(input_data[4][0], input_data[4][-1]),
        "tau_ofc": cp.Uniform(input_data[5][0], input_data[5][-1]),
        "Cost_per_kWh": cp.Uniform(input_data[6][0], input_data[6][-1])
    }
    
    # Create joint distribution
    joint_distribution = cp.J(*distr_params.values())
    
    # Generate samples
    number_of_samples = 50000
    print(f"Generating {number_of_samples} samples...")
    samples = joint_distribution.sample(number_of_samples, rule="L")
    print(f"Sample generation complete. Sample shape: {samples.shape}")
    
    # Initialize arrays for results
    startup_times = np.full(number_of_samples, np.nan)
    dollar_lost = np.full(number_of_samples, np.nan)
    success_flags = np.zeros(number_of_samples, dtype=bool)
    
    # Process in chunks for better memory management
    chunk_size = 1000
    n_chunks = (number_of_samples + chunk_size - 1) // chunk_size
    
    print(f"Processing {number_of_samples} samples in {n_chunks} chunks...")
    
    # Progress bars
    chunk_pbar = tqdm(total=n_chunks, desc="Processing chunks", unit="chunk", position=0)
    overall_pbar = tqdm(total=number_of_samples, desc="Overall progress", unit="sample", position=1)
    
    processed_count = 0
    successful_count = 0
    
    for chunk_start in range(0, number_of_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, number_of_samples)
        chunk_indices = np.arange(chunk_start, chunk_end)
        current_chunk_size = len(chunk_indices)
        
        # Get samples for this chunk
        chunk_samples = samples[:, chunk_start:chunk_end].T  # Transpose to get samples as rows
        
        # Process chunk in parallel
        n_jobs = -1  # Use all available cores
        chunk_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(solve_for_sample)(idx, sample) 
            for idx, sample in zip(chunk_indices, chunk_samples)
        )
        
        # Process results and store in arrays
        chunk_successes = 0
        for result in chunk_results:
            idx = result['sample_idx']
            success = result.get('success', False)
            
            success_flags[idx] = success
            startup_times[idx] = result.get('startup_time', np.nan)
            dollar_lost[idx] = result.get('Dollar_lost', np.nan)
            
            # Count successes
            if success and np.isfinite(result.get('startup_time', np.nan)):
                successful_count += 1
                chunk_successes += 1
        
        processed_count += current_chunk_size
        
        # Update progress bars
        chunk_pbar.update(1)
        overall_pbar.update(current_chunk_size)
        
        # Update progress descriptions
        success_rate = (successful_count / processed_count) * 100
        chunk_pbar.set_postfix({"Success Rate": f"{success_rate:.1f}%"})
        overall_pbar.set_postfix({"Current Chunk": f"{chunk_successes}/{current_chunk_size}"})
    
    # Close progress bars
    chunk_pbar.close()
    overall_pbar.close()
    
    print(f"\n✅ Computation complete!")
    print(f"Successful runs: {successful_count}/{number_of_samples} ({success_rate:.1f}%)")
    
    # Convert to days for easier interpretation
    startup_times_days = startup_times / (24 * 3600)
    
    # Now perform PCE analysis on the results
    print("\nPerforming PCE analysis...")
    
    # Filter valid results for PCE
    valid_mask = success_flags & np.isfinite(dollar_lost)
    valid_dollars = dollar_lost[valid_mask]
    valid_samples = samples[:, valid_mask]
    
    print(f"Valid samples for PCE: {np.sum(valid_mask)}/{number_of_samples}")
    
    if np.sum(valid_mask) > 10:  # Only proceed if we have enough valid samples
        # Generate polynomial basis
        order = 3
        poly_expansion = cp.generate_expansion(order, joint_distribution)
        
        # Fit PCE model
        print("Fitting PCE model...")
        pce_model = cp.fit_regression(poly_expansion, valid_samples, valid_dollars)
        
        # Compute Sobol indices
        print("Computing Sobol indices...")
        sobol_first = cp.Sens_m(pce_model, joint_distribution)
        sobol_total = cp.Sens_t(pce_model, joint_distribution)
        
        print("\nSensitivity Analysis Results:")
        print("Parameter\t\tFirst-order\tTotal-order")
        print("------------------------------------------------")
        for i, param_name in enumerate(distr_params.keys()):
            print(f"{param_name:15s}\t{sobol_first[i]:.6f}\t{sobol_total[i]:.6f}")
        
        # Visualize Sobol indices
        plt.figure(figsize=(10, 6))
        x = np.arange(len(distr_params))
        width = 0.35
        
        plt.bar(x - width/2, sobol_first, width, label='First-order')
        plt.bar(x + width/2, sobol_total, width, label='Total-order')
        
        plt.xlabel('Parameters')
        plt.ylabel('Sobol Indices')
        plt.title('Sensitivity Analysis')
        plt.xticks(x, distr_params.keys())
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough valid samples for PCE analysis.")
    
    # Return the results for further analysis if needed
    return {
        'samples': samples,
        'startup_times': startup_times,
        'startup_times_days': startup_times_days,
        'dollar_lost': dollar_lost,
        'success_flags': success_flags,
        'param_names': list(distr_params.keys())
    }

if __name__ == "__main__":
    results = main()