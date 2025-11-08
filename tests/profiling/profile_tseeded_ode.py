"""
Detailed profiling for T-seeded ODE solver bottlenecks.
"""
import time
import numpy as np
from ddstartup.physics.Tseeded_functions import solve_ode_system
from ddstartup.physics.reactivity_functions import sigmav_DD_BoschHale, sigmav_DT_BoschHale
from ddstartup.utils.units_and_constants import tritium_mass

def profile_tseeded_single_case():
    """Profile a single T-seeded case in detail."""
    
    # Typical parameters
    V_plasma = 150.0  # m³
    T_i = 17.0  # keV
    n_tot = 1.75e20  # m⁻³
    tau_p_T = 0.75  # s
    TBR_DT = 1.1
    TBR_DDn = 0.7
    tau_ifc = 10800.0  # 3 hours
    tau_ofc = 10800.0  # 3 hours
    max_simulation_time = 31536000  # 1 year
    injection_rate_max = 1e24  # atoms/s
    
    # Get reactivities
    print("Getting reactivities...")
    t0 = time.time()
    sigmav_DD_p, sigmav_DD_n, _ = sigmav_DD_BoschHale(T_i)
    sigmav_DT = sigmav_DT_BoschHale(T_i)
    reactivity_time = time.time() - t0
    print(f"  Reactivity lookup: {reactivity_time*1000:.2f} ms")
    
    # Run ODE solver
    print("\nRunning ODE solver...")
    t0 = time.time()
    result = solve_ode_system(
        V_plasma=V_plasma,
        n_tot=n_tot,
        tau_p_T=tau_p_T,
        TBR_DT=TBR_DT,
        TBR_DDn=TBR_DDn,
        tau_ifc=tau_ifc,
        tau_ofc=tau_ofc,
        sigmav_DD_p=sigmav_DD_p,
        sigmav_DD_n=sigmav_DD_n,
        sigmav_DT=sigmav_DT,
        injection_rate_max=injection_rate_max,
        max_simulation_time=max_simulation_time
    )
    ode_time = time.time() - t0
    print(f"  ODE solver: {ode_time*1000:.2f} ms")
    
    # Results
    print(f"\nResults:")
    print(f"  Success: {result.get('sol_success', False)}")
    print(f"  t_startup: {result.get('t_startup', np.inf)/3600:.1f} hours")
    print(f"  Time points: {len(result.get('t', []))}")
    print(f"  Error: {result.get('error', 'None')}")
    
    return reactivity_time, ode_time, result


def profile_tseeded_batch():
    """Profile multiple T-seeded cases to get statistics."""
    
    print("="*80)
    print("BATCH T-SEEDED PROFILING")
    print("="*80)
    
    # Parameter variations
    V_plasma_values = np.linspace(140, 160, 5)
    T_i_values = np.linspace(14, 20, 5)
    n_tot_values = np.linspace(1.5e20, 2.0e20, 3)
    
    times = []
    successes = []
    n_timepoints = []
    
    total_cases = len(V_plasma_values) * len(T_i_values) * len(n_tot_values)
    print(f"\nTesting {total_cases} parameter combinations...")
    
    case_num = 0
    for V_plasma in V_plasma_values:
        for T_i in T_i_values:
            # Get reactivities once per T_i
            sigmav_DD_p, sigmav_DD_n, _ = sigmav_DD_BoschHale(T_i)
            sigmav_DT = sigmav_DT_BoschHale(T_i)
            
            for n_tot in n_tot_values:
                case_num += 1
                
                t0 = time.time()
                result = solve_ode_system(
                    V_plasma=V_plasma,
                    n_tot=n_tot,
                    tau_p_T=0.75,
                    TBR_DT=1.1,
                    TBR_DDn=0.7,
                    tau_ifc=10800.0,
                    tau_ofc=10800.0,
                    sigmav_DD_p=sigmav_DD_p,
                    sigmav_DD_n=sigmav_DD_n,
                    sigmav_DT=sigmav_DT,
                    injection_rate_max=1e24,
                    max_simulation_time=31536000
                )
                elapsed = time.time() - t0
                
                times.append(elapsed)
                successes.append(result.get('sol_success', False))
                n_timepoints.append(len(result.get('t', [])))
                
                if case_num % 10 == 0:
                    print(f"  Completed {case_num}/{total_cases} cases...")
    
    times = np.array(times)
    successes = np.array(successes)
    n_timepoints = np.array(n_timepoints)
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total cases: {len(times)}")
    print(f"Success rate: {np.mean(successes)*100:.1f}%")
    print(f"\nTiming statistics:")
    print(f"  Mean: {np.mean(times)*1000:.1f} ms/case")
    print(f"  Median: {np.median(times)*1000:.1f} ms/case")
    print(f"  Min: {np.min(times)*1000:.1f} ms/case")
    print(f"  Max: {np.max(times)*1000:.1f} ms/case")
    print(f"  Std: {np.std(times)*1000:.1f} ms")
    
    print(f"\nODE time points:")
    print(f"  Mean: {np.mean(n_timepoints):.0f}")
    print(f"  Median: {np.median(n_timepoints):.0f}")
    print(f"  Min: {np.min(n_timepoints):.0f}")
    print(f"  Max: {np.max(n_timepoints):.0f}")
    
    # Analyze failures
    failed_indices = np.where(~successes)[0]
    if len(failed_indices) > 0:
        print(f"\nFailed cases: {len(failed_indices)}")
        print(f"  Average time points in failures: {np.mean(n_timepoints[failed_indices]):.0f}")
        print(f"  Average time for failures: {np.mean(times[failed_indices])*1000:.1f} ms")
    
    # Compare success vs failure times
    success_indices = np.where(successes)[0]
    if len(success_indices) > 0 and len(failed_indices) > 0:
        print(f"\nSuccess vs Failure timing:")
        print(f"  Success mean: {np.mean(times[success_indices])*1000:.1f} ms")
        print(f"  Failure mean: {np.mean(times[failed_indices])*1000:.1f} ms")
        print(f"  Ratio: {np.mean(times[failed_indices])/np.mean(times[success_indices]):.2f}x")


if __name__ == '__main__':
    print("="*80)
    print("T-SEEDED ODE SOLVER PROFILING")
    print("="*80)
    
    print("\n1. Single case detailed profiling:")
    print("-"*80)
    profile_tseeded_single_case()
    
    print("\n\n2. Batch profiling:")
    print("-"*80)
    profile_tseeded_batch()
