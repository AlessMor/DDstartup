# Time-Dependent Auxiliary Heating Power (P_aux) Implementation

## Overview

This document describes the implementation of a physically sound, time-dependent auxiliary heating power model in the manual T-seeded verification notebook.

## Physical Model

### Energy Balance Equation

The auxiliary heating power required at any time t is determined by the plasma energy balance:

```
P_aux(t) = P_in(t) - P_charged(t) - P_ohmic(t)
```

### Components

#### 1. Input Power (P_in)

The power required to maintain the stored plasma energy against confinement losses:

```
P_in(t) = W_stored(t) / τ_E
```

where:
- **W_stored(t)** = Total thermal energy stored in the plasma [J]
- **τ_E** = Energy confinement time [s]

#### 2. Stored Energy (W_stored)

Total thermal energy in the plasma:

```
W_stored(t) = (3/2) * k_B * (n_e * T_e + n_i * T_i) * V_plasma
```

For a quasi-neutral, fully ionized plasma:
- n_e = n_i = n_D(t) + n_T(t)
- Typically T_e ≈ T_i (can be refined)

In practical units (with temperature in keV):
```
W_stored(t) = (3/2) * [(n_e * T_e + n_i * T_i) * keV_to_J] * V_plasma
```

where keV_to_J = 1.60218e-16 J/keV

#### 3. Charged Particle Heating (P_charged)

Power deposited into the plasma by charged fusion products:

##### DT Fusion: D + T → α(3.5 MeV) + n(14.1 MeV)
```
P_alpha = n_D(t) * n_T(t) * <σv>_DT * V_plasma * E_alpha
```
- E_alpha = 3.5 MeV = 3.5e6 * 1.60218e-19 J

##### DD(p) Fusion: D + D → T(1.01 MeV) + p(3.02 MeV)
```
P_T+p = (1/2) * n_D(t)^2 * <σv>_DD_p * V_plasma * (E_T + E_p)
```
- E_T = 1.01 MeV, E_p = 3.02 MeV
- Total charged energy = 4.03 MeV

##### DD(n) Fusion: D + D → ³He(0.82 MeV) + n(2.45 MeV)
```
P_He3 = (1/2) * n_D(t)^2 * <σv>_DD_n * V_plasma * E_He3
```
- E_He3 = 0.82 MeV

**Total charged particle heating:**
```
P_charged(t) = P_alpha(t) + P_T+p(t) + P_He3(t)
```

**Note**: Neutron energy is NOT included in P_charged as neutrons escape the plasma without depositing energy.

#### 4. Ohmic Heating (P_ohmic)

Resistive heating from plasma current. For fusion plasmas at high temperature, this is typically small and can be approximated as:
- Constant value
- Zero (for simplicity)
- Function of plasma parameters: P_ohmic ~ I_p² * R / (a² * T_e^1.5)

## Implementation in the Notebook

### New Cells Added

1. **Block 1b**: Time-dependent P_aux parameters
   - Energy confinement time τ_E
   - Electron temperature T_e
   - Ohmic power P_ohmic

2. **Function definition**: `calculate_P_aux_time_dependent()`
   - Inputs: n_D, n_T, T_i, T_e, V_plasma, τ_E, P_ohmic, reaction rates
   - Outputs: P_aux, W_stored, P_in, P_charged

3. **Block 1c**: Modified ODE system
   - Includes P_aux calculation for diagnostic purposes
   - ODEs remain unchanged (P_aux is not an input parameter)

4. **Block 2b**: P_aux evolution calculation
   - Computes P_aux(t) along the solution trajectory
   - Detects ignition (when P_aux < 0)
   - Visualizes results

5. **Energy balance verification**
   - Checks: P_aux + P_charged + P_ohmic = P_in
   - Computes time-dependent Q factors

## Key Results

### At t = 0 (Pure Deuterium)
- High P_aux required
- Only DD reactions (low fusion power)
- Most power comes from auxiliary heating

### During Transition
- P_aux decreases as tritium builds up
- P_charged increases due to DT reactions
- May reach ignition (P_aux < 0) before DT equilibrium

### At t = t_startup (DT Equilibrium)
- Lower P_aux (or negative if ignited)
- Significant DT fusion power
- P_charged dominates the energy balance

## Energy Confinement Time (τ_E)

The energy confinement time is critical for determining P_aux. Several options:

### 1. Constant Value (Current Implementation)
```python
tau_E = 2.0  # seconds
```

### 2. ITER IPB98(y,2) Scaling Law
```
τ_E = H * 0.0562 * I_p^0.93 * B_T^0.15 * P^(-0.69) * 
      n_e^0.41 * M^0.19 * R^1.97 * ε^0.58 * κ^0.78
```

where:
- H = H-factor (typically 1.0-1.2 for ITER)
- I_p = Plasma current [MA]
- B_T = Toroidal magnetic field [T]
- P = Total heating power [MW]
- n_e = Line-averaged electron density [10^19 m^-3]
- M = Average ion mass [AMU]
- R = Major radius [m]
- ε = Inverse aspect ratio (a/R)
- κ = Elongation

### 3. Measurement/Design Specification
- Use experimental measurements if available
- Or specify as a design requirement

## Extensions and Refinements

### 1. Temperature Evolution
Currently assumes constant T_i and T_e. Could extend to:
```python
def ode_system_with_temperature(t, y):
    # y = [N_ofc, N_ifc, N_st, n_T, T_i, T_e]
    ...
    dT_dt = (P_aux + P_charged + P_ohmic - P_radiation - P_loss) / W_stored
    ...
```

### 2. Radiation Losses
Add bremsstrahlung and synchrotron radiation:
```python
P_bremsstrahlung = C_brems * n_e^2 * sqrt(T_e) * V_plasma
P_synchrotron = C_sync * n_e * T_e^2 * B^2 * V_plasma
```

### 3. Profile Effects
Replace uniform density/temperature with profiles:
```python
W_stored = integral[(3/2) * n(r) * T(r) * dV]
```

### 4. Impurity Effects
Include impurity contribution to stored energy:
```python
n_e_effective = n_D + n_T + sum(Z_i * n_i)  # impurities
```

### 5. Time-Dependent τ_E
Use scaling law with time-dependent parameters:
```python
tau_E(t) = f(I_p(t), P_total(t), n_e(t), ...)
```

## Validation and Checks

### Energy Balance Verification
The code checks:
```
P_aux + P_charged + P_ohmic ≈ P_in
```

This should be satisfied to within numerical precision (~1e-6 MW).

### Physical Constraints
- P_aux > 0 during startup (unless ignited)
- W_stored increases with n and T
- P_charged increases with fusion reaction rates

### Q Factor
Instantaneous fusion gain:
```
Q(t) = P_fusion(t) / P_aux(t)
```

Average fusion gain over startup:
```
Q_avg = <P_fusion> / <P_aux>
```

## Usage Example

```python
# Calculate P_aux at a given state
P_aux, W_stored, P_in, P_charged = calculate_P_aux_time_dependent(
    n_D=1.0e20,          # m^-3
    n_T=0.5e20,          # m^-3
    T_i=14.0,            # keV
    T_e=14.0,            # keV
    V_plasma=150.0,      # m^3
    tau_E=2.0,           # s
    P_ohmic=0.0,         # W
    sigmav_DD_p=sigmav_DD_p,
    sigmav_DD_n=sigmav_DD_n,
    sigmav_DT=sigmav_DT
)

print(f"Required P_aux = {P_aux/1e6:.2f} MW")
```

## References

1. J. Wesson, "Tokamak Physics", Oxford University Press (2004)
2. ITER Physics Basis, Nuclear Fusion 39 (1999) 2137
3. Bosch & Hale, "Improved formulas for fusion cross-sections and thermal reactivities", Nuclear Fusion 32 (1992) 611

## Notes

- This implementation provides the framework for time-dependent P_aux
- The value of τ_E is critical and should be chosen carefully
- Negative P_aux indicates plasma ignition (self-heating)
- The model can be refined by adding more physics (radiation, profiles, etc.)
