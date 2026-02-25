from scipy import constants as const
from pint import UnitRegistry
import numpy as np

u = UnitRegistry()

# Define USD currency unit
try:
    u.define('USD = [currency]')
except Exception:
    # If already defined, ignore
    pass

# constants
N_A = (const.N_A * u.mol**-1).to('1/mol').magnitude  # Avogadro's number [mol^-1]

# Energy released by fusion reactions
E_DDp = 4.03*u("MeV").to("J").magnitude # [J] energy released by DDp reactions
E_DDn = 3.46*u("MeV").to("J").magnitude # [J] energy released by DDn reactions
E_DT = 17.6*u("MeV").to("J").magnitude # [J] energy released by DT reactions
E_DHe3 = 18.0153*u("MeV").to("J").magnitude # [J] energy released by DHe3 reactions

# quantities related to Tritium
molecular_weight_T = (3.016 * u.gram / u.mol).to('kg/mol').magnitude  # Molecular weight of ATOMIC tritium [g/mol]
tritium_mass = (molecular_weight_T/N_A)  # Mass of a tritium atom [kg]
lambda_T = (np.log(2) / ((12.32 * u.year).to('s'))).magnitude

# Atomic masses (kg) for multispecies model
amu_kg = 1.66053906660e-27
species_mass = {
    "D": 2.014 * amu_kg,
    "T": 3.016 * amu_kg,
    "He3": 3.016 * amu_kg,
    "He4": 4.002602 * amu_kg,
}
