from scipy import constants as const
from pint import UnitRegistry
import numpy as np


#CHOOSE ONE OF THE TWO:
# OLD VERSION USING unit as UnitRegistry
#unit = UnitRegistry()
u = UnitRegistry()



if 'unit' in locals() or 'unit' in globals():
    # constants
    N_A = const.N_A * unit.mol**-1  # Avogadro's number [mol^-1]

    # Energy released by fusion reactions
    E_DDp = 4.03*unit("MeV").to("J") # [J] energy released by DDp reactions
    E_DDn = 3.46*unit("MeV").to("J") # [J] energy released by DDn reactions
    E_DT = 17.6*unit("MeV").to("J") # [J] energy released by DT reactions
    E_DHe3 = 18.0153*unit("MeV").to("J") # [J] energy released by DHe3 reactions

    # quantities related to Tritium
    molecular_weight_T = 3.016 * unit.gram / unit.mol  # Molecular weight of ATOMIC tritium [g/mol]

# NEW VERSION USING u as UnitRegistry
if 'u' in locals() or 'u' in globals():
    # Define USD currency unit
    try:
        u.define('USD = [currency]')
    except Exception:
        # If already defined, ignore
        pass

    # constants
    N_A = const.N_A * u.mol**-1  # Avogadro's number [mol^-1]

    # Energy released by fusion reactions
    E_DDp = 4.03*u("MeV").to("J") # [J] energy released by DDp reactions
    E_DDn = 3.46*u("MeV").to("J") # [J] energy released by DDn reactions
    E_DT = 17.6*u("MeV").to("J") # [J] energy released by DT reactions
    E_DHe3 = 18.0153*u("MeV").to("J") # [J] energy released by DHe3 reactions

    # quantities related to Tritium
    molecular_weight_T = 3.016 * u.gram / u.mol  # Molecular weight of ATOMIC tritium [g/mol]
    tritium_mass = molecular_weight_T/N_A
    lambda_T = np.log(2) / ((12.32 * u.year).to('s'))