from scipy import constants as const
from pint import UnitRegistry
unit = UnitRegistry()

# constants
N_A = const.N_A * unit.mol**-1  # Avogadro's number [mol^-1]

# Energy released by fusion reactions
E_DDp = 4.03*unit("MeV").to("J") # [J] energy released by DDp reactions
E_DDn = 3.46*unit("MeV").to("J") # [J] energy released by DDn reactions
E_DT = 17.6*unit("MeV").to("J") # [J] energy released by DT reactions
E_DHe3 = 18.0153*unit("MeV").to("J") # [J] energy released by DHe3 reactions

# quantities related to Tritium
molecular_weight_T = 3.016 * unit.gram / unit.mol  # Molecular weight of ATOMIC tritium [g/mol]