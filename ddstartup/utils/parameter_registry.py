"""
Centralized parameter management for DD startup analyses.

This module provides a single source of truth for all parameters used across
different analysis types (lump, T_seeded). It consolidates:
- Parameter schema (roles, analysis types, units, descriptions)
- Parameter symbols (LaTeX formatting for plots)
- Parameter registry (query interface)

All parameter definitions are in PARAMETER_SCHEMA, eliminating the need
for separate YAML files and reducing redundancy.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Iterable, Tuple

from .units_and_constants import u


# ============================================================================
# PARAMETER SCHEMA - Single source of truth
# ============================================================================

PARAMETER_SCHEMA = {
    # INPUT PARAMETERS
    'V_plasma': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'm³',
        'symbol': r'$V_{\mathrm{plasma}}$',
        'description': 'Plasma volume'
    },
    'T_i': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'keV',
        'symbol': r'$T_i$',
        'description': 'Ion temperature'
    },
    'n_tot': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'm⁻³',
        'symbol': r'$n_{\mathrm{tot}}$',
        'description': 'Total ion density'
    },
    'tau_p_T': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 's',
        'symbol': r'$\tau_{p,T}$',
        'description': 'Tritium particle confinement time'
    },
    'tau_p_He3': {
        'role': 'input',
        'analysis_types': ['lump'],
        'unit': 's',
        'symbol': r'$\tau_{p,\mathrm{He3}}$',
        'description': 'Helium-3 particle confinement time'
    },
    'P_aux': {
        'role': 'flexible',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'W',
        'symbol': r'$P_{\mathrm{aux}}$',
        'description': 'Auxiliary heating power during DD startup phase',
        'computed_when_null': True,
        'vector': True,
        'vector_length': 5
    },
    'P_aux_DT_eq': {
        'role': 'flexible',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'W',
        'symbol': r'$P_{\mathrm{aux,DT}}$',
        'description': 'Auxiliary heating power at DT equilibrium',
        'computed_when_null': True,
        'vector': True,
        'vector_length': 5
    },
    'TBR_DT': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'dimensionless',
        'symbol': r'$\mathrm{TBR}_{\mathrm{DT}}$',
        'description': 'Tritium breeding ratio from DT reactions'
    },
    'TBR_DDn': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'dimensionless',
        'symbol': r'$\mathrm{TBR}_{\mathrm{DD}_n}$',
        'description': 'Tritium breeding ratio from DD neutron branch'
    },
    'tau_ifc': {
        'role': 'input',
        'analysis_types': ['T_seeded'],
        'unit': 's',
        'symbol': r'$\tau_{\mathrm{ifc}}$',
        'description': 'Internal fuel cycle time constant'
    },
    'tau_ofc': {
        'role': 'input',
        'analysis_types': ['T_seeded'],
        'unit': 's',
        'symbol': r'$\tau_{\mathrm{ofc}}$',
        'description': 'External fuel cycle time constant'
    },
    'I_target': {
        'role': 'input',
        'analysis_types': ['lump'],
        'unit': 'kg',
        'symbol': r'$I_{\mathrm{target}}$',
        'description': 'Target tritium inventory'
    },
    'eta_th': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'dimensionless',
        'symbol': r'$\eta_{\mathrm{th}}$',
        'description': 'Thermal conversion efficiency'
    },
    'capacity_factor': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'dimensionless',
        'symbol': r'$C_{\mathrm{f}}$',
        'description': 'Plant capacity factor'
    },
    'price_of_electricity': {
        'role': 'input',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': '$/J',
        'symbol': r'$C_{\mathrm{kWh}}$',
        'description': 'Price of electricity',
        'aliases': ['price_of_electricity']
    },
    
    # OUTPUT PARAMETERS
    'n_T': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'm⁻³',
        'symbol': r'$n_T$',
        'description': 'Tritium ion density',
        'vector': True
    },
    'n_D': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'm⁻³',
        'symbol': r'$n_D$',
        'description': 'Deuterium ion density',
        'vector': True
    },
    'n_He3': {
        'role': 'output',
        'analysis_types': ['lump'],
        'unit': 'm⁻³',
        'symbol': r'$n_{\mathrm{He3}}$',
        'description': 'Helium-3 ion density'
    },
    'N_ofc': {
        'role': 'output',
        'analysis_types': ['T_seeded'],
        'unit': 'kg',
        'symbol': r'$N_{\mathrm{ofc}}$',
        'description': 'Tritium inventory in outer fuel cycle',
        'vector': True
    },
    'N_ifc': {
        'role': 'output',
        'analysis_types': ['T_seeded'],
        'unit': 'kg',
        'symbol': r'$N_{\mathrm{ifc}}$',
        'description': 'Tritium inventory in inner fuel cycle',
        'vector': True
    },
    'N_stor': {
        'role': 'output',
        'analysis_types': ['T_seeded'],
        'unit': 'kg',
        'symbol': r'$N_{\mathrm{stor}}$',
        'description': 'Tritium inventory in storage',
        'vector': True
    },
    'P_DDn': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'W',
        'symbol': r'$P_{\mathrm{DD}_n}$',
        'description': 'Fusion power from DD neutron branch',
        'vector': True
    },
    'P_DDp': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'W',
        'symbol': r'$P_{\mathrm{DD}_p}$',
        'description': 'Fusion power from DD proton branch',
        'vector': True
    },
    'P_DT': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'W',
        'symbol': r'$P_{\mathrm{DT}}$',
        'description': 'Fusion power from DT reactions',
        'vector': True
    },
    'P_DT_eq': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'W',
        'symbol': r'$P_{\mathrm{DT,eq}}$',
        'description': 'Fusion power at DT equilibrium'
    },
    't_startup': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 's',
        'symbol': r'$t_{\mathrm{startup}}$',
        'description': 'Time to reach target conditions'
    },
    'Q_DD': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'dimensionless',
        'symbol': r'$Q_{\mathrm{DD}}$',
        'description': 'Fusion gain during DD startup phase'
    },
    'Q_DT_eq': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'dimensionless',
        'symbol': r'$Q_{\mathrm{DT,eq}}$',
        'description': 'Fusion gain at DT equilibrium'
    },
    'TBE': {
        'role': 'output',
        'analysis_types': ['T_seeded'],
        'unit': 'dimensionless',
        'symbol': r'$\mathrm{TBE}$',
        'description': 'Tritium breeding efficiency',
        'vector': True
    },
    'E_lost': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'J',
        'symbol': r'$E_{\mathrm{lost}}$',
        'description': 'Energy lost during startup'
    },
    'unrealized_profits': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'J',
        'symbol': 'G',
        'description': 'Unrealized profits during startup',
        'aliases': ['unrealized_gains']
    },
    'sol_success': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'boolean',
        'symbol': 'Success',
        'description': 'Whether the solver converged successfully'
    },
    'linear_index': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'dimensionless',
        'symbol': 'Index',
        'description': 'Linear index of parameter combination'
    },
    'error': {
        'role': 'output',
        'analysis_types': ['lump', 'T_seeded'],
        'unit': 'string',
        'symbol': 'Error',
        'description': 'Error message if computation failed'
    }
}


# ============================================================================
# PARAMETER REGISTRY - Query interface
# ============================================================================

class ParameterRegistry:
    """
    Central registry for querying parameter properties.
    
    Provides methods to filter parameters by role, analysis type, and other criteria.
    """
    
    def __init__(self):
        """Initialize with PARAMETER_SCHEMA."""
        self.parameters = PARAMETER_SCHEMA
        # Build alias map for quick resolution
        self._alias_to_name = {}
        for name, props in self.parameters.items():
            for alias in props.get('aliases', []) or []:
                self._alias_to_name[alias] = name
    
    def get_parameter_names(self, role: Optional[str] = None, 
                          analysis_type: Optional[str] = None,
                          include_flexible: bool = True) -> List[str]:
        """
        Get list of parameter names matching criteria.
        
        Args:
            role: Filter by role ('input', 'output', or 'flexible'). None = all roles.
            analysis_type: Filter by analysis type ('lump' or 'T_seeded'). None = all types.
            include_flexible: If role='input', whether to include flexible parameters.
            
        Returns:
            List of parameter names matching the criteria
        """
        results = []
        
        for name, props in self.parameters.items():
            # Check role
            if role is not None:
                param_role = props['role']
                if param_role != role:
                    # For input queries, include flexible parameters if requested
                    if role == 'input' and include_flexible and param_role == 'flexible':
                        pass  # Include it
                    else:
                        continue
            
            # Check analysis type
            if analysis_type is not None:
                if analysis_type not in props['analysis_types']:
                    continue
            
            results.append(name)
        
        return results
    
    def get_input_names(self, analysis_type: Optional[str] = None) -> List[str]:
        """Get input parameter names for an analysis type."""
        return self.get_parameter_names(role='input', analysis_type=analysis_type, 
                                       include_flexible=True)
    
    def get_output_names(self, analysis_type: Optional[str] = None) -> List[str]:
        """Get output parameter names for an analysis type."""
        return self.get_parameter_names(role='output', analysis_type=analysis_type)
    
    def get_all_field_names(self, analysis_type: Optional[str] = None) -> List[str]:
        """Get all parameter names (inputs + outputs) for an analysis type."""
        inputs = self.get_input_names(analysis_type)
        outputs = self.get_output_names(analysis_type)
        # Remove duplicates while preserving order
        return list(dict.fromkeys(inputs + outputs))
    
    def get_vector_fields(self, analysis_type: Optional[str] = None) -> List[str]:
        """Get names of parameters that are vectors (arrays) in outputs."""
        results = []
        for name, props in self.parameters.items():
            if props.get('vector', False):
                if analysis_type is None or analysis_type in props['analysis_types']:
                    results.append(name)
        return results
    
    def get_unit(self, param_name: str) -> str:
        """Get unit for a parameter."""
        return self.parameters.get(param_name, {}).get('unit', '')
    
    def get_param_unit(self, param_name: str) -> str:
        """Alias for get_unit for external callers."""
        return self.get_unit(param_name)
    
    def get_symbol(self, param_name: str) -> str:
        """Get LaTeX symbol for a parameter."""
        return self.parameters.get(param_name, {}).get('symbol', param_name)
    
    def get_units_dict(self, analysis_type: Optional[str] = None) -> Dict[str, str]:
        """Get dictionary mapping parameter names to units."""
        params = self.get_all_field_names(analysis_type)
        return {name: self.get_unit(name) for name in params}
    
    def is_computed_when_null(self, param_name: str) -> bool:
        """Check if a flexible parameter is computed when null/NaN."""
        param = self.parameters.get(param_name, {})
        return param.get('computed_when_null', False)
    
    def get_vector_length(self, param_name: str, default: int) -> int:
        """Return preferred vector length for a parameter."""
        return int(self.parameters.get(param_name, {}).get('vector_length', default))
    
    def resolve_alias(self, name: str) -> str:
        """Return canonical parameter name for an alias or the name itself."""
        return self._alias_to_name.get(name, name)
    
    def missing_required(self, provided_names: Iterable[str], analysis_type: str) -> List[str]:
        """List required inputs absent from provided_names (flexible allowed to be null)."""
        provided = {self.resolve_alias(n) for n in provided_names}
        missing = []
        for name in self.get_input_names(analysis_type):
            if name in provided:
                continue
            if self.is_computed_when_null(name):
                continue
            missing.append(name)
        return missing
    
    def convert_to_default_unit(self, param_name: str, values: np.ndarray, source_unit: str) -> Tuple[np.ndarray, str]:
        """
        Convert values to the canonical unit defined in the schema.
        Returns (values_converted, unit_used). Falls back to source_unit if conversion fails.
        """
        target_unit = self.get_unit(param_name) or source_unit or 'dimensionless'
        source_unit = source_unit or target_unit
        
        try:
            q = (values * u(source_unit)).to(target_unit)
            return q.magnitude, target_unit
        except Exception:
            if target_unit != source_unit:
                # Keep original unit if canonical one is not parseable
                return np.asarray(values, dtype=float), source_unit
            raise
    
    def make_result_dict(self, values: Dict[str, Any], 
                        analysis_type: Optional[str] = None,
                        default_value: Any = np.nan) -> Dict[str, Any]:
        """
        Create a unified result dictionary with ALL parameter fields.
        
        This ensures consistent HDF5 file structure regardless of analysis type.
        Only fields relevant to the analysis_type are populated from values;
        irrelevant fields are set to default_value (typically np.nan).
        
        Args:
            values: Dictionary with actual computed/measured values
            analysis_type: Optional. If provided, only updates fields relevant to this type.
                          If None, updates all fields present in values.
            default_value: Default value for all fields (default: np.nan)
            
        Returns:
            Complete dictionary with ALL parameter fields (inputs + outputs)
            
        Examples:
            >>> # T_seeded analysis - only T_seeded fields are populated
            >>> make_result_dict({'V_plasma': 150, 't_startup': 1000}, 'T_seeded')
            {'V_plasma': 150, 'tau_p_He3': nan, 't_startup': 1000, 'n_He3': nan, ...}
            
            >>> # No analysis_type - update any fields provided
            >>> make_result_dict({'V_plasma': 150, 'custom_field': 42})
            {'V_plasma': 150, 'custom_field': 42, 'T_i': nan, ...}
        """
        # Start with all fields set to default
        all_field_names = list(self.parameters.keys())
        result = {field: default_value for field in all_field_names}
        
        if analysis_type is None:
            # No filtering - update any provided fields
            result.update(values)
        else:
            # Get relevant fields for this analysis type
            relevant_fields = set(self.get_all_field_names(analysis_type))
            
            # Only update fields that are relevant to this analysis type
            for key, value in values.items():
                if key in relevant_fields or key not in self.parameters:
                    # Include if: relevant field OR not in schema (e.g., 'error', 'sol_success')
                    result[key] = value
        
        return result
    
    def get_param_label(self, param_name: str, unit: Optional[str] = None, 
                       use_symbol: bool = True) -> str:
        """
        Get formatted parameter label for plotting.
        
        Args:
            param_name: Parameter name (e.g., 'V_plasma', 't_startup')
            unit: Unit string (e.g., 'm³', 'keV', 's'). If None, uses default unit.
            use_symbol: If True, use LaTeX symbol; if False, use parameter name
        
        Returns:
            Formatted label for use in plots
        
        Examples:
            >>> reg.get_param_label('V_plasma')
            '$V_{\\mathrm{plasma}}$ [m³]'
            
            >>> reg.get_param_label('T_i', use_symbol=False)
            'T_i [keV]'
        """
        if use_symbol:
            label = self.get_symbol(param_name)
        else:
            label = param_name
        
        # Use provided unit or default from schema
        if unit is None:
            unit = self.get_unit(param_name)
        
        if unit and unit not in ['boolean', 'string', 'dimensionless']:
            label = f"{label} [{unit}]"
        
        return label


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_registry = None

def get_registry() -> ParameterRegistry:
    """Get singleton parameter registry instance."""
    global _registry
    if _registry is None:
        _registry = ParameterRegistry()
    return _registry
