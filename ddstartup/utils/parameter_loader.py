"""
YAML-based parameter loading for DD startup simulations.

This module loads parameter configurations from YAML files and converts them
to numpy arrays with proper units, ready for computation.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy.stats import norm
from .units_and_constants import u
from .parameter_registry import get_registry


class ParameterLoader:
    """
    Load parameter fields from YAML configuration files.
    
    Converts YAML parameter definitions into numpy arrays with unit metadata,
    handling unit conversion and validation.
    
    Example YAML structure:
        parameters:
          V_plasma_field:
            type: linear
            min: 1000
            max: 2000
            points: 10
            unit: m^3
            description: Plasma volume
          
          T_i_field:
            type: scalar
            value: 30
            unit: keV
            description: Ion temperature
    """
    
    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> Dict[str, Any]:
        """
        Load parameter fields from YAML file.
        
        Args:
            yaml_path: Path to YAML parameter configuration file
            
        Returns:
            Dictionary mapping field names to tuples of (values_array, unit_string, metadata)
            or scalar values for simple parameters
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML structure is invalid or required fields are missing
            yaml.YAMLError: If YAML file is malformed
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'parameters' not in config:
            raise ValueError("YAML file must contain 'parameters' section")
        
        param_fields = {}
        
        # Get expected parameter field names from registry (append '_field' suffix)
        registry = get_registry()
        param_names = registry.get_parameter_names()
        field_names = [f"{name}_field" for name in param_names] + ['total_time']
        
        for field_name in field_names:
            if field_name in config['parameters']:
                param_def = config['parameters'][field_name]
                param_fields[field_name] = cls._create_parameter_array(field_name, param_def)
            else:
                # Return None for missing parameters (they may be optional)
                param_fields[field_name] = None
        
        return param_fields
    
    @classmethod
    def _create_parameter_array(cls, name: str, definition: Dict[str, Any]) -> Any:
        """
        Create a parameter array from YAML definition.
        
        Args:
            name: Parameter name (e.g., 'V_plasma', 'T_i', 'max_simulation_time')
            definition: YAML parameter definition dictionary
            
        Returns:
            Tuple of (numpy_array, unit_string, metadata_dict) or scalar value for max_simulation_time
            
        Raises:
            ValueError: If parameter definition is invalid
        """
        param_type = definition.get('type', 'scalar')
        
        # Handle special case: max_simulation_time is just a scalar value
        if name == 'max_simulation_time':
            return definition.get('value', 10 * 365 * 24 * 3600)
        
        # Get unit string
        unit_str = definition.get('unit', 'dimensionless')
        
        # Get number of points
        param_points = definition.get('points', 1)
        
        # Generate values based on type
        if param_type == 'scalar':
            values = cls._generate_scalar_values(name, definition, param_points)
        elif param_type == 'linear':
            values = cls._generate_linear_values(name, definition, param_points)
        elif param_type == 'normal':
            values = cls._generate_normal_values(name, definition, param_points)
        elif param_type == 'vector':
            values = cls._generate_vector_values(name, definition, param_points)
        else:
            raise ValueError(f"Unknown parameter type '{param_type}' for {name}")
        
        # Create metadata
        metadata = {
            'name': name,
            'type': param_type,
            'description': definition.get('description', '')
        }
        
        return (values, unit_str, metadata)
    

    @classmethod
    def _parse_numeric_value(cls, value: Any) -> float:
        """
        Parse a numeric value from YAML, handling strings and special values.
        
        Args:
            value: Value to parse (int, float, str, etc.)
            
        Returns:
            Parsed float value
        """
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Handle NaN
            if value.lower() in ['nan', '.nan', 'null', 'none']:
                return float('nan')
            # Try to parse as float (handles scientific notation)
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Cannot parse numeric value: {value}")
        raise ValueError(f"Cannot parse numeric value of type {type(value)}: {value}")
    
    @classmethod
    def _generate_scalar_values(cls, name: str, definition: Dict[str, Any], param_points: int) -> np.ndarray:
        """Generate constant parameter values."""
        if 'value' not in definition:
            raise ValueError(f"Scalar parameter '{name}' must have 'value' field")
        
        value = cls._parse_numeric_value(definition['value'])
        return np.full(param_points, value)
    
    @classmethod
    def _generate_linear_values(cls, name: str, definition: Dict[str, Any], param_points: int) -> np.ndarray:
        """Generate linearly spaced parameter values."""
        if 'min' not in definition or 'max' not in definition:
            raise ValueError(f"Linear parameter '{name}' must have 'min' and 'max' fields")
        
        min_val = cls._parse_numeric_value(definition['min'])
        max_val = cls._parse_numeric_value(definition['max'])
        
        if param_points == 1:
            return np.array([(min_val + max_val) / 2])
        
        return np.linspace(min_val, max_val, param_points)
    
    @classmethod
    def _generate_normal_values(cls, name: str, definition: Dict[str, Any], param_points: int) -> np.ndarray:
        """Generate normally distributed parameter values."""
        if 'mean' not in definition:
            raise ValueError(f"Normal parameter '{name}' must have 'mean' field")
        
        mean = cls._parse_numeric_value(definition['mean'])
        std = cls._parse_numeric_value(definition.get('std', 1.0))
        
        if param_points == 1:
            return np.array([mean])
        
        # Avoid 0 and 1 percentiles to prevent infinite values
        percentiles = np.linspace(0, 1, param_points + 2)[1:-1]
        return mean + std * norm.ppf(percentiles)
    
    @classmethod
    def _generate_vector_values(cls, name: str, definition: Dict[str, Any], param_points: int) -> np.ndarray:
        """Generate parameter values from provided vector."""
        if 'values' not in definition:
            raise ValueError(f"Vector parameter '{name}' must have 'values' field")
        
        values = definition['values']
        if not isinstance(values, list):
            raise ValueError(f"Vector parameter '{name}' values must be a list")
        
        # Parse all values
        parsed_values = [cls._parse_numeric_value(v) for v in values]
        
        # Note: param_points is overridden by actual vector length
        if len(parsed_values) != param_points and param_points != 1:
            # Update param_points to match vector length
            pass
        
        return np.array(parsed_values)


def validate_parameter_fields(param_fields: Dict[str, Any], analysis_type: str) -> None:
    """
    Validate that all required parameters are present for the given analysis type.
    
    Uses parameter registry to determine required fields dynamically.
    
    Args:
        param_fields: Dictionary of parameter fields
        analysis_type: Type of analysis ('T_seeded' or 'lump')
        
    Raises:
        ValueError: If required parameters are missing
    """
    if analysis_type not in ['T_seeded', 'lump']:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Get required parameters from registry and convert to field names
    registry = get_registry()
    required_params = registry.get_input_names(analysis_type)
    required_fields = [f"{name}_field" for name in required_params]
    
    missing = [field for field in required_fields if param_fields.get(field) is None]
    
    if missing:
        raise ValueError(
            f"Missing required parameters for {analysis_type} analysis: {', '.join(missing)}"
        )
