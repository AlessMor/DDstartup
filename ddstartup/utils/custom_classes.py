import numpy as np
import pint
from typing import Optional, Sequence, Union
from scipy.stats import norm
from .units_and_constants import u


class ParameterField:
    """
    Flexible parameter field for parametric and time dependent analysis.
    
    Supports multiple parametrization types (normal, linear, scalar, vector) and
    optional spatial profiles (uniform, pedestal) for plasma simulations.
    
    Instance Attributes:
        data (pint.Quantity): NumPy array with pint units, shape (param_points,) or 
            (param_points, space_points). Contains the actual parameter values.
        unit (pint.Unit): Pint unit for the parameter (e.g., u.keV, u.m**3, u.dimensionless).
        name (str or None): Optional descriptive name for the parameter field.
        parametrization_type (str): Type of parameter generation - one of "normal", "linear", 
            "scalar", or "vector".
        param_points (int): Number of parameter points in the parametric sweep.
        (NOT USED!) space_points (int): Number of spatial points for spatial profiles (1 if no spatial profile).
        (NOT USED!) spatial_profile (str or None): Type of spatial profile - None, "uniform", or "pedestal".
        (NOT USED!) transition_ratio (float): Radial position [0, 1] where pedestal transitions to edge 
            (only used for pedestal profiles).
        (NOT USED!) value_center (float, ParameterField, or None): Relative value at plasma center for 
            pedestal profile (multiplied by base parameter value).
        (NOT USED!) value_ped (float, ParameterField, or None): Relative value at pedestal region for 
            pedestal profile (multiplied by base parameter value).
        (NOT USED!) value_edge (float, ParameterField, or None): Relative value at plasma edge for 
            pedestal profile (multiplied by base parameter value).
        mean (float or None): Mean value used for "normal" or "scalar" parametrization.
        std (float or None): Standard deviation used for "normal" parametrization.
        min_val (float or None): Minimum value used for "linear" parametrization.
        max_val (float or None): Maximum value used for "linear" parametrization.
        vector (np.ndarray or None): Custom values array used for "vector" parametrization.
    
    Class Attributes:
        VALID_PARAM_TYPES (set): Supported parametrization types {"normal", "linear", "scalar", "vector"}.
        VALID_SPATIAL_PROFILES (set): Supported spatial profiles {None, "uniform", "pedestal"}.
    
    Note:
        The parametrization parameters (mean, std, min_val, max_val, vector) are stored for 
        reproducibility and introspection. Only the parameters relevant to the chosen 
        parametrization_type will be non-None.
    """
    
    # Valid parametrization types
    VALID_PARAM_TYPES = {"normal", "linear", "scalar", "vector"}
    VALID_SPATIAL_PROFILES = {None, "uniform", "pedestal"}
    
    def __init__(
        self,
        unit: Optional[pint.Unit] = None,      # pint unit or dimensionless
        name: Optional[str] = None,
        # Parameter options
        param_points: int = 1,                 # Number of parameter points
        parametrization_type: str = "scalar",  # Required: "normal", "linear", or "scalar"
        mean: Optional[float] = None,          # For normal and scalar
        std: Optional[float] = None,           # For normal
        min_val: Optional[float] = None,       # For linear
        max_val: Optional[float] = None,       # For linear
        vector: Optional[Sequence[float]] = None,  # For scalar
        # Spatial options
        spatial_profile: Optional[str] = None,  # None, "uniform", or "pedestal"
        space_points: int = 1,
        value_center: Optional[Union[float, "ParameterField"]] = None,
        value_ped: Optional[Union[float, "ParameterField"]] = None,
        value_edge: Optional[Union[float, "ParameterField"]] = None,
        transition_ratio: float = 0.95,
    ):
        # Store basic attributes
        self.parametrization_type = parametrization_type
        self.unit = unit if unit is not None else u.dimensionless
        self.param_points = param_points
        self.space_points = space_points
        self.name = name
        self.spatial_profile = spatial_profile
        self.transition_ratio = transition_ratio
        self.value_center = value_center
        self.value_ped = value_ped
        self.value_edge = value_edge
        
        # Validate parametrization type
        self._validate_parametrization_type()
        
        # Convert pint quantities to floats with unit checking
        mean = self._convert_to_magnitude(mean, "mean")
        std = self._convert_to_magnitude(std, "std")
        min_val = self._convert_to_magnitude(min_val, "min_val")
        max_val = self._convert_to_magnitude(max_val, "max_val")
        
        # Store parametrization parameters for reproducibility
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.vector = vector if vector is None else np.array(vector)
        
        # Generate parameter values based on type
        param_values = self._generate_param_values(mean, std, min_val, max_val, vector)
        base_data = param_values

        # Apply spatial profile
        data = self._apply_spatial_profile(base_data)
        self.data = data * self.unit
    
    def _validate_parametrization_type(self):
        """Validate that the parametrization type is supported."""
        if self.parametrization_type not in self.VALID_PARAM_TYPES:
            raise ValueError(
                f"parametrization_type must be one of {self.VALID_PARAM_TYPES}, "
                f"got '{self.parametrization_type}'"
            )
    
    def _convert_to_magnitude(self, value, name):
        """
        Convert pint quantities to float magnitudes.
        
        Args:
            value: Value to convert (float, int, pint.Quantity, or None)
            name: Parameter name for error messages
            
        Returns:
            Float magnitude or None
        """
        if value is None or isinstance(value, (float, int)):
            return value
        
        if isinstance(value, pint.Quantity):
            magnitude = value.to(self.unit).magnitude
            print(f"Warning: {name} converted to float: {magnitude}")
            return magnitude
        
        raise ValueError(f"{name} must be a float, int, or pint.Quantity")
    
    def _generate_param_values(self, mean, std, min_val, max_val, vector):
        """
        Generate parameter values based on parametrization type.
        
        Returns:
            NumPy array of shape (param_points,)
        """
        param_type = self.parametrization_type
        
        if param_type == "normal":
            return self._generate_normal_values(mean, std)
        elif param_type == "linear":
            return self._generate_linear_values(min_val, max_val)
        elif param_type == "scalar":
            return self._generate_scalar_values(mean)
        elif param_type == "vector":
            return self._generate_vector_values(vector)
    
    def _generate_normal_values(self, mean, std):
        """Generate normally distributed parameter values."""
        if mean is None:
            raise ValueError("mean is required for normal parametrization")
        
        if self.param_points == 1:
            return np.array([mean])
        
        # Avoid 0 and 1 percentiles to prevent infinite values
        percentiles = np.linspace(0, 1, self.param_points + 2)[1:-1]
        std_dev = std if std is not None else 1
        return mean + std_dev * norm.ppf(percentiles)
    
    def _generate_linear_values(self, min_val, max_val):
        """Generate linearly spaced parameter values."""
        if min_val is None or max_val is None:
            raise ValueError("min_val and max_val are required for linear parametrization")
        
        if self.param_points == 1:
            return np.array([(min_val + max_val) / 2])
        
        return np.linspace(min_val, max_val, self.param_points)
    
    def _generate_scalar_values(self, mean):
        """Generate constant parameter values."""
        if mean is None:
            raise ValueError("mean is required for scalar parametrization")
        
        return np.full(self.param_points, mean)
    
    def _generate_vector_values(self, vector):
        """Generate parameter values from provided vector."""
        if vector is None:
            raise ValueError("vector is required for vector parametrization")
        
        if len(vector) != self.param_points:
            raise ValueError(
                f"Length of vector ({len(vector)}) must match param_points ({self.param_points})"
            )
        
        return np.array(vector)
    
    def _apply_spatial_profile(self, base_data):
        """
        Apply spatial profile to parameter values.
        
        Args:
            base_data: 1D array of shape (param_points,)
            
        Returns:
            1D or 2D array depending on spatial profile
        """
        if self.spatial_profile is None or self.space_points == 1:
            return base_data
        elif self.spatial_profile == "uniform":
            return np.broadcast_to(base_data[:, np.newaxis], (self.param_points, self.space_points))
        elif self.spatial_profile == "pedestal":
            return self._generate_pedestal_profiles(base_data)
        else:
            raise ValueError(
                f"Unknown spatial_profile '{self.spatial_profile}'. "
                f"Must be one of {self.VALID_SPATIAL_PROFILES}"
            )

    def _get_parameter_values(self, param, default_value=None):
        """
        Extract parameter values from ParameterField or scalar.
        
        Args:
            param: ParameterField, scalar, or None
            default_value: Default value if param is None
            
        Returns:
            Scalar or array of parameter values
        """
        if isinstance(param, ParameterField):
            if param.param_points != self.param_points:
                raise ValueError("Mismatched dimensions in spatial parameter fields")
            return param.data.magnitude
        
        return param if param is not None else default_value

    def _generate_pedestal_profiles(self, base_data):
        """
        Generate pedestal spatial profiles for all parameter points.
        
        Args:
            base_data: 1D array of shape (param_points,) containing scale factors
            
        Returns:
            2D array of shape (param_points, space_points)
        """
        r = np.linspace(0, 1, self.space_points)
        spatial_data = np.zeros((self.param_points, self.space_points))

        center_vals = self._get_parameter_values(self.value_center, 1.0)
        ped_vals = self._get_parameter_values(self.value_ped, 0.5)
        edge_vals = self._get_parameter_values(self.value_edge, 0.1)

        for i in range(self.param_points):
            scale = base_data[i]
            center = self._get_scaled_value(center_vals, i, scale)
            ped = self._get_scaled_value(ped_vals, i, scale)
            edge = self._get_scaled_value(edge_vals, i, scale)
            spatial_data[i, :] = self._compute_pedestal_profile(r, center, ped, edge)

        return spatial_data
    
    @staticmethod
    def _get_scaled_value(val_array, index, scale):
        """
        Get scaled value from array or scalar.
        
        Args:
            val_array: Array or scalar value
            index: Index to extract if array
            scale: Scale factor to apply
            
        Returns:
            Scaled value
        """
        if np.isscalar(val_array):
            return val_array * scale
        return val_array[index] * scale

    def _compute_pedestal_profile(self, r, center, ped, edge):
        """
        Compute a single pedestal profile.
        
        The profile has a parabolic core region and linear edge region,
        with transition at self.transition_ratio.
        
        Args:
            r: Normalized radial coordinate array [0, 1]
            center: Value at center (r=0)
            ped: Value at pedestal (r=transition_ratio)
            edge: Value at edge (r=1)
            
        Returns:
            Profile array of same shape as r
        """
        profile = np.zeros_like(r)
        tr = self.transition_ratio
        core_mask = r <= tr
        edge_mask = r > tr

        # Core region: parabolic profile from center to pedestal
        if tr > 0:
            profile[core_mask] = center - (center - ped) * (r[core_mask] / tr) ** 2
        else:
            profile[core_mask] = center

        # Edge region: linear profile from pedestal to edge
        if tr < 1:
            profile[edge_mask] = ped + (edge - ped) * (r[edge_mask] - tr) / (1 - tr)
        else:
            profile[edge_mask] = ped

        return profile

    def __repr__(self):
        """String representation of ParameterField."""
        header = (
            f"<ParameterField '{self.name or 'unnamed'}' "
            f"type={self.parametrization_type}, "
            f"profile={self.spatial_profile or 'none'}, "
            f"shape={self.data.shape}, unit={self.unit}>"
        )

        # For 1D data, show all values in single line
        if (self.spatial_profile is None or self.space_points == 1) and self.data.ndim == 1:
            values_str = " ".join(f"{v:.3f}" for v in self.data.magnitude)
            return f"{header}\n[{values_str}] {self.unit}"
        
        # For 2D data, show each parameter's spatial profile
        lines = [
            f"param {i}: [{', '.join(f'{v:.3f}' for v in self.data.magnitude[i, :])}]"
            for i in range(self.param_points)
        ]
        return f"{header}\nData (magnitude):\n" + "\n".join(lines)

    def plot(self):
        """
        Plot the parameter field.
        
        For 1D data: bar plot of parameter values
        For 2D data: line plot of spatial profiles
        
        Returns:
            tuple: (figure, axes) matplotlib objects
        """
        import matplotlib.pyplot as plt

        title = f"{self.name or 'ParameterField'}"
        
        if self.space_points == 1:
            # Plot parameter values as bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            param_vals = self.data.magnitude if self.data.ndim == 1 else self.data.magnitude[:, 0]
            ax.bar(range(self.param_points), param_vals)
            ax.set_xlabel("Parameter Index")
            ax.set_ylabel(f"Value [{self.unit}]")
            ax.set_title(f"{title} - Parameter Values")
        else:
            # Plot spatial profiles
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.linspace(0, 1, self.space_points)
            
            for i in range(self.param_points):
                y = self.data.magnitude[i, :]
                label = f"param {i}" if self.param_points > 1 else self.name or "data"
                ax.plot(x, y, label=label, alpha=0.8)
            
            ax.set_xlabel("Normalized Radial Position")
            ax.set_ylabel(f"Value [{self.unit}]")
            ax.set_title(f"{title} - Spatial Profiles")
            
            if self.param_points > 1:
                ax.legend()
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fig, ax