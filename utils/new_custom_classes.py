import numpy as np
import pint
from typing import Callable, Optional, Sequence, Union
from .units_and_constants import u

class ParameterField:
    """
    Flexible parameter field for parametric and time dependent analysis.
    Shape: (param_points)
    """
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
        # Spatial options
        spatial_profile: Optional[str] = None,  # None, "uniform", or "pedestal"
        space_points: int = 1,
        value_center: Optional[Union[float, "ParameterField"]] = None,
        value_ped: Optional[Union[float, "ParameterField"]] = None,
        value_edge: Optional[Union[float, "ParameterField"]] = None,
        transition_ratio: float = 0.95,
    ):
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
        if parametrization_type not in ["normal", "linear", "scalar"]:
            raise ValueError(f"parametrization_type must be 'normal', 'linear', or 'scalar', got {parametrization_type}")
        
        # check mean, std, min_val, max_val - they should be floats
              # if they are pint quantities, convert to float and use the pint quantity for self.unit
              # also print a warning
        if mean is not None and not isinstance(mean, (float, int)):
            if isinstance(mean, pint.Quantity):
                mean = mean.to(self.unit).magnitude
                print(f"Warning: mean converted to float: {mean}")
            else:
                raise ValueError("mean must be a float or pint.Quantity")
        if std is not None and not isinstance(std, (float, int)):
            if isinstance(std, pint.Quantity):
                std = std.to(self.unit).magnitude
                print(f"Warning: std converted to float: {std}")
            else:
                raise ValueError("std must be a float or pint.Quantity")
        if min_val is not None and not isinstance(min_val, (float, int)):
            if isinstance(min_val, pint.Quantity):
                min_val = min_val.to(self.unit).magnitude
                print(f"Warning: min_val converted to float: {min_val}")
            else:
                raise ValueError("min_val must be a float or pint.Quantity")
        if max_val is not None and not isinstance(max_val, (float, int)):
            if isinstance(max_val, pint.Quantity):
                max_val = max_val.to(self.unit).magnitude
                print(f"Warning: max_val converted to float: {max_val}")
            else:
                raise ValueError("max_val must be a float or pint.Quantity")
        
        
        # Generate parameter values based on type
        if parametrization_type == "normal":
            if mean is None:
                raise ValueError("mean is required for normal parametrization")
            if param_points == 1:
                param_values = np.array([mean])
            else:
                percentiles = np.linspace(0, 1, param_points + 2)[1:-1]  # avoid 0 and 1
                from scipy.stats import norm
                param_values = mean + (std if std is not None else 1) * norm.ppf(percentiles)
                
        elif parametrization_type == "linear":
            if min_val is None or max_val is None:
                raise ValueError("min_val and max_val are required for linear parametrization")
            if param_points == 1:
                param_values = np.array([(min_val + max_val) / 2])
            else:
                param_values = np.linspace(min_val, max_val, param_points)
                
        elif parametrization_type == "scalar":
            if mean is None:
                raise ValueError("mean is required for scalar parametrization")
            param_values = np.full(param_points, mean)
        
        # Fix the broadcasting issue
        base_data = param_values  # param_values is already shape (param_points,)

        # Spatial profile
        if spatial_profile is None or space_points == 1:
            # No spatial dependence: 1D array
            data = base_data  # shape: (param_points,)
        elif spatial_profile == "uniform":
            # Uniform spatial profile: 2D array (param_points, space_points)
            data = np.broadcast_to(base_data[:, np.newaxis], (param_points, space_points))
        elif spatial_profile == "pedestal":
            data = self._generate_pedestal_profiles(base_data)
        else:
            raise ValueError(f"Unknown spatial_profile: {spatial_profile}")

        self.data = data * self.unit

    def _get_parameter_values(self, param, default_value=None):
        if isinstance(param, ParameterField):
            if param.param_points != self.param_points:
                raise ValueError("Mismatched dimensions in spatial parameter fields")
            return param.data.magnitude
        elif param is not None:
            return param
        else:
            return default_value

    def _generate_pedestal_profiles(self, base_data):
        r = np.linspace(0, 1, self.space_points)
        spatial_data = np.zeros((self.param_points, self.space_points))

        center_vals = self._get_parameter_values(self.value_center, 1.0)
        ped_vals = self._get_parameter_values(self.value_ped, 0.5)
        edge_vals = self._get_parameter_values(self.value_edge, 0.1)

        def get_val(val_array, i, scale):
            if np.isscalar(val_array):
                return val_array * scale
            return val_array[i] * scale

        for i in range(self.param_points):
            scale = base_data[i]
            center = get_val(center_vals, i, scale)
            ped = get_val(ped_vals, i, scale)
            edge = get_val(edge_vals, i, scale)
            spatial_data[i, :] = self._compute_pedestal_profile(r, center, ped, edge)

        return spatial_data

    def _compute_pedestal_profile(self, r, center, ped, edge):
        profile = np.zeros_like(r)
        tr = self.transition_ratio
        core = r <= tr
        edge_mask = r > tr

        if tr > 0:
            profile[core] = center - (center - ped) * (r[core] / tr) ** 2
        else:
            profile[core] = center

        if tr < 1:
            profile[edge_mask] = ped + (edge - ped) * (r[edge_mask] - tr) / (1 - tr)
        else:
            profile[edge_mask] = ped

        return profile

    def __repr__(self):
        header = (
            f"<ParameterField '{self.name or 'unnamed'}' "
            f"type={self.parametrization_type}, "
            f"profile={self.spatial_profile or 'none'}, "
            f"shape={self.data.shape}, unit={self.unit}>"
        )

        if (self.spatial_profile is None or self.space_points == 1) and self.data.ndim == 1:
            # 1D array: just print the values
            values_str = " ".join(f"{v:.3f}" for v in self.data.magnitude)
            return f"{header}\n[{values_str}] {self.unit}"
        else:
            # 2D array: print each parameter's spatial profile
            lines = []
            for i in range(self.param_points):
                values = self.data.magnitude[i, :]
                values_str = ", ".join(f"{v:.3f}" for v in values)
                lines.append(f"param {i}: [{values_str}]")
            return f"{header}\nData (magnitude):\n" + "\n".join(lines)

    def plot(self):
        import matplotlib.pyplot as plt

        if self.space_points == 1:
            # For scalar spatial data, plot parameter values as a bar plot
            fig, ax = plt.subplots(figsize=(8, 6))
            param_vals = self.data.magnitude if self.data.ndim == 1 else self.data.magnitude[:, 0]
            ax.bar(range(self.param_points), param_vals)
            ax.set_xlabel("Parameter Index")
            ax.set_ylabel(f"Value [{self.unit}]")
            ax.set_title(f"{self.name or 'ParameterField'} - Parameter Values")
        else:
            # For spatial data, plot spatial profiles
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.linspace(0, 1, self.space_points)
            for i in range(self.param_points):
                y = self.data.magnitude[i, :]
                label = f"param {i}" if self.param_points > 1 else self.name or "data"
                ax.plot(x, y, label=label, alpha=0.8)
            ax.set_xlabel("Normalized Radial Position")
            ax.set_ylabel(f"Value [{self.unit}]")
            ax.set_title(f"{self.name or 'ParameterField'} - Spatial Profiles")
            if self.param_points > 1:
                ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig, ax