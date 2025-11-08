"""
Tests for utils/custom_classes.py

This module tests the ParameterField class which is used to create flexible
parameter fields for parametric and time-dependent analysis with optional
spatial profiles.

Test coverage includes:
- Parametrization types: normal, linear, scalar, vector
- Spatial profiles: None, uniform, pedestal
- Unit handling with pint
- Error handling and validation
- Data shape verification
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddstartup.utils.custom_classes import ParameterField
from ddstartup.utils.units_and_constants import u


# =============================================================================
# PARAMETRIZATION TYPE TESTS
# =============================================================================

class TestScalarParametrization:
    """Tests for scalar parametrization type (constant values)"""
    
    def test_scalar_single_point(self):
        """Test creating a scalar field with a single parameter point"""
        # Create a scalar ParameterField with one point
        field = ParameterField(
            unit=u.m,
            name="test_scalar",
            parametrization_type="scalar",
            mean=5.0,
            param_points=1
        )
        
        # Verify the field contains the expected value
        assert field.data.shape == (1,)
        assert field.data.magnitude[0] == 5.0
        assert field.unit == u.m
    
    def test_scalar_multiple_points(self):
        """Test creating a scalar field with multiple parameter points"""
        # Scalar type should repeat the same value for all points
        field = ParameterField(
            unit=u.keV,
            name="temperature",
            parametrization_type="scalar",
            mean=69.0,
            param_points=5
        )
        
        # Expected: All five values should be identical (constant field)
        # Expected vector: [69.0, 69.0, 69.0, 69.0, 69.0]
        expected_vector = np.array([69.0, 69.0, 69.0, 69.0, 69.0])
        
        assert field.data.shape == (5,)
        np.testing.assert_array_equal(field.data.magnitude, expected_vector)
    
    def test_scalar_requires_mean(self):
        """Test that scalar parametrization requires a mean value"""
        # Should raise ValueError if mean is not provided
        with pytest.raises(ValueError, match="mean is required"):
            ParameterField(
                parametrization_type="scalar",
                param_points=5
                # Missing mean parameter
            )


class TestLinearParametrization:
    """Tests for linear parametrization type (linearly spaced values)"""
    
    def test_linear_multiple_points(self):
        """Test creating a linear field with multiple points"""
        # Create linearly spaced values from min to max
        # With 5 points from 10 to 50, the spacing is (50-10)/(5-1) = 10
        field = ParameterField(
            unit=u.MW,
            name="power",
            parametrization_type="linear",
            min_val=10.0,
            max_val=50.0,
            param_points=5
        )
        
        # Expected: Linearly spaced values from 10.0 to 50.0
        # Expected vector: [10.0, 20.0, 30.0, 40.0, 50.0]
        # This is equivalent to np.linspace(10.0, 50.0, 5)
        expected_vector = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        assert field.data.shape == (5,)
        np.testing.assert_array_almost_equal(field.data.magnitude, expected_vector)
    
    def test_linear_single_point(self):
        """Test creating a linear field with a single point (should be midpoint)"""
        # With one point, should return the midpoint between min and max
        # Midpoint = (min_val + max_val) / 2 = (0 + 100) / 2 = 50.0
        field = ParameterField(
            unit=u.m**3,
            parametrization_type="linear",
            min_val=0.0,
            max_val=100.0,
            param_points=1
        )
        
        # Expected: Single value at the midpoint between min and max
        # Expected vector: [50.0]
        expected_vector = np.array([50.0])
        
        assert field.data.shape == (1,)
        np.testing.assert_array_equal(field.data.magnitude, expected_vector)
    
    def test_linear_requires_min_max(self):
        """Test that linear parametrization requires both min and max values"""
        # Should raise ValueError if min_val or max_val is missing
        with pytest.raises(ValueError, match="min_val and max_val are required"):
            ParameterField(
                parametrization_type="linear",
                min_val=0.0,
                param_points=5
                # Missing max_val
            )
        
        with pytest.raises(ValueError, match="min_val and max_val are required"):
            ParameterField(
                parametrization_type="linear",
                max_val=100.0,
                param_points=5
                # Missing min_val
            )


class TestNormalParametrization:
    """Tests for normal (Gaussian) parametrization type"""
    
    def test_normal_distribution(self):
        """Test creating a field with normal distribution"""
        # Create normally distributed values using quantiles
        # Uses scipy.stats.norm.ppf() to convert percentiles to z-scores
        # With 5 points, percentiles are [0.1667, 0.3333, 0.5, 0.6667, 0.8333]
        # (avoiding 0 and 1 to prevent infinite values)
        field = ParameterField(
            unit=u.s,
            name="tau",
            parametrization_type="normal",
            mean=10.0,
            std=2.0,
            param_points=5
        )
        
        # Expected: Values distributed around mean=10.0 with std=2.0
        # Approximate expected vector (from norm.ppf): 
        # [8.05, 9.16, 10.0, 10.84, 11.95] (values vary based on percentiles)
        # We verify the mean and shape rather than exact values
        
        assert field.data.shape == (5,)
        # Mean should be approximately the specified mean
        assert np.abs(field.data.magnitude.mean() - 10.0) < 1.0
        # Values should be sorted (percentiles are monotonic)
        assert np.all(np.diff(field.data.magnitude) > 0)
    
    def test_normal_single_point(self):
        """Test normal distribution with single point returns mean"""
        # With one point, should just return the mean
        # (std is ignored since there's only one value)
        field = ParameterField(
            unit=u.dimensionless,
            parametrization_type="normal",
            mean=5.0,
            std=1.0,
            param_points=1
        )
        
        # Expected: Single value equal to the mean
        # Expected vector: [5.0]
        expected_vector = np.array([5.0])
        
        assert field.data.shape == (1,)
        np.testing.assert_array_equal(field.data.magnitude, expected_vector)
    
    def test_normal_default_std(self):
        """Test that normal distribution uses default std=1 if not provided"""
        # If std is not provided, should default to 1.0
        field = ParameterField(
            parametrization_type="normal",
            mean=10.0,
            param_points=5
        )
        
        assert field.data.shape == (5,)
        # Should have some spread around mean
        assert field.data.magnitude.std() > 0
    
    def test_normal_requires_mean(self):
        """Test that normal parametrization requires a mean value"""
        with pytest.raises(ValueError, match="mean is required"):
            ParameterField(
                parametrization_type="normal",
                std=2.0,
                param_points=5
                # Missing mean
            )


class TestVectorParametrization:
    """Tests for vector parametrization type (user-provided values)"""
    
    def test_vector_custom_values(self):
        """Test creating a field from a custom vector of values"""
        # Provide exact values to use
        # This allows complete control over the parameter values
        custom_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        field = ParameterField(
            unit=u.kg,
            name="inventory",
            parametrization_type="vector",
            vector=custom_values,
            param_points=5
        )
        
        # Expected: Exact match to the provided custom values
        # Expected vector: [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert field.data.shape == (5,)
        np.testing.assert_array_equal(field.data.magnitude, expected_vector)
    
    def test_vector_requires_vector_param(self):
        """Test that vector parametrization requires vector parameter"""
        with pytest.raises(ValueError, match="vector is required"):
            ParameterField(
                parametrization_type="vector",
                param_points=3
                # Missing vector parameter
            )
    
    def test_vector_length_mismatch(self):
        """Test that vector length must match param_points"""
        with pytest.raises(ValueError, match="Length of vector.*must match param_points"):
            ParameterField(
                parametrization_type="vector",
                vector=[1.0, 2.0, 3.0],
                param_points=5  # Mismatch: vector has 3 elements, param_points is 5
            )


class TestInvalidParametrizationType:
    """Tests for invalid parametrization type handling"""
    
    def test_invalid_type_raises_error(self):
        """Test that invalid parametrization type raises ValueError"""
        with pytest.raises(ValueError, match="parametrization_type must be one of"):
            ParameterField(
                parametrization_type="invalid_type",
                param_points=3
            )


# =============================================================================
# SPATIAL PROFILE TESTS
# =============================================================================

class TestNoSpatialProfile:
    """Tests for fields without spatial profiles (1D arrays)"""
    
    def test_no_spatial_profile(self):
        """Test field without spatial profile is 1D"""
        # Without spatial profile, data should be 1D
        field = ParameterField(
            parametrization_type="linear",
            min_val=0.0,
            max_val=10.0,
            param_points=5,
            spatial_profile=None,
            space_points=1
        )
        
        assert field.data.ndim == 1
        assert field.data.shape == (5,)


class TestUniformSpatialProfile:
    """Tests for uniform spatial profile (constant in space)"""
    
    def test_uniform_spatial_profile(self):
        """Test uniform spatial profile creates 2D array with constant spatial values"""
        # Uniform profile broadcasts parameter values across space
        field = ParameterField(
            unit=u.dimensionless,
            parametrization_type="linear",
            min_val=1.0,
            max_val=5.0,
            param_points=3,
            spatial_profile="uniform",
            space_points=10
        )
        
        # Should be 2D: (param_points, space_points)
        assert field.data.shape == (3, 10)
        
        # Each row should be constant (uniform in space)
        for i in range(3):
            assert np.all(field.data.magnitude[i, :] == field.data.magnitude[i, 0])
    
    def test_uniform_different_param_values(self):
        """Test that uniform profile preserves different parameter values"""
        # Create 3 different parameter values with uniform spatial profile
        field = ParameterField(
            parametrization_type="vector",
            vector=[10.0, 20.0, 30.0],
            param_points=3,
            spatial_profile="uniform",
            space_points=5
        )
        
        # Expected: 2D array where each row is constant (uniform in space)
        # Expected shape: (3 parameters, 5 spatial points)
        # Expected data:
        #   Row 0 (param 0): [10.0, 10.0, 10.0, 10.0, 10.0]
        #   Row 1 (param 1): [20.0, 20.0, 20.0, 20.0, 20.0]
        #   Row 2 (param 2): [30.0, 30.0, 30.0, 30.0, 30.0]
        
        assert field.data.shape == (3, 5)
        
        # Verify each parameter has constant spatial profile
        expected_row_0 = np.full(5, 10.0)
        expected_row_1 = np.full(5, 20.0)
        expected_row_2 = np.full(5, 30.0)
        
        np.testing.assert_array_almost_equal(field.data.magnitude[0, :], expected_row_0)
        np.testing.assert_array_almost_equal(field.data.magnitude[1, :], expected_row_1)
        np.testing.assert_array_almost_equal(field.data.magnitude[2, :], expected_row_2)


class TestPedestalSpatialProfile:
    """Tests for pedestal spatial profile (core + pedestal + edge)"""
    
    def test_pedestal_profile_shape(self):
        """Test pedestal profile creates correct 2D shape"""
        # Pedestal profile creates a radial profile with core, pedestal, edge
        field = ParameterField(
            unit=u.keV,
            parametrization_type="scalar",
            mean=10.0,
            param_points=2,
            spatial_profile="pedestal",
            space_points=20,
            value_center=1.0,  # Center value (relative to mean)
            value_ped=0.8,     # Pedestal value (relative to mean)
            value_edge=0.3,    # Edge value (relative to mean)
            transition_ratio=0.9
        )
        
        assert field.data.shape == (2, 20)
    
    def test_pedestal_profile_structure(self):
        """Test pedestal profile has expected structure (decreasing from center to edge)"""
        # Create a pedestal profile with:
        # - Parabolic core from r=0 to r=0.8 (center to pedestal)
        # - Linear edge from r=0.8 to r=1.0 (pedestal to edge)
        # Base value is mean=10.0, scaled by relative values
        field = ParameterField(
            parametrization_type="scalar",
            mean=10.0,
            param_points=1,
            spatial_profile="pedestal",
            space_points=100,
            value_center=1.0,    # Center: 1.0 * 10.0 = 10.0
            value_ped=0.5,       # Pedestal: 0.5 * 10.0 = 5.0
            value_edge=0.1,      # Edge: 0.1 * 10.0 = 1.0
            transition_ratio=0.8  # Transition at 80% of radius
        )
        
        profile = field.data.magnitude[0, :]
        
        # Expected profile structure:
        # - At r=0 (center): 1.0 * 10.0 = 10.0
        # - At r=0.8 (pedestal): 0.5 * 10.0 = 5.0
        # - At r=1.0 (edge): 0.1 * 10.0 = 1.0
        # - Core region (r < 0.8): parabolic decrease from 10.0 to 5.0
        # - Edge region (r > 0.8): linear decrease from 5.0 to 1.0
        
        # Center value should be highest
        assert profile[0] == pytest.approx(10.0, rel=1e-2)
        
        # Edge value should be lowest
        assert profile[-1] == pytest.approx(1.0, rel=1e-2)
        
        # Profile should generally decrease from center to edge
        assert profile[0] > profile[-1]
        
        # Approximate pedestal value at transition point (r ≈ 0.8)
        transition_index = int(0.8 * 100)
        assert profile[transition_index] == pytest.approx(5.0, rel=0.1)
    
    def test_pedestal_with_parameter_field_values(self):
        """Test pedestal profile using ParameterField for spatial parameters"""
        # Create parameter fields for center, pedestal, and edge values
        center_field = ParameterField(
            parametrization_type="vector",
            vector=[1.0, 0.9],
            param_points=2
        )
        
        # Use ParameterField for defining spatial profile values
        field = ParameterField(
            parametrization_type="vector",
            vector=[10.0, 20.0],
            param_points=2,
            spatial_profile="pedestal",
            space_points=10,
            value_center=center_field,  # ParameterField input
            value_ped=0.5,
            value_edge=0.1,
            transition_ratio=0.8
        )
        
        assert field.data.shape == (2, 10)
        
        # First param scaled by 10.0, second by 20.0
        # First center: 1.0 * 10.0 = 10.0
        # Second center: 0.9 * 20.0 = 18.0
        assert field.data.magnitude[0, 0] == pytest.approx(10.0, rel=1e-2)
        assert field.data.magnitude[1, 0] == pytest.approx(18.0, rel=1e-2)
    
    def test_pedestal_dimension_mismatch_error(self):
        """Test that pedestal profile with mismatched ParameterField raises error"""
        # Create a ParameterField with wrong number of points
        wrong_field = ParameterField(
            parametrization_type="scalar",
            mean=1.0,
            param_points=3  # Mismatch
        )
        
        with pytest.raises(ValueError, match="Mismatched dimensions"):
            ParameterField(
                parametrization_type="scalar",
                mean=10.0,
                param_points=2,  # Different from wrong_field
                spatial_profile="pedestal",
                space_points=10,
                value_center=wrong_field  # Incompatible dimensions
            )


class TestInvalidSpatialProfile:
    """Tests for invalid spatial profile handling"""
    
    def test_invalid_spatial_profile_raises_error(self):
        """Test that invalid spatial profile type raises ValueError"""
        with pytest.raises(ValueError, match="Unknown spatial_profile"):
            ParameterField(
                parametrization_type="scalar",
                mean=10.0,
                param_points=2,
                spatial_profile="invalid_profile",
                space_points=10
            )


# =============================================================================
# UNIT HANDLING TESTS
# =============================================================================

class TestUnitHandling:
    """Tests for pint unit handling"""
    
    def test_default_dimensionless_unit(self):
        """Test that default unit is dimensionless"""
        field = ParameterField(
            parametrization_type="scalar",
            mean=1.0,
            param_points=1
        )
        
        assert field.unit == u.dimensionless
    
    def test_custom_unit(self):
        """Test setting custom units"""
        field = ParameterField(
            unit=u.keV,
            parametrization_type="scalar",
            mean=69.0,
            param_points=1
        )
        
        assert field.unit == u.keV
        assert field.data.magnitude[0] == 69.0
    
    def test_compound_unit(self):
        """Test compound units (e.g., m^3, kg/s)"""
        field = ParameterField(
            unit=u.m**3,
            parametrization_type="scalar",
            mean=100.0,
            param_points=1
        )
        
        assert field.unit == u.m**3
    
    def test_pint_quantity_conversion(self, capsys):
        """Test automatic conversion of pint Quantity to magnitude"""
        # If mean is provided as pint Quantity, should convert to magnitude
        field = ParameterField(
            unit=u.keV,
            parametrization_type="scalar",
            mean=50.0 * u.keV,  # Pint quantity
            param_points=1
        )
        
        # Should issue a warning
        captured = capsys.readouterr()
        assert "Warning: mean converted to float" in captured.out
        
        # Value should be extracted
        assert field.data.magnitude[0] == 50.0
    
    def test_invalid_pint_quantity_type(self):
        """Test that invalid types for mean/std/min/max raise errors"""
        with pytest.raises(ValueError, match="must be a float"):
            ParameterField(
                parametrization_type="scalar",
                mean="not_a_number",  # Invalid type
                param_points=1
            )


# =============================================================================
# REPRESENTATION AND UTILITY TESTS
# =============================================================================

class TestRepresentation:
    """Tests for __repr__ method"""
    
    def test_repr_1d_field(self):
        """Test string representation of 1D field"""
        field = ParameterField(
            unit=u.m,
            name="test_field",
            parametrization_type="scalar",
            mean=5.0,
            param_points=3
        )
        
        repr_str = repr(field)
        
        # Should contain key information
        assert "test_field" in repr_str
        assert "scalar" in repr_str
        assert "none" in repr_str  # No spatial profile
        assert str(field.data.shape) in repr_str
    
    def test_repr_2d_field(self):
        """Test string representation of 2D field (with spatial profile)"""
        field = ParameterField(
            name="spatial_field",
            parametrization_type="linear",
            min_val=1.0,
            max_val=5.0,
            param_points=2,
            spatial_profile="uniform",
            space_points=3
        )
        
        repr_str = repr(field)
        
        assert "spatial_field" in repr_str
        assert "linear" in repr_str
        assert "uniform" in repr_str
        assert "Data (magnitude)" in repr_str


class TestAttributes:
    """Tests for ParameterField attributes"""
    
    def test_basic_attributes(self):
        """Test that all basic attributes are set correctly"""
        field = ParameterField(
            unit=u.MW,
            name="power_field",
            parametrization_type="linear",
            min_val=10.0,
            max_val=50.0,
            param_points=5,
            spatial_profile="uniform",
            space_points=10,
            transition_ratio=0.85
        )
        
        assert field.name == "power_field"
        assert field.parametrization_type == "linear"
        assert field.param_points == 5
        assert field.space_points == 10
        assert field.spatial_profile == "uniform"
        assert field.transition_ratio == 0.85
        assert field.unit == u.MW
    
    def test_parametrization_params_stored(self):
        """Test that parametrization parameters are stored for reproducibility"""
        # Test linear parametrization stores min_val and max_val
        linear_field = ParameterField(
            parametrization_type="linear",
            min_val=10.0,
            max_val=50.0,
            param_points=5
        )
        assert linear_field.min_val == 10.0
        assert linear_field.max_val == 50.0
        assert linear_field.mean is None
        assert linear_field.std is None
        assert linear_field.vector is None
        
        # Test normal parametrization stores mean and std
        normal_field = ParameterField(
            parametrization_type="normal",
            mean=100.0,
            std=15.0,
            param_points=10
        )
        assert normal_field.mean == 100.0
        assert normal_field.std == 15.0
        assert normal_field.min_val is None
        assert normal_field.max_val is None
        assert normal_field.vector is None
        
        # Test scalar parametrization stores mean
        scalar_field = ParameterField(
            parametrization_type="scalar",
            mean=42.0,
            param_points=5
        )
        assert scalar_field.mean == 42.0
        assert scalar_field.std is None
        assert scalar_field.min_val is None
        assert scalar_field.max_val is None
        assert scalar_field.vector is None
        
        # Test vector parametrization stores vector
        vector_vals = [1.0, 2.0, 3.0]
        vector_field = ParameterField(
            parametrization_type="vector",
            vector=vector_vals,
            param_points=3
        )
        np.testing.assert_array_equal(vector_field.vector, vector_vals)
        assert vector_field.mean is None
        assert vector_field.std is None
        assert vector_field.min_val is None
        assert vector_field.max_val is None


# =============================================================================
# EDGE CASES AND INTEGRATION TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_single_space_point_with_spatial_profile(self):
        """Test that spatial_profile is ignored when space_points=1"""
        # Even with spatial profile specified, should behave as 1D if space_points=1
        field = ParameterField(
            parametrization_type="scalar",
            mean=10.0,
            param_points=3,
            spatial_profile="uniform",
            space_points=1
        )
        
        # Should still be 1D
        assert field.data.ndim == 1
        assert field.data.shape == (3,)
    
    def test_very_large_param_points(self):
        """Test handling of large number of parameter points"""
        # Should handle large arrays efficiently
        # Creates 1000 linearly spaced values from 0 to 1000
        field = ParameterField(
            parametrization_type="linear",
            min_val=0.0,
            max_val=1000.0,
            param_points=1000
        )
        
        # Expected: 1000 values linearly spaced from 0 to 1000
        # Expected vector: [0.0, 1.001, 2.002, ..., 998.998, 1000.0]
        # Spacing = 1000 / 999 ≈ 1.001
        
        assert field.data.shape == (1000,)
        assert field.data.magnitude[0] == pytest.approx(0.0)
        assert field.data.magnitude[-1] == pytest.approx(1000.0)
        
        # Check that spacing is approximately uniform
        spacing = np.diff(field.data.magnitude)
        expected_spacing = 1000.0 / 999
        np.testing.assert_array_almost_equal(spacing, expected_spacing, decimal=6)
    
    def test_zero_transition_ratio(self):
        """Test pedestal profile with transition_ratio=0"""
        # Edge case: transition at center (no core region)
        field = ParameterField(
            parametrization_type="scalar",
            mean=10.0,
            param_points=1,
            spatial_profile="pedestal",
            space_points=10,
            value_center=1.0,
            value_ped=0.5,
            value_edge=0.1,
            transition_ratio=0.0
        )
        
        # Should not raise error
        assert field.data.shape == (1, 10)
    
    def test_one_transition_ratio(self):
        """Test pedestal profile with transition_ratio=1"""
        # Edge case: transition at edge (no edge region)
        field = ParameterField(
            parametrization_type="scalar",
            mean=10.0,
            param_points=1,
            spatial_profile="pedestal",
            space_points=10,
            value_center=1.0,
            value_ped=0.5,
            value_edge=0.1,
            transition_ratio=1.0
        )
        
        # Should not raise error
        assert field.data.shape == (1, 10)


class TestClassConstants:
    """Tests for class-level constants"""
    
    def test_valid_param_types_constant(self):
        """Test that VALID_PARAM_TYPES contains expected values"""
        expected = {"normal", "linear", "scalar", "vector"}
        assert ParameterField.VALID_PARAM_TYPES == expected
    
    def test_valid_spatial_profiles_constant(self):
        """Test that VALID_SPATIAL_PROFILES contains expected values"""
        expected = {None, "uniform", "pedestal"}
        assert ParameterField.VALID_SPATIAL_PROFILES == expected


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features"""
    
    def test_complex_parameter_field(self):
        """Test creating a complex field with multiple features"""
        # Create a realistic physics parameter with spatial profile
        field = ParameterField(
            unit=u.keV,
            name="Ion Temperature",
            parametrization_type="normal",
            mean=69.0,
            std=5.0,
            param_points=10,
            spatial_profile="pedestal",
            space_points=50,
            value_center=1.0,
            value_ped=0.8,
            value_edge=0.3,
            transition_ratio=0.9
        )
        
        # Verify all aspects
        assert field.name == "Ion Temperature"
        assert field.unit == u.keV
        assert field.data.shape == (10, 50)
        assert field.parametrization_type == "normal"
        assert field.spatial_profile == "pedestal"
        
        # Each parameter point should have different mean but similar profile shape
        for i in range(10):
            profile = field.data.magnitude[i, :]
            # Profile should decrease from center to edge
            assert profile[0] > profile[-1]
    
    def test_parameter_field_for_sensitivity_analysis(self):
        """Test creating parameter fields suitable for sensitivity analysis"""
        # Create multiple fields as would be done for a sensitivity study
        # This demonstrates how different parametrization types can be combined
        fields = []
        
        # Volume parameter: Linear spacing from 50 to 200 m³
        # Expected: 20 values linearly spaced
        # Example values: [50.0, 57.89, 65.79, ..., 192.1, 200.0]
        fields.append(ParameterField(
            unit=u.m**3,
            name="Plasma Volume",
            parametrization_type="linear",
            min_val=50.0,
            max_val=200.0,
            param_points=20
        ))
        
        # Temperature parameter: Normal distribution around 69 keV
        # Expected: 20 values from normal distribution percentiles
        # Example values: [49.3, 54.8, 58.5, ..., 79.5, 84.2, 88.7] keV
        fields.append(ParameterField(
            unit=u.keV,
            name="Ion Temperature",
            parametrization_type="normal",
            mean=69.0,
            std=10.0,
            param_points=20
        ))
        
        # Custom parameter: Logarithmic spacing from 10 to 100 MW
        # Expected: 20 values logarithmically spaced
        # Example values: [10.0, 12.12, 14.68, ..., 68.13, 82.54, 100.0] MW
        log_vector = np.logspace(1, 2, 20)  # 10^1 to 10^2
        fields.append(ParameterField(
            unit=u.MW,
            name="Auxiliary Power",
            parametrization_type="vector",
            vector=log_vector,
            param_points=20
        ))
        
        # All fields should have consistent param_points for parameter sweep
        for field in fields:
            assert field.param_points == 20
            assert field.data.shape == (20,)
        
        # Verify specific properties
        # Linear field: first and last values
        assert fields[0].data.magnitude[0] == pytest.approx(50.0)
        assert fields[0].data.magnitude[-1] == pytest.approx(200.0)
        
        # Logarithmic field: first and last values
        assert fields[2].data.magnitude[0] == pytest.approx(10.0, rel=1e-6)
        assert fields[2].data.magnitude[-1] == pytest.approx(100.0, rel=1e-6)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
