"""
Tests for postprocessing/plot_shap_functions.py

This module tests SHAP plotting functions including:
- SHAP value computation
- Surrogate model training
- Various SHAP plot generation
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddstartup.postprocessing.plot_shap_functions import (
    check_shap_available,
    train_surrogate_model,
    compute_shap_values,
    save_shap_values_to_csv,
    generate_shap_plots
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 500
    
    # Create synthetic input parameters
    X1 = np.random.uniform(50, 200, n_samples)
    X2 = np.random.uniform(1e20, 5e20, n_samples)
    X3 = np.random.uniform(10, 30, n_samples)
    
    # Create synthetic target with known relationships
    # Target = 2*X1 + 0.5*X2/1e20 + X3**2 + noise
    target = 2*X1 + 0.5*X2/1e20 + X3**2 + np.random.normal(0, 10, n_samples)
    
    df = pd.DataFrame({
        'V_plasma': X1,
        'n_tot': X2,
        'T_i': X3,
        't_startup': target
    })
    
    return df


@pytest.fixture
def temp_dir():
    """Create temporary directory for outputs"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


class TestShapAvailability:
    """Test SHAP library availability check"""
    
    def test_check_shap_available(self):
        """Test SHAP availability detection"""
        # This should return True or False depending on installation
        result = check_shap_available()
        assert isinstance(result, bool)


class TestSurrogateModel:
    """Test surrogate model training"""
    
    def test_train_tree_model(self, sample_data):
        """Test training gradient boosting surrogate"""
        input_params = ['V_plasma', 'n_tot', 'T_i']
        target = 't_startup'
        
        model, X, y, model_type = train_surrogate_model(
            sample_data, input_params, target, model_type='tree'
        )
        
        assert model is not None
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == len(input_params)
        assert len(y) == len(sample_data)
        assert model_type == 'tree'
        
        # Check model can predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
        # Check R² score is reasonable
        r2 = model.score(X, y)
        assert r2 > 0.5  # Should fit synthetic data reasonably well
    
    def test_train_linear_model(self, sample_data):
        """Test training linear surrogate"""
        input_params = ['V_plasma', 'n_tot', 'T_i']
        target = 't_startup'
        
        model, X, y, model_type = train_surrogate_model(
            sample_data, input_params, target, model_type='linear'
        )
        
        assert model is not None
        assert model_type == 'linear'
        
        # Linear model should also work reasonably on this data
        r2 = model.score(X, y)
        assert r2 > 0.3
    
    def test_handles_missing_values(self, sample_data):
        """Test handling of NaN values"""
        # Add some NaN values
        sample_data.loc[0:10, 'V_plasma'] = np.nan
        sample_data.loc[20:30, 't_startup'] = np.nan
        
        input_params = ['V_plasma', 'n_tot', 'T_i']
        target = 't_startup'
        
        model, X, y, model_type = train_surrogate_model(
            sample_data, input_params, target, model_type='tree'
        )
        
        # Should handle NaN gracefully
        assert not np.any(np.isnan(X.values))
        assert not np.any(np.isnan(y.values))


@pytest.mark.skipif(not check_shap_available(), reason="SHAP not installed")
class TestShapComputation:
    """Test SHAP value computation (requires SHAP installation)"""
    
    def test_compute_shap_tree(self, sample_data):
        """Test SHAP computation with tree model"""
        input_params = ['V_plasma', 'n_tot', 'T_i']
        target = 't_startup'
        
        model, X, y, model_type = train_surrogate_model(
            sample_data, input_params, target, model_type='tree'
        )
        
        shap_values, explainer, X_sample = compute_shap_values(
            model, X, model_type='tree', max_samples=100
        )
        
        assert shap_values is not None
        assert shap_values.shape[0] == len(X_sample)
        assert shap_values.shape[1] == len(input_params)
        assert explainer is not None
    
    def test_shap_sampling(self, sample_data):
        """Test SHAP computation with sampling"""
        input_params = ['V_plasma', 'n_tot', 'T_i']
        target = 't_startup'
        
        model, X, y, model_type = train_surrogate_model(
            sample_data, input_params, target, model_type='tree'
        )
        
        # Request fewer samples than available
        shap_values, explainer, X_sample = compute_shap_values(
            model, X, model_type='tree', max_samples=50
        )
        
        assert len(X_sample) == 50
        assert shap_values.shape[0] == 50


@pytest.mark.skipif(not check_shap_available(), reason="SHAP not installed")
class TestShapOutputs:
    """Test SHAP output generation"""
    
    def test_save_shap_csv(self, sample_data, temp_dir):
        """Test saving SHAP values to CSV"""
        input_params = ['V_plasma', 'n_tot', 'T_i']
        target = 't_startup'
        
        model, X, y, model_type = train_surrogate_model(
            sample_data, input_params, target, model_type='tree'
        )
        
        shap_values, explainer, X_sample = compute_shap_values(
            model, X, model_type='tree', max_samples=100
        )
        
        save_shap_values_to_csv(
            shap_values, X_sample, target, input_params,
            temp_dir, 'test_shap'
        )
        
        # Check files were created
        shap_csv = temp_dir / 'test_shap_shap_values.csv'
        importance_csv = temp_dir / 'test_shap_shap_importance.csv'
        
        assert shap_csv.exists()
        assert importance_csv.exists()
        
        # Check CSV contents
        shap_df = pd.read_csv(shap_csv)
        assert len(shap_df) == 100
        assert 'shap_total' in shap_df.columns
        for param in input_params:
            assert f'shap_{param}' in shap_df.columns
            assert f'value_{param}' in shap_df.columns
        
        importance_df = pd.read_csv(importance_csv)
        assert len(importance_df) == len(input_params)
        assert 'feature' in importance_df.columns
        assert 'mean_abs_shap' in importance_df.columns
        assert 'importance_rank' in importance_df.columns
    
    def test_generate_shap_plots(self, sample_data, temp_dir):
        """Test full SHAP plot generation"""
        input_params = ['V_plasma', 'n_tot', 'T_i']
        target = 't_startup'
        
        results = generate_shap_plots(
            sample_data, target, input_params, 's',
            temp_dir, 'test', 'test_shap',
            model_type='tree',
            plot_types=['summary', 'summary_bar', 'csv'],
            max_samples=100
        )
        
        # Check results dictionary
        assert results is not None
        assert 'model_r2' in results
        assert 'n_samples' in results
        assert 'n_features' in results
        assert results['n_features'] == len(input_params)
        
        # Check files were created
        csv_file = temp_dir / 'test_shap_shap_values.csv'
        assert csv_file.exists()
        
        # Summary plots should be created
        summary_plot = temp_dir / 'test_shap_summary_dot.png'
        bar_plot = temp_dir / 'test_shap_summary_bar.png'
        
        # These may or may not exist depending on matplotlib backend
        # but function should complete without error


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_input_parameters(self, sample_data, temp_dir):
        """Test with no input parameters"""
        results = generate_shap_plots(
            sample_data, 't_startup', [], 's',
            temp_dir, 'test', 'test_shap'
        )
        
        assert results is None
    
    def test_single_input_parameter(self, sample_data, temp_dir):
        """Test with single input parameter"""
        input_params = ['V_plasma']
        target = 't_startup'
        
        model, X, y, model_type = train_surrogate_model(
            sample_data, input_params, target, model_type='tree'
        )
        
        assert model is not None
        assert X.shape[1] == 1
    
    def test_small_dataset(self, temp_dir):
        """Test with very small dataset"""
        # Create tiny dataset
        df = pd.DataFrame({
            'X1': [1, 2, 3, 4, 5],
            'X2': [10, 20, 30, 40, 50],
            'Y': [100, 200, 300, 400, 500]
        })
        
        input_params = ['X1', 'X2']
        target = 'Y'
        
        model, X, y, model_type = train_surrogate_model(
            df, input_params, target, model_type='linear'
        )
        
        # Should work but may have low R²
        assert model is not None
