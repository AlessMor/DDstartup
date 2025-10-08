"""
Tests for main.py

This module tests the main entry point including:
- Argument parsing
- Integration of I/O functions
- Error handling
- Dry run mode
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddstartup.main import parse_arguments, main


class TestParseArguments:
    """Tests for parse_arguments function"""
    
    def test_parse_required_arguments(self):
        """Test parsing required arguments"""
        test_args = ['main.py', 'config', 'parametric_tseeded']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.params == 'config'
            assert args.config == 'parametric_tseeded'
            assert args.verbose == False
            assert args.dry_run == False
    
    def test_parse_with_verbose_flag(self):
        """Test parsing with --verbose flag"""
        test_args = ['main.py', 'config', 'parametric_tseeded', '--verbose']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.verbose == True
    
    def test_parse_with_dry_run_flag(self):
        """Test parsing with --dry-run flag"""
        test_args = ['main.py', 'config', 'parametric_tseeded', '--dry-run']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.dry_run == True
    
    def test_parse_with_both_flags(self):
        """Test parsing with both --verbose and --dry-run"""
        test_args = ['main.py', 'config', 'parametric_tseeded', '--verbose', '--dry-run']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.verbose == True
            assert args.dry_run == True
    
    def test_parse_with_full_paths(self):
        """Test parsing with full file paths"""
        test_args = ['main.py', 'inputs/config.py', 'inputs/test.yaml']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_arguments()
            assert args.params == 'inputs/config.py'
            assert args.config == 'inputs/test.yaml'
    
    def test_parse_missing_arguments(self):
        """Test that missing arguments raise SystemExit"""
        test_args = ['main.py', 'config']  # Missing second argument
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestMainFunction:
    """Tests for main function"""
    
    def test_main_file_not_found(self, capsys):
        """Test main function with nonexistent files"""
        test_args = ['main.py', 'nonexistent', 'nonexistent']
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
            
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "Error" in captured.err
    
    def test_main_dry_run_mode(self, temp_dir, sample_yaml_file, capsys):
        """Test main function in dry-run mode"""
        # Create a simple param file
        param_file = temp_dir / "test_params.py"
        param_content = '''
# Minimal param file for testing
V_plasma_field = None
T_i_field = None
n_tot_field = None
tau_p_T_field = None
tau_p_He3_field = None
P_aux_field = None
P_aux_DT_eq_field = None
TBR_DT_field = None
TBR_DDn_field = None
tau_ifc_field = None
tau_ofc_field = None
eta_th_field = None
capacity_factor_field = None
cost_of_electricity_field = None
I_target_field = None
total_time = None
'''
        with open(param_file, 'w') as f:
            f.write(param_content)
        
        test_args = [
            'main.py',
            str(param_file),
            str(sample_yaml_file),
            '--dry-run'
        ]
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
            
            captured = capsys.readouterr()
            assert exit_code == 0
            assert "Dry run completed" in captured.out
    
    def test_main_verbose_override(self, temp_dir, sample_yaml_file):
        """Test that --verbose flag overrides config setting"""
        # Create config with verbose=False
        import yaml
        config_data = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'verbose': False
        }
        yaml_file = temp_dir / "test.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create minimal param file
        param_file = temp_dir / "params.py"
        with open(param_file, 'w') as f:
            f.write("total_time = None\n")
            for field in ['V_plasma_field', 'T_i_field', 'n_tot_field']:
                f.write(f"{field} = None\n")
        
        test_args = [
            'main.py',
            str(param_file),
            str(yaml_file),
            '--verbose',
            '--dry-run'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('ddstartup.main.load_parameter_fields') as mock_load:
                with patch('ddstartup.main.prepare_input_data') as mock_prepare:
                    mock_load.return_value = {
                        'total_time': None,
                        'V_plasma_field': None
                    }
                    mock_prepare.return_value = {}
                    
                    exit_code = main()
                    # The verbose flag should have been set to True
                    assert exit_code == 0
    
    def test_main_invalid_yaml_config(self, temp_dir, capsys):
        """Test main with invalid YAML configuration"""
        # Create param file
        param_file = temp_dir / "params.py"
        param_file.write_text("total_time = None")
        
        # Create invalid YAML (missing required fields)
        yaml_file = temp_dir / "bad.yaml"
        yaml_file.write_text("verbose: true\n")  # Missing analysis_type and method
        
        test_args = ['main.py', str(param_file), str(yaml_file)]
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
            
            assert exit_code == 1
            captured = capsys.readouterr()
            assert "Error loading configuration" in captured.err
    
    def test_main_import_error(self, temp_dir, sample_yaml_file, capsys):
        """Test main with parameter import error"""
        # Point to nonexistent module
        test_args = ['main.py', 'nonexistent_module', str(sample_yaml_file)]
        
        with patch.object(sys, 'argv', test_args):
            exit_code = main()
            
            assert exit_code == 1
            captured = capsys.readouterr()
            # Should have an error message
            assert "Error" in captured.err
    
    def test_main_successful_config_load(self, temp_dir, capsys):
        """Test successful configuration loading"""
        import yaml
        
        # Create valid YAML
        yaml_data = {
            'analysis_type': 'T_seeded',
            'method': 'parametric'
        }
        yaml_file = temp_dir / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_data, f)
        
        # Create valid param file (mock)
        param_file = temp_dir / "params.py"
        param_file.write_text("total_time = None\n")
        
        test_args = ['main.py', str(param_file), str(yaml_file), '--verbose']
        
        with patch.object(sys, 'argv', test_args):
            with patch('ddstartup.main.load_parameter_fields') as mock_load:
                with patch('ddstartup.main.prepare_input_data') as mock_prepare:
                    with patch('ddstartup.main.print_configuration'):
                        with patch('ddstartup.main.run_parametric_analysis') as mock_analysis:
                            with patch('ddstartup.main.print_parametric_summary'):
                                mock_load.return_value = {'total_time': None}
                                mock_prepare.return_value = {}
                                mock_analysis.return_value = {
                                    'n_combinations': 1,
                                    'n_success': 1,
                                    'success_rate': 100.0
                                }
                                
                                exit_code = main()
                                
                                assert exit_code == 0
                                # Just check that analysis was called
                                assert mock_analysis.called


class TestMainIntegration:
    """Integration tests for main function"""
    
    def test_full_workflow_dry_run(self, temp_dir):
        """Test complete workflow in dry-run mode"""
        import yaml
        
        # Setup complete test environment
        inputs_dir = temp_dir / "inputs"
        inputs_dir.mkdir()
        
        # Create YAML config
        yaml_data = {
            'analysis_type': 'T_seeded',
            'method': 'parametric',
            'verbose': True
        }
        yaml_file = inputs_dir / "test_config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_data, f)
        
        # Create parameter file with minimal content
        param_file = inputs_dir / "test_params.py"
        param_content = """
# Test parameters
total_time = 31536000
V_plasma_field = None
T_i_field = None
n_tot_field = None
tau_p_T_field = None
tau_p_He3_field = None
P_aux_field = None
P_aux_DT_eq_field = None
TBR_DT_field = None
TBR_DDn_field = None
tau_ifc_field = None
tau_ofc_field = None
eta_th_field = None
capacity_factor_field = None
cost_of_electricity_field = None
I_target_field = None
"""
        with open(param_file, 'w') as f:
            f.write(param_content)
        
        test_args = [
            'main.py',
            str(param_file),
            str(yaml_file),
            '--dry-run'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('ddstartup.main.prepare_input_data') as mock_prepare:
                mock_prepare.return_value = {}
                exit_code = main()
                assert exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
