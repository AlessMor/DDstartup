I/O Functions API Reference
============================

The ``utils/io_functions.py`` module provides all input/output operations for the DD Startup Analysis Tool. These functions handle file path resolution, configuration loading, parameter extraction, and data preparation.

Module Overview
---------------

The I/O functions module contains five main functions:

1. ``resolve_file_path()`` - Smart file path resolution with directory searching
2. ``load_config()`` - YAML configuration loading with validation
3. ``load_parameter_fields()`` - Dynamic Python module importing
4. ``prepare_input_data()`` - Data extraction with automatic unit conversion
5. ``print_configuration()`` - Configuration display utility

Import
------

.. code-block:: python

   # Direct import
   from ddstartup.utils.io_functions import (
       resolve_file_path,
       load_config,
       load_parameter_fields,
       prepare_input_data,
       print_configuration
   )
   
   # Or use auto-import
   from ddstartup.utils import resolve_file_path, load_config

Function Reference
------------------

resolve_file_path
~~~~~~~~~~~~~~~~~

.. function:: resolve_file_path(filename: str, default_dir: str, extensions: List[str]) -> Path

   Resolve file path with intelligent searching and extension handling.

   This function implements smart file resolution:
   
   * Checks if filename is absolute path
   * Searches in current directory
   * Searches in default directory
   * Tries adding each extension if missing
   
   :param filename: Name or path of file to find
   :type filename: str
   :param default_dir: Default directory to search (relative to main.py)
   :type default_dir: str
   :param extensions: List of valid extensions to try (e.g., ``['.py', '.yaml']``)
   :type extensions: List[str]
   
   :returns: Absolute path to resolved file
   :rtype: Path
   
   :raises FileNotFoundError: If file cannot be found in any location
   
   **Example:**
   
   .. code-block:: python
   
      from pathlib import Path
      from ddstartup.utils import resolve_file_path
      
      # Resolve Python config file
      config_path = resolve_file_path(
          filename="config_test",
          default_dir="inputs",
          extensions=[".py"]
      )
      # Returns: /full/path/to/inputs/config_test.py
      
      # Resolve YAML file
      yaml_path = resolve_file_path(
          filename="parametric_tseeded",
          default_dir="inputs",
          extensions=[".yaml", ".yml"]
      )
      # Returns: /full/path/to/inputs/parametric_tseeded.yaml
   
   **Search Strategy:**
   
   1. Check if ``filename`` is absolute path → return if exists
   2. Check current directory for ``filename`` → return if exists
   3. Check current directory for ``filename + extension`` → return if exists
   4. Check ``default_dir`` for ``filename`` → return if exists
   5. Check ``default_dir`` for ``filename + extension`` → return if exists
   6. Raise ``FileNotFoundError``
   
   **Notes:**
   
   * Case-sensitive on Linux/macOS
   * Handles both relative and absolute paths
   * Validates file existence before returning

load_config
~~~~~~~~~~~

.. function:: load_config(yaml_path: Path) -> Dict[str, Any]

   Load and validate YAML configuration file.

   Reads YAML configuration and validates required fields. Applies default values for optional fields.
   
   :param yaml_path: Path to YAML configuration file
   :type yaml_path: Path
   
   :returns: Configuration dictionary with all required and optional fields
   :rtype: Dict[str, Any]
   
   :raises FileNotFoundError: If YAML file doesn't exist
   :raises yaml.YAMLError: If YAML syntax is invalid
   :raises ValueError: If required fields are missing
   
   **Required Fields:**
   
   * ``analysis_type``: 'T_seeded' or 'lump'
   * ``method``: 'parametric', 'sobol', or 'lhs'
   
   **Optional Fields with Defaults:**
   
   * ``vector_length``: 100
   * ``n_jobs``: null (auto-detect)
   * ``chunk_size``: null (auto-detect)
   * ``batch_size``: null (auto-detect)
   * ``sobol_samples``: null (auto-detect)
   * ``sobol_order``: null (auto-detect)
   * ``verbose``: false
   
   **Example:**
   
   .. code-block:: python
   
      from pathlib import Path
      from ddstartup.utils import load_config
      
      # Load configuration
      config_path = Path("inputs/parametric_tseeded.yaml")
      config = load_config(config_path)
      
      # Access values
      print(config['analysis_type'])  # 'T_seeded'
      print(config['method'])         # 'parametric'
      print(config['vector_length'])  # 100
      print(config['n_jobs'])         # None (will be auto-detected)
   
   **YAML File Format:**
   
   .. code-block:: yaml
   
      # parametric_tseeded.yaml
      analysis_type: T_seeded
      method: parametric
      vector_length: 100
      n_jobs: null
      verbose: true
   
   **Notes:**
   
   * Uses ``yaml.safe_load()`` for security
   * Validates immediately after loading
   * Provides helpful error messages
   * null/None values trigger auto-detection

load_parameter_fields
~~~~~~~~~~~~~~~~~~~~~

.. function:: load_parameter_fields(param_file_path: Path) -> Dict[str, Any]

   Dynamically import parameter module and extract ParameterField objects.

   This function imports a Python module and extracts all ``ParameterField`` objects defined in it.
   
   :param param_file_path: Path to parameter definition file (Python module)
   :type param_file_path: Path
   
   :returns: Dictionary mapping field names to ParameterField objects
   :rtype: Dict[str, Any]
   
   :raises FileNotFoundError: If parameter file doesn't exist
   :raises ImportError: If module import fails
   :raises ValueError: If no ParameterField objects found
   
   **Example:**
   
   .. code-block:: python
   
      from pathlib import Path
      from ddstartup.utils import load_parameter_fields
      
      # Load parameter fields
      param_path = Path("inputs/config_test.py")
      param_fields = load_parameter_fields(param_path)
      
      # Access fields
      print(param_fields.keys())
      # dict_keys(['V_plasma_field', 'T_i_field', 'n_tot_field', ...])
      
      # Use field data
      v_plasma = param_fields['V_plasma_field']
      print(v_plasma.name)  # 'plasma_volume'
      print(v_plasma.unit)  # <Unit('meter ** 3')>
      print(v_plasma.data)  # <Quantity(150.0, 'meter ** 3')>
   
   **Parameter File Format:**
   
   .. code-block:: python
   
      # inputs/config_test.py
      from ddstartup.utils import u, ParameterField
      import numpy as np
      
      V_plasma_field = ParameterField(
          unit=u.m**3,
          name="plasma_volume",
          parametrization_type="normal",
          mean=150.0,
          param_points=1
      )
      
      T_i_field = ParameterField(
          unit=u.keV,
          name="ion_temperature",
          parametrization_type="normal",
          mean=17.0,
          param_points=1
      )
      
      # ... more fields
   
   **Notes:**
   
   * Uses dynamic module importing (``importlib``)
   * Automatically detects all ``ParameterField`` instances
   * Validates that at least one field exists
   * Returns dictionary for easy access by name

prepare_input_data
~~~~~~~~~~~~~~~~~~

.. function:: prepare_input_data(param_fields: Dict[str, Any], analysis_type: str) -> Dict[str, np.ndarray]

   Prepare input data dictionary by extracting and converting parameter data.

   Extracts data from ParameterField objects, converts to SI units, and returns numpy arrays ready for computation.
   
   :param param_fields: Dictionary of ParameterField objects (from ``load_parameter_fields``)
   :type param_fields: Dict[str, Any]
   :param analysis_type: Type of analysis ('T_seeded' or 'lump')
   :type analysis_type: str
   
   :returns: Dictionary mapping parameter names to numpy arrays (SI units, magnitude only)
   :rtype: Dict[str, np.ndarray]
   
   :raises ValueError: If analysis_type is not recognized
   :raises KeyError: If required field is missing
   
   **Example:**
   
   .. code-block:: python
   
      from ddstartup.utils import load_parameter_fields, prepare_input_data
      
      # Load parameter fields
      param_fields = load_parameter_fields("inputs/config_test.py")
      
      # Extract data for T_seeded analysis
      input_data = prepare_input_data(param_fields, 'T_seeded')
      
      # Access numpy arrays (SI units)
      print(input_data['V_plasma'])  # array([150.]) in m^3
      print(input_data['T_i'])       # array([17.]) in keV
      print(input_data['n_tot'])     # array([1.7e+20]) in 1/m^3
      
      # Ready for computation
      import numpy as np
      total_particles = input_data['n_tot'] * input_data['V_plasma']
   
   **Analysis Types:**
   
   **T_seeded** (13 parameters):
   
   * ``V_plasma`` (m³) - Plasma volume
   * ``T_i`` (keV) - Ion temperature
   * ``n_tot`` (1/m³) - Total density
   * ``tau_p_T`` (s) - Tritium particle confinement time
   * ``P_aux`` (W) - Auxiliary heating power
   * ``P_aux_DT_eq`` (W) - Auxiliary power at DT equilibrium
   * ``TBR_DT`` (dimensionless) - DT tritium breeding ratio
   * ``TBR_DDn`` (dimensionless) - DDn tritium breeding ratio
   * ``tau_ifc`` (s) - Inboard first wall campaign time
   * ``tau_ofc`` (s) - Outboard first wall campaign time
   * ``eta_th`` (dimensionless) - Thermal efficiency
   * ``capacity_factor`` (dimensionless) - Plant capacity factor
   * ``price_of_electricity`` ($/J) - Price of electricity
   
   **lump** (13 parameters, includes He3):
   
   All parameters from T_seeded, plus:
   
   * ``tau_p_He3`` (s) - He3 particle confinement time
   * ``I_target`` (A) - Target current (replaces P_aux_DT_eq)
   
   **Unit Conversions:**
   
   The function automatically converts all quantities to SI units:
   
   .. code-block:: python
   
      # ParameterField stores with units
      T_i_field.data = Quantity(17.0, 'keV')
      
      # prepare_input_data extracts magnitude in target units
      input_data['T_i'] = T_i_field.data.to('keV').magnitude
      # Returns: array([17.0])
   
   **Notes:**
   
   * All units converted to SI automatically
   * Returns magnitudes only (no units) for numpy operations
   * Different parameters for different analysis types
   * Validates analysis_type before processing

print_configuration
~~~~~~~~~~~~~~~~~~~

.. function:: print_configuration(config: Dict, param_fields: Dict, input_data: Dict, param_file_path: Path, config_file_path: Path) -> None

   Print formatted configuration summary.

   Displays comprehensive configuration information including file paths, settings, and parameter statistics.
   
   :param config: Configuration dictionary from ``load_config()``
   :type config: Dict[str, Any]
   :param param_fields: Parameter fields dictionary from ``load_parameter_fields()``
   :type param_fields: Dict[str, Any]
   :param input_data: Input data dictionary from ``prepare_input_data()``
   :type input_data: Dict[str, np.ndarray]
   :param param_file_path: Path to parameter file
   :type param_file_path: Path
   :param config_file_path: Path to configuration file
   :type config_file_path: Path
   
   :returns: None (prints to stdout)
   :rtype: None
   
   **Example:**
   
   .. code-block:: python
   
      from ddstartup.utils import (
          resolve_file_path,
          load_config,
          load_parameter_fields,
          prepare_input_data,
          print_configuration
      )
      
      # Load everything
      param_path = resolve_file_path("config_test", "inputs", [".py"])
      config_path = resolve_file_path("parametric_tseeded", "inputs", [".yaml"])
      
      param_fields = load_parameter_fields(param_path)
      config = load_config(config_path)
      input_data = prepare_input_data(param_fields, config['analysis_type'])
      
      # Print summary
      print_configuration(config, param_fields, input_data, 
                         param_path, config_path)
   
   **Output Format:**
   
   .. code-block:: text
   
      ============================================================
      DD STARTUP ANALYSIS CONFIGURATION
      ============================================================
      Parameter file: /path/to/inputs/config_test.py
      Config file: /path/to/inputs/parametric_tseeded.yaml
      Analysis type: T_seeded
      Method: parametric
      
      System Profiling:
        CPU Cores: 12 physical, 24 logical
        RAM Available: 31.2 GB
        Recommended n_jobs: 11
        Recommended chunk_size: 5500
        Recommended batch_size: 100
      
      Input parameter fields:
        V_plasma            : shape=(1,), range=[1.500e+02, 1.500e+02]
        T_i                 : shape=(1,), range=[1.700e+01, 1.700e+01]
        n_tot               : shape=(1,), range=[1.700e+20, 1.700e+20]
        ...
      
      Total parameter combinations: 256
      ============================================================
   
   **Notes:**
   
   * Only prints if ``config['verbose']`` is True
   * Shows system profiling information
   * Displays each parameter with shape and range
   * Calculates total parameter combinations

create_output_directory
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: create_output_directory(base_dir: str, timestamp: str, analysis_method: str, analysis_type: str) -> Path

   Create output directory with timestamp and analysis information.

   Creates a directory structure: ``base_dir/timestamp_method_type/``
   
   :param base_dir: Base output directory (e.g., 'outputs')
   :type base_dir: str
   :param timestamp: Timestamp string (e.g., '20251006_123045')
   :type timestamp: str
   :param analysis_method: Analysis method ('parametric', 'sobol', 'lhs')
   :type analysis_method: str
   :param analysis_type: Analysis type ('T_seeded', 'lump')
   :type analysis_type: str
   
   :returns: Path object to the created directory
   :rtype: Path
   
   :raises OSError: If directory cannot be created
   
   **Example:**
   
   .. code-block:: python
   
      from ddstartup.utils import create_output_directory
      
      output_dir = create_output_directory(
          base_dir='outputs',
          timestamp='20251006_123045',
          analysis_method='parametric',
          analysis_type='T_seeded'
      )
      
      print(output_dir)
      # outputs/20251006_123045_parametric_T_seeded
   
   **Directory Naming Convention:**
   
   The directory name follows the pattern: ``{timestamp}_{method}_{type}``
   
   Examples:
   
   * ``20251006_123045_parametric_T_seeded``
   * ``20251006_140530_sobol_lump``
   * ``20251006_093022_lhs_T_seeded``
   
   **Features:**
   
   * Creates parent directories automatically (``parents=True``)
   * Does not raise error if directory already exists (``exist_ok=True``)
   * Returns absolute path to created directory
   * Validates that directory is writable

generate_output_path
~~~~~~~~~~~~~~~~~~~~

.. function:: generate_output_path(base_dir: str = 'outputs', analysis_method: str = 'parametric', analysis_type: str = 'T_seeded', timestamp: Optional[str] = None) -> Tuple[Path, str]

   Generate output directory and file path for HDF5 results.

   Creates directory structure and generates full path to output file:
   ``base_dir/timestamp_method_type/ddstartup_timestamp_method_type.h5``
   
   :param base_dir: Base output directory (default: 'outputs')
   :type base_dir: str
   :param analysis_method: Analysis method ('parametric', 'sobol', 'lhs')
   :type analysis_method: str
   :param analysis_type: Analysis type ('T_seeded', 'lump')
   :type analysis_type: str
   :param timestamp: Optional timestamp string. If None, generates current timestamp
   :type timestamp: Optional[str]
   
   :returns: Tuple of (output_directory_path, full_output_file_path)
   :rtype: Tuple[Path, str]
   
   **Example:**
   
   .. code-block:: python
   
      from ddstartup.utils import generate_output_path
      
      # Auto-generate timestamp
      output_dir, output_file = generate_output_path(
          base_dir='outputs',
          analysis_method='parametric',
          analysis_type='T_seeded'
      )
      
      print(output_dir)
      # outputs/20251006_123045_parametric_T_seeded
      
      print(output_file)
      # outputs/20251006_123045_parametric_T_seeded/ddstartup_20251006_123045_parametric_T_seeded.h5
      
      # Use custom timestamp
      output_dir, output_file = generate_output_path(
          base_dir='results',
          analysis_method='sobol',
          analysis_type='lump',
          timestamp='20251006_120000'
      )
   
   **File Naming Convention:**
   
   The output file follows the pattern: ``ddstartup_{timestamp}_{method}_{type}.h5``
   
   Examples:
   
   * ``ddstartup_20251006_123045_parametric_T_seeded.h5``
   * ``ddstartup_20251006_140530_sobol_lump.h5``
   * ``ddstartup_20251006_093022_lhs_T_seeded.h5``
   
   **Timestamp Format:**
   
   Timestamps follow the pattern: ``YYYYMMDD_HHMMSS``
   
   * ``YYYY``: 4-digit year (e.g., 2025)
   * ``MM``: 2-digit month (01-12)
   * ``DD``: 2-digit day (01-31)
   * ``HH``: 2-digit hour (00-23, 24-hour format)
   * ``MM``: 2-digit minute (00-59)
   * ``SS``: 2-digit second (00-59)
   
   **Return Values:**
   
   The function returns a tuple containing:
   
   1. **output_directory_path** (``Path``): Path object to the created directory
   2. **full_output_file_path** (``str``): Complete string path to the HDF5 file
   
   **Notes:**
   
   * Automatically creates output directory
   * Generates unique timestamp if not provided
   * Returns string path for direct use with h5py.File()
   * Directory structure keeps results organized by date and analysis type

Data Extraction Workflow
-------------------------

Complete Data Flow
~~~~~~~~~~~~~~~~~~

The typical data extraction workflow follows these steps:

.. code-block:: python

   from pathlib import Path
   from ddstartup.utils import (
       resolve_file_path,
       load_config,
       load_parameter_fields,
       prepare_input_data,
       print_configuration
   )
   
   # Step 1: Resolve file paths
   param_path = resolve_file_path("config_test", "inputs", [".py"])
   config_path = resolve_file_path("parametric_tseeded", "inputs", [".yaml"])
   
   # Step 2: Load configuration
   config = load_config(config_path)
   
   # Step 3: Load parameter fields
   param_fields = load_parameter_fields(param_path)
   
   # Step 4: Extract input data with unit conversion
   input_data = prepare_input_data(param_fields, config['analysis_type'])
   
   # Step 5: Display configuration (if verbose)
   if config['verbose']:
       print_configuration(config, param_fields, input_data,
                          param_path, config_path)
   
   # Step 6: Use data for analysis
   # ... your analysis code here

Command-Line Flow
~~~~~~~~~~~~~~~~~

When using the command-line interface:

.. code-block:: bash

   $ python main.py config_test parametric_tseeded --verbose --dry-run

The flow is:

1. ``parse_arguments()`` → Parse command-line arguments
2. ``resolve_file_path()`` → Find config and parameter files
3. ``load_config()`` → Load YAML configuration
4. ``load_parameter_fields()`` → Import parameter module
5. ``prepare_input_data()`` → Extract and convert data
6. ``print_configuration()`` → Display summary (if verbose)
7. Exit (dry run) or continue to analysis

Unit Conversion Details
------------------------

Automatic Conversion
~~~~~~~~~~~~~~~~~~~~

The ``prepare_input_data()`` function automatically converts all quantities to SI units using Pint:

.. code-block:: python

   # ParameterField definition (with units)
   T_i_field = ParameterField(
       unit=u.keV,
       name="ion_temperature",
       data=Quantity(17.0, 'keV')
   )
   
   # Automatic conversion in prepare_input_data()
   input_data['T_i'] = T_i_field.data.to('keV').magnitude
   
   # Result: numpy array with magnitude only
   # array([17.0])

Supported Units
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Unit
     - Conversion
   * - V_plasma
     - m³
     - ``.to('m^3')``
   * - T_i
     - keV
     - ``.to('keV')``
   * - n_tot
     - 1/m³
     - ``.to('1/m^3')``
   * - tau_p_T, tau_p_He3
     - s
     - ``.to('s')``
   * - P_aux, P_aux_DT_eq
     - W
     - ``.to('W')``
   * - TBR_DT, TBR_DDn
     - dimensionless
     - ``.to('dimensionless')``
   * - tau_ifc, tau_ofc
     - s
     - ``.to('s')``
   * - eta_th
     - dimensionless
     - ``.to('dimensionless')``
   * - capacity_factor
     - dimensionless
     - ``.to('dimensionless')``
   * - price_of_electricity
     - $/J
     - ``.to('dollar/J')``
   * - I_target
     - A
     - ``.to('A')``

Error Handling
--------------

The I/O functions include comprehensive error handling:

File Not Found
~~~~~~~~~~~~~~

.. code-block:: python

   try:
       path = resolve_file_path("nonexistent", "inputs", [".py"])
   except FileNotFoundError as e:
       print(f"Error: {e}")
       # Error: Could not find file 'nonexistent' in:
       #   - Current directory
       #   - inputs/

Invalid Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       config = load_config(Path("invalid_config.yaml"))
   except ValueError as e:
       print(f"Error: {e}")
       # Error: Missing required field: analysis_type

Missing Parameter Field
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       input_data = prepare_input_data(incomplete_fields, 'T_seeded')
   except KeyError as e:
       print(f"Error: Missing required field: {e}")
       # Error: Missing required field: 'V_plasma_field'

Best Practices
--------------

1. **Always validate files before loading:**

   .. code-block:: python
   
      if not config_path.exists():
          raise FileNotFoundError(f"Config not found: {config_path}")

2. **Use type hints for clarity:**

   .. code-block:: python
   
      def my_function(config: Dict[str, Any]) -> None:
          pass

3. **Handle errors gracefully:**

   .. code-block:: python
   
      try:
          config = load_config(config_path)
      except ValueError as e:
          print(f"Configuration error: {e}")
          sys.exit(1)

4. **Provide helpful error messages:**

   .. code-block:: python
   
      if 'analysis_type' not in config:
          raise ValueError(
              "Missing 'analysis_type' in configuration. "
              "Must be 'T_seeded' or 'lump'"
          )

5. **Use Path objects for file operations:**

   .. code-block:: python
   
      from pathlib import Path
      config_path = Path("inputs/config.yaml")

See Also
--------

* :doc:`../user_guide/configuration_files` - Configuration file format
* :doc:`../user_guide/parameter_definitions` - Parameter definitions
* :doc:`../developer_guide/architecture` - System architecture
