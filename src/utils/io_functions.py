"""
I/O functions for DD Startup analysis tool.

This module contains functions for:
- File path resolution
- Configuration loading and validation
- Parameter field loading
- Input data preparation
- Configuration display
- Output directory and file creation
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from tqdm import tqdm

from .system_profiler import apply_parallelization_defaults
from .parameter_registry import (
    PARAMETER_SCHEMA,
    MULTISPECIES_PARAMETER_SCHEMA,
    SPECIES,
    get_registry,
    get_multispecies_registry,
)
from .units_and_constants import u  # Pint UnitRegistry


PathLike = Union[str, Path]


def _normalize_analysis_type(analysis_type: Optional[str]) -> Optional[str]:
    """Normalize analysis type aliases used across code paths."""
    if analysis_type is None:
        return None
    return str(analysis_type).strip()


def _is_multispecies_analysis(analysis_type: Optional[str]) -> bool:
    """Return True when analysis_type targets the merged multispecies model."""
    return _normalize_analysis_type(analysis_type) == "multispecies"

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='DD Startup Analysis Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'params',
        type=str,
        help='YAML parameter file name (e.g., "my_parameters") or path to file (e.g., "inputs/my_parameters.yaml")'
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='YAML configuration file name (e.g., "parametric_tseeded") or path to file (e.g., "run_configs/parametric.yaml")'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without running analysis'
    )
    
    return parser.parse_args()

def resolve_file_path(filename: str, default_dir: str, extension: Optional[str] = None, extensions: Optional[Union[List[str], Tuple[str, ...]]] = None,) -> Path:
    """
    Resolve file path - check if it's a direct path or needs default directory.

    Args:
        filename: File name or path
        default_dir: Default directory to search in (e.g., 'inputs')
        extension: Optional single file extension to try (e.g., '.yaml' or 'yaml')
        extensions: Optional list/tuple of extensions to try (ignored if extension is provided)

    Returns:
        Path object to the file

    Raises:
        FileNotFoundError: If file cannot be found in any of the expected locations
    """
    # 1) As given
    p = Path(filename)
    if p.exists():
        return p.resolve()

    stem = p.stem
    candidates = []
    ext_list: List[str] = []
    if extension:
        ext_list = [extension]
    elif extensions:
        ext_list = list(extensions)
    # 2) With optional extension(s) in default_dir and ../default_dir
    for ext in ext_list:
        ext = ext if ext.startswith('.') else f'.{ext}'
        candidates.extend([
            Path(default_dir) / f"{stem}{ext}",
            Path('..') / default_dir / f"{stem}{ext}",
        ])
    # 3) Raw filename inside default_dir and ../default_dir
    candidates.extend([
        Path(default_dir) / filename,
        Path('..') / default_dir / filename,
    ])
    tried = []
    for c in candidates:
        tried.append(str(c))
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"File not found: {filename}\nSearched paths:\n - " + "\n - ".join(tried))


def load_config(yaml_path: Path) -> Dict[str, Any]:
    """
    Load and validate YAML configuration.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration with defaults applied
        
    Raises:
        ValueError: If required fields are missing
        yaml.YAMLError: If YAML file is malformed
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config)!r}")

    # Required fields
    for field in ("analysis_type", "method"):
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    # Multispecies model path.
    if _is_multispecies_analysis(config.get("analysis_type")):
        return _load_multispecies_config(yaml_path)

    # Static defaults (non-performance)
    static_defaults = {
        "vector_length": 100,
        "max_simulation_time": 10 * 365 * 24 * 3600,
        "verbose": False,
        "output_dir": "outputs",
        "filter": None,  # parameter filter expression
    }
    for key, default in static_defaults.items():
        config.setdefault(key, default)

    # Parallelization-related keys: None means "auto"
    for key in ("n_jobs", "chunk_size", "batch_size", "N_SAMPLES", "order"):
        config.setdefault(key, None)

    # Fill in parallelization defaults based on system profiling
    config = apply_parallelization_defaults(config, verbose=config["verbose"])

    return config


def load_params(yaml_path: Path, analysis_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Load parameters from YAML, convert to canonical units from PARAMETER_SCHEMA,
    and validate that required inputs for the requested analysis type exist.

    Returns:
        dict[base_param_name] = (values_in_default_unit, unit_str, metadata_dict)
    """

    def _parse_numeric(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value.lower()
            if v in {"nan", ".nan", "null", "none"}:
                return float("nan")
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Cannot parse numeric value: {value!r}")
        raise ValueError(f"Cannot parse numeric value of type {type(value)}: {value!r}")

    normalized_analysis_type = _normalize_analysis_type(analysis_type)
    if _is_multispecies_analysis(normalized_analysis_type):
        return _load_multispecies_params(yaml_path, analysis_type="multispecies")

    registry = get_registry()

    if not yaml_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {yaml_path}")
    if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Parameter file must be YAML (.yaml/.yml), got: {yaml_path.suffix}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if "parameters" not in cfg or not isinstance(cfg["parameters"], dict):
        raise ValueError("YAML file must contain top-level 'parameters' mapping")

    params_cfg = cfg["parameters"]
    result: Dict[str, Any] = {}

    def _values_from_definition(field: str, kind: str, pts: int, definition: Dict[str, Any]) -> np.ndarray:
        if kind == "scalar":
            if "value" not in definition:
                raise ValueError(f"Scalar parameter '{field}' must have 'value'")
            return np.full(pts, _parse_numeric(definition["value"]), dtype=float)
        if kind == "linear":
            if "min" not in definition or "max" not in definition:
                raise ValueError(f"Linear parameter '{field}' must have 'min' and 'max'")
            vmin = _parse_numeric(definition["min"])
            vmax = _parse_numeric(definition["max"])
            return np.array([(vmin + vmax) / 2.0], dtype=float) if pts == 1 else np.linspace(vmin, vmax, pts, dtype=float)
        if kind == "normal":
            if "mean" not in definition:
                raise ValueError(f"Normal parameter '{field}' must have 'mean'")
            mean = _parse_numeric(definition["mean"])
            std = _parse_numeric(definition.get("std", 1.0))
            if pts == 1:
                return np.array([mean], dtype=float)
            percentiles = np.linspace(0.0, 1.0, pts + 2)[1:-1]
            return mean + std * norm.ppf(percentiles)
        if kind == "vector":
            raw_vals = definition.get("values")
            if not isinstance(raw_vals, list):
                raise ValueError(f"Vector parameter '{field}' values must be a list")
            return np.array([_parse_numeric(v) for v in raw_vals], dtype=float)
        raise ValueError(f"Unknown parameter type '{kind}' for '{field}'")

    for field_name, definition in params_cfg.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Parameter '{field_name}' must be a mapping")

        param_type = definition.get("type", "scalar")
        points = int(definition.get("points", 1))
        if points < 1:
            raise ValueError(f"'points' must be >= 1 for '{field_name}'")

        # Derive canonical parameter name from field name and aliases
        base_guess = field_name[:-6] if field_name.endswith("_field") else field_name
        base_name = registry.resolve_alias(base_guess)
        schema = PARAMETER_SCHEMA.get(base_name, {})

        allowed_extra = {"max_simulation_time"}
        if not schema and base_name not in allowed_extra:
            raise ValueError(f"Unknown parameter '{base_guess}' in YAML")

        yaml_unit = definition.get("unit")
        source_unit = yaml_unit or schema.get("unit") or "dimensionless"

        values = _values_from_definition(field_name, param_type, points, definition)

        if schema:
            try:
                values_default, final_unit = registry.convert_to_default_unit(base_name, values, source_unit)
            except Exception as e:
                raise ValueError(
                    f"Unit conversion failed for '{field_name}': {source_unit!r} -> {schema.get('unit', source_unit)!r}: {e}"
                )
        else:
            # Extra field (e.g., max_simulation_time) - keep as provided
            values_default, final_unit = np.asarray(values, dtype=float), source_unit

        metadata = {
            "name": base_name,
            "field": field_name,
            "type": param_type,
            "description": definition.get("description") or schema.get("description", ""),
            "symbol": definition.get("symbol") or schema.get("symbol"),
            "role": schema.get("role"),
            "analysis_types": schema.get("analysis_types"),
        }

        result[base_name] = (values_default, final_unit, metadata)

    if normalized_analysis_type:
        provided_schema_names = [k for k in result.keys() if k in PARAMETER_SCHEMA]
        missing = registry.missing_required(provided_schema_names, normalized_analysis_type)
        if missing:
            raise ValueError(
                f"Missing required parameters for {normalized_analysis_type} analysis: {', '.join(missing)}"
            )
        # Fill computed-when-null entries if not provided
        for name in registry.get_input_names(normalized_analysis_type):
            if name in result or not registry.is_computed_when_null(name):
                continue
            schema = PARAMETER_SCHEMA.get(name, {})
            default_unit = schema.get("unit", "dimensionless")
            result[name] = (
                np.array([np.nan], dtype=float),
                default_unit,
                {
                    "name": name,
                    "field": f"{name}_field",
                    "type": "computed",
                    "description": schema.get("description", ""),
                    "symbol": schema.get("symbol"),
                    "role": schema.get("role"),
                    "analysis_types": schema.get("analysis_types"),
                },
            )

    return result

def prepare_input_data(
    param_fields: Dict[str, Any],
    analysis_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Build ordered input arrays for the requested analysis type."""
    normalized_analysis_type = _normalize_analysis_type(analysis_type)
    if _is_multispecies_analysis(normalized_analysis_type):
        return _prepare_multispecies_input_data(
            param_fields,
            analysis_type="multispecies",
            config=config,
        )

    registry = get_registry()
    if normalized_analysis_type not in {"lump", "T_seeded"}:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    input_names = registry.get_input_names(normalized_analysis_type)
    input_data: Dict[str, Optional[np.ndarray]] = {}

    for name in input_names:
        data = param_fields.get(name)
        if data is None:
            if registry.is_computed_when_null(name):
                input_data[name] = None
                continue
            raise ValueError(f"Missing parameter '{name}' for {normalized_analysis_type}")

        values = np.asarray(data[0], dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)

        if registry.is_computed_when_null(name) and np.all(np.isnan(values)):
            input_data[name] = None
        else:
            input_data[name] = values

    return input_data


def print_configuration(
    config: Dict[str, Any],
    param_fields: Dict[str, Any],
    input_data: Dict[str, np.ndarray],
    param_file: Path,
    config_file: Path,
) -> None:
    """Compact console overview for dry runs."""
    normalized_analysis_type = _normalize_analysis_type(config.get("analysis_type"))
    if _is_multispecies_analysis(normalized_analysis_type):
        config_for_print = dict(config)
        config_for_print["analysis_type"] = "multispecies"
        _print_multispecies_configuration(
            config_for_print,
            param_fields,
            input_data,
            param_file,
            config_file,
        )
        return

    registry = get_registry()
    years = config['max_simulation_time'] / 365 / 24 / 3600

    print("\n" + "=" * 60)
    print("DD STARTUP ANALYSIS CONFIGURATION")
    print("=" * 60)
    print(f"Parameter file: {param_file}")
    print(f"Config file:    {config_file}")
    print(f"Analysis type:  {config['analysis_type']}")
    print(f"Method:         {config['method']}")
    print(f"Vector length:  {config['vector_length']}")
    print(f"Max sim time:   {years:.2f} years")
    print(f"n_jobs:         {config['n_jobs'] or 'auto'}")
    print(f"chunk_size:     {config['chunk_size'] or 'auto'}")
    print(f"batch_size:     {config['batch_size']}")
    if config.get("filter"):
        print(f"Filter:         {config['filter']}")

    param_shapes = [arr.shape[0] for arr in input_data.values() if arr is not None]
    n_combinations = int(np.prod(param_shapes)) if param_shapes else 0

    print("\nInput parameters:")
    for name in registry.get_input_names(config['analysis_type']):
        arr = input_data.get(name)
        label = registry.get_param_label(name, use_symbol=False)
        if arr is None:
            status = "computed during run"
        else:
            status = f"{arr.shape[0]} values, min={np.nanmin(arr):.3g}, max={np.nanmax(arr):.3g}"
        print(f"  {label:20s}: {status}")

    print(f"\nTotal parameter combinations: {n_combinations:,}")
    print("=" * 60 + "\n")


def generate_output_path(
    base_dir: str = "outputs",
    analysis_method: str = "parametric",
    analysis_type: str = "T_seeded",
    timestamp: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[Path, str]:
    """Create output folder and filename."""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir = Path(base_dir) / f"{timestamp}_{analysis_method}_{analysis_type}"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"ddstartup_{timestamp}_{analysis_method}_{analysis_type}.h5"
    return output_dir, str(output_dir / filename)


def latest_output_folder(outputs_dir: Path) -> Tuple[Path | None, List[Path]]:
    """Return (latest_timestamped_folder, sorted_h5_files) or (None, [])."""
    if not outputs_dir.exists():
        return None, []
    dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not dirs:
        return None, []

    def _key(p: Path):
        m = re.match(r"(\d{8})_(\d{6})", p.name)
        if m:
            return (1, m.group(1) + m.group(2))
        st = p.stat()
        return (0, getattr(st, "st_birthtime", st.st_mtime))

    dirs.sort(key=_key, reverse=True)
    latest = dirs[0]
    return latest, sorted(latest.glob("*.h5"))


def latest_h5(outputs_dir: Path) -> Path | None:
    """Return most recently modified .h5 in outputs_dir, or None."""
    h5s = sorted(outputs_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    return h5s[0] if h5s else None


def resolve_h5_inputs(spec: PathLike | List[PathLike], root: Path) -> Tuple[List[Path], Path | None]:
    """
    Resolve 'files' spec into a deduped, ordered list of .h5 paths.
    Returns (files, latest_folder_if_used_else_None).
    Accepted forms:
      - "latest": pick newest timestamped folder; else newest .h5 in <root>/outputs
      - path(s) to .h5 or directories (absolute, CWD-relative, <root>-relative, or <root>/outputs-relative)
    Raises FileNotFoundError / ValueError with clear messages.
    """
    outputs = root / "outputs"

    if isinstance(spec, str) and spec == "latest":
        if not outputs.exists():
            raise FileNotFoundError(f"Outputs folder missing: {outputs}")
        folder, files = latest_output_folder(outputs)
        if folder and files:
            return files, folder
        f = latest_h5(outputs)
        if not f:
            raise FileNotFoundError(f"No .h5 found in {outputs}")
        return [f], outputs

    specs = [spec] if isinstance(spec, (str, Path)) else list(spec)
    files: List[Path] = []
    latest_folder: Path | None = None

    for s in specs:
        s = Path(s)
        if not s.exists():
            for base in (root, outputs):
                cand = base / s
                if cand.exists():
                    s = cand
                    break
        if not s.exists():
            raise FileNotFoundError(f"Path not found: {s}")

        if s.is_dir():
            h5s = sorted(s.glob("*.h5"))
            if not h5s:
                raise FileNotFoundError(f"No .h5 in directory: {s}")
            files.extend(h5s)
            if latest_folder is None:
                latest_folder = s
        else:
            if s.suffix.lower() != ".h5":
                raise ValueError(f"Not an .h5 file: {s}")
            files.append(s)

    seen, uniq = set(), []
    for p in files:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq, latest_folder


def _row_nbytes(ds: h5py.Dataset) -> int:
    """Bytes occupied by a single row of a dataset (all trailing dims)."""
    mult = 1
    if ds.ndim > 1:
        mult = int(np.prod(ds.shape[1:], dtype=int))
    return int(ds.dtype.itemsize * mult)


def _choose_chunk_rows(f: h5py.File, read_names: List[str], user_chunk_size: int | None = None, target_mb: int = 128) -> int:
    """Heuristic for chunk rows: honor user value, else aim for ~target_mb and align with HDF5 chunking."""
    if user_chunk_size is not None and user_chunk_size > 0:
        return int(user_chunk_size)

    total_row_bytes = 0
    chunk_axis_candidates = []
    for name in read_names:
        ds = f[name]
        total_row_bytes += _row_nbytes(ds)
        if ds.chunks and len(ds.chunks) >= 1 and ds.chunks[0]:
            chunk_axis_candidates.append(int(ds.chunks[0]))

    if total_row_bytes <= 0:
        return user_chunk_size or 1

    target_bytes = int(target_mb * 1024 * 1024)
    est_rows = max(1, target_bytes // total_row_bytes)

    # Align to smallest chunk size on axis 0 if available
    if chunk_axis_candidates:
        align = min(chunk_axis_candidates)
        if est_rows < align:
            est_rows = align
        else:
            est_rows = max(align, (est_rows // align) * align)

    return int(est_rows)


def stream_h5_to_df(
    h5_path: Path,
    *,
    columns: List[str] | None = None,   # which datasets to read (default: all present)
    chunk_size: int | None = None,
    downcast_float32: bool = False,
    vectors_to_scalar: bool = False,    # If True, extract only last value from vector columns (saves memory)
    verbose: bool = True,
):
    """
    Generator that streams an HDF5 file into DataFrame chunks.
    
    If vectors_to_scalar=True, vector columns are reduced to their last value (scalar).
    This drastically reduces memory usage for large datasets.
    """
    def _to2d(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return a[:, None]
        if a.ndim == 2:
            return a
        return a.reshape(a.shape[0], int(np.prod(a.shape[1:], dtype=int)))

    def _append_col(builder: dict, name: str, a2d: np.ndarray, downcast_f32: bool, vec_to_scalar: bool) -> int:
        # If 1D vector, preserve as object column (each row is a 1D array)
        if a2d.ndim == 1:
            builder[name] = [np.array([v]) if not isinstance(v, (np.ndarray, list)) else np.array(v) for v in a2d]
            return len(a2d[0]) if hasattr(a2d[0], '__len__') else 1
        # If 2D and shape[1] == 1, treat as scalar
        if a2d.ndim == 2 and a2d.shape[1] == 1:
            col = a2d[:, 0]
            if downcast_f32 and np.issubdtype(col.dtype, np.floating):
                col = col.astype(np.float32, copy=False)
            builder[name] = col
            return 1
        # If 2D and shape[1] > 1, treat as vector
        if a2d.ndim == 2 and a2d.shape[1] > 1:
            if vec_to_scalar:
                # Memory optimization: only keep last value
                col = a2d[:, -1]
                if downcast_f32 and np.issubdtype(col.dtype, np.floating):
                    col = col.astype(np.float32, copy=False)
                builder[name] = col
                return 1  # Treated as scalar
            else:
                if downcast_f32 and np.issubdtype(a2d.dtype, np.floating):
                    a2d = a2d.astype(np.float32, copy=False)
                builder[name] = [a2d[i].copy() for i in range(a2d.shape[0])]
                return int(a2d.shape[1])

    with h5py.File(h5_path, "r") as f:
        # Which datasets to read
        if columns is None:
            read_names = [k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].ndim >= 1]
        else:
            read_names = [k for k in columns if k in f]  # intersect with actual file contents

        # Determine row count
        n = f[read_names[0]].shape[0] if read_names else 0
        chunk_rows = _choose_chunk_rows(f, read_names, user_chunk_size=chunk_size)
        if verbose:
            scal = sum(1 for k in read_names if f[k].ndim == 1)
            vec  = sum(1 for k in read_names if f[k].ndim > 1)
            print(f"   Loading data (core), chunk_size={chunk_rows} ...")
            print(f"   Loading chunks of {chunk_rows:,} rows; will read {scal} scalar and {vec} vector datasets.")
            if vectors_to_scalar and vec > 0:
                print(f"   ⚡ Memory optimization: extracting last value from {vec} vector columns")

        for start in tqdm(range(0, n, chunk_rows), desc="   Loading chunks", unit="chunk"):
            end = min(start + chunk_rows, n)
            coldict: dict[str, Any] = {}
            inner_dims: dict[str, int] = {}

            for name in read_names:
                ds = f[name]
                dest_shape = (end - start, *ds.shape[1:])
                buf = np.empty(dest_shape, dtype=ds.dtype)
                # Copy avoidance: read directly into buffer
                ds.read_direct(buf, source_sel=np.s_[start:end], dest_sel=np.s_[: end - start])
                a2d = _to2d(np.asarray(buf))
                inn = _append_col(coldict, name, a2d, downcast_float32, vectors_to_scalar)
                inner_dims.setdefault(name, inn)

            df_chunk = pd.DataFrame(coldict)
            df_chunk.attrs["_inner_dims"] = inner_dims
            yield df_chunk


def h5_to_df_core(
    h5_path: Path,
    *,
    columns: List[str] | None = None,   # which datasets to read (default: all present)
    chunk_size: int | None = 500_000,
    downcast_float32: bool = False,
    vectors_to_scalar: bool = False,    # If True, extract only last value from vector columns
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stream an HDF5 file to a DataFrame.
    
    If vectors_to_scalar=True, vector columns are reduced to their last value (scalar).
    This drastically reduces memory usage for large datasets.
    """
    parts: list[pd.DataFrame] = []
    inner_dims: dict[str, int] = {}

    for df_chunk in stream_h5_to_df(
        h5_path,
        columns=columns,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        vectors_to_scalar=vectors_to_scalar,
        verbose=verbose,
    ):
        inner_dims.update(df_chunk.attrs.get("_inner_dims", {}))
        parts.append(df_chunk)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.attrs["_inner_dims"] = inner_dims
    return df


# ============================================================================
# MULTISPECIES HELPERS (merged from multispecies_io_functions.py)
# ============================================================================

_MS_SUPPORTED_ANALYSIS_TYPE = "multispecies"
_MS_DD_STARTUP_METHODS = {"none", "T-seeded_old", "lump_old", "T-seeded", "lump"}


def _ms_auto_parallel_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    cpu = os.cpu_count() or 2
    n_jobs = config.get("n_jobs")
    if n_jobs is None:
        n_jobs = max(1, cpu - 1)

    chunk_size = config.get("chunk_size")
    if chunk_size is None:
        chunk_size = max(64, 4 * n_jobs)

    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = max(128, 8 * n_jobs)

    config["n_jobs"] = int(n_jobs)
    config["chunk_size"] = int(chunk_size)
    config["batch_size"] = int(batch_size)
    return config


def _ms_parse_float(value: Any) -> float:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"nan", ".nan", "none", "null"}:
            return float("nan")
        if v in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if v in {"-inf", "-infinity"}:
            return float("-inf")
        return float(value)
    raise ValueError(f"Cannot parse float from {value!r}")


def _ms_parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot parse bool from {value!r}")


def _ms_parse_str(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        return value
    return str(value)


def _ms_normalize_dd_startup_method(value: Any) -> str:
    if value is None:
        return "none"
    raw = _ms_parse_str(value).strip()
    if raw == "":
        return "none"

    token = raw.lower().replace("_", "").replace("-", "")
    alias_map = {
        "none": "none",
        "tseededold": "T-seeded_old",
        "lumpold": "lump_old",
        "tseeded": "T-seeded",
        "lump": "lump",
    }
    if token not in alias_map:
        allowed = ", ".join(sorted(_MS_DD_STARTUP_METHODS))
        raise ValueError(f"Invalid dd_startup_method={value!r}. Allowed values: {allowed}")
    return alias_map[token]


def _ms_parse_targets(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("Config field 'targets' must be a list of dictionaries or null")

    parsed: List[Dict[str, Any]] = []
    allowed_species = set(SPECIES)
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"targets[{i}] must be a dictionary")
        sp = item.get("target_specie", None)
        if sp is None:
            raise ValueError(f"targets[{i}] must define 'target_specie'")
        sp = _ms_parse_str(sp)
        if sp not in allowed_species:
            raise ValueError(f"targets[{i}]['target_specie'] must be one of {sorted(allowed_species)}")

        target_entry: Dict[str, Any] = {"target_specie": sp}
        frac = item.get("target_fraction_in_plasma", None)
        if frac is not None:
            target_entry["target_fraction_in_plasma"] = float(_ms_parse_float(frac))
        inv_ifc = item.get("target_inventory_ifc", None)
        if inv_ifc is not None:
            target_entry["target_inventory_ifc"] = float(_ms_parse_float(inv_ifc))
        inv_ofc = item.get("target_inventory_ofc", None)
        if inv_ofc is not None:
            target_entry["target_inventory_ofc"] = float(_ms_parse_float(inv_ofc))
        inv_st = item.get("target_inventory_storage", None)
        if inv_st is not None:
            target_entry["target_inventory_storage"] = float(_ms_parse_float(inv_st))

        has_metric = any(
            k in target_entry
            for k in (
                "target_fraction_in_plasma",
                "target_inventory_ifc",
                "target_inventory_ofc",
                "target_inventory_storage",
            )
        )
        if has_metric:
            parsed.append(target_entry)

    return parsed


def _load_multispecies_config(yaml_path: Path) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config)!r}")

    for field in ("analysis_type", "method"):
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    analysis_type = _normalize_analysis_type(config["analysis_type"])
    config["analysis_type"] = analysis_type
    if analysis_type != _MS_SUPPORTED_ANALYSIS_TYPE:
        raise ValueError(f"multispecies model supports only analysis_type='{_MS_SUPPORTED_ANALYSIS_TYPE}'")
    if config["method"] != "parametric":
        raise ValueError("multispecies model supports only method='parametric'")

    defaults = {
        "vector_length": 200,
        "max_simulation_time": 10 * 365 * 24 * 3600,
        "targets": [{"target_specie": "T", "target_fraction_in_plasma": 0.5}],
        "enforce_constant_total_density": True,
        "allow_negative_auto_injection": False,
        "auto_injection_use_storage_limits": False,
        "route_the3_ch3_to_he4": False,
        "dd_startup_method": "none",
        "verbose": False,
        "output_dir": "outputs",
        "filter": None,
    }
    for key, default in defaults.items():
        config.setdefault(key, default)
    config.setdefault("n_jobs", None)
    config.setdefault("chunk_size", None)
    config.setdefault("batch_size", None)

    config["vector_length"] = int(round(_ms_parse_float(config["vector_length"])))
    config["max_simulation_time"] = float(_ms_parse_float(config["max_simulation_time"]))
    config["dd_startup_method"] = _ms_normalize_dd_startup_method(config.get("dd_startup_method"))
    config["targets"] = _ms_parse_targets(config["targets"])

    dd_method = config["dd_startup_method"]
    if dd_method in {"T-seeded_old", "T-seeded"}:
        config["targets"] = [{"target_specie": "T", "target_fraction_in_plasma": 0.5}]
    elif dd_method in {"lump_old", "lump"}:
        storage_targets = [t for t in config["targets"] if "target_inventory_storage" in t]
        if not storage_targets:
            raise ValueError(
                "dd_startup_method 'lump'/'lump_old' requires at least one target with "
                "'target_inventory_storage' in config.targets"
            )
        config["targets"] = storage_targets

    for key in (
        "enforce_constant_total_density",
        "allow_negative_auto_injection",
        "auto_injection_use_storage_limits",
        "route_the3_ch3_to_he4",
    ):
        config[key] = _ms_parse_bool(config[key])

    return _ms_auto_parallel_defaults(config)


def _ms_values_from_definition(name: str, definition: Dict[str, Any], dtype: str) -> np.ndarray:
    kind = definition.get("type", "scalar")
    points = int(definition.get("points", 1))
    if points < 1:
        raise ValueError(f"'points' must be >= 1 for '{name}'")

    parser = {
        "float": _ms_parse_float,
        "int": lambda x: int(round(_ms_parse_float(x))),
        "bool": _ms_parse_bool,
        "str": _ms_parse_str,
    }[dtype]

    if kind == "scalar":
        if "value" not in definition:
            raise ValueError(f"Scalar parameter '{name}' must provide 'value'")
        v = parser(definition["value"])
        return np.array([v] * points, dtype=object if dtype == "str" else None)

    if kind == "vector":
        vals = definition.get("values")
        if not isinstance(vals, list):
            raise ValueError(f"Vector parameter '{name}' must provide list 'values'")
        parsed = [parser(v) for v in vals]
        return np.array(parsed, dtype=object if dtype == "str" else None)

    if dtype in {"bool", "str"}:
        raise ValueError(f"Parameter '{name}' with dtype={dtype} supports only scalar/vector")

    if kind == "linear":
        if "min" not in definition or "max" not in definition:
            raise ValueError(f"Linear parameter '{name}' must have min/max")
        vmin = _ms_parse_float(definition["min"])
        vmax = _ms_parse_float(definition["max"])
        if points == 1:
            return np.array([(vmin + vmax) / 2.0], dtype=float)
        return np.linspace(vmin, vmax, points, dtype=float)

    if kind == "normal":
        if "mean" not in definition:
            raise ValueError(f"Normal parameter '{name}' must have mean")
        mean = _ms_parse_float(definition["mean"])
        std = _ms_parse_float(definition.get("std", 1.0))
        if points == 1:
            return np.array([mean], dtype=float)
        percentiles = np.linspace(0.0, 1.0, points + 2)[1:-1]
        return mean + std * norm.ppf(percentiles)

    raise ValueError(f"Unknown parameter type '{kind}' for '{name}'")


def _ms_coerce_param_definition(raw_value: Any, field_name: str) -> Dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    return {"type": "scalar", "value": raw_value, "description": f"Auto-coerced scalar from '{field_name}'"}


def _ms_canonical_species_name(raw_species: Any) -> str:
    if not isinstance(raw_species, str):
        raise ValueError(f"Species key must be a string, got {type(raw_species)!r}")
    normalized = raw_species.strip().replace("_", "").replace("-", "").lower()
    aliases = {"d": "D", "t": "T", "he3": "He3", "he4": "He4"}
    if normalized not in aliases:
        raise ValueError(f"Unknown species '{raw_species}'. Expected one of: {', '.join(SPECIES)}")
    return aliases[normalized]


def _ms_detect_suffix_species(base_name: str) -> Optional[str]:
    for sp in SPECIES:
        if base_name.endswith(f"_{sp}"):
            return sp
    return None


def _ms_is_species_fraction_key(raw_param_name: str) -> bool:
    name = raw_param_name.strip().replace("-", "_").lower()
    return name in {"f0", "f_0", "f_init", "initial_fraction", "fraction_0"}


def _ms_flatten_parameter_definitions(params_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    entries: List[Tuple[str, Dict[str, Any]]] = []

    for field_name, definition in params_cfg.items():
        if field_name == "species_params":
            if not isinstance(definition, dict):
                raise ValueError("'species_params' must be a mapping from species to parameter mappings")
            for raw_sp, sp_params in definition.items():
                sp = _ms_canonical_species_name(raw_sp)
                if not isinstance(sp_params, dict):
                    raise ValueError(f"'species_params.{raw_sp}' must be a parameter mapping")
                for raw_param_name, raw_param_def in sp_params.items():
                    param_def = _ms_coerce_param_definition(raw_param_def, f"species_params.{raw_sp}.{raw_param_name}")
                    if _ms_is_species_fraction_key(raw_param_name):
                        entries.append((f"f_{sp}_0_field", param_def))
                        continue
                    base_name = raw_param_name[:-6] if raw_param_name.endswith("_field") else raw_param_name
                    suffix_species = _ms_detect_suffix_species(base_name)
                    if suffix_species is None:
                        canonical_base = f"{base_name}_{sp}"
                    else:
                        if suffix_species != sp:
                            raise ValueError(
                                f"Species mismatch for '{raw_param_name}' inside species_params.{raw_sp}: "
                                f"suffix species is {suffix_species}"
                            )
                        canonical_base = base_name
                    entries.append((f"{canonical_base}_field", param_def))
            continue

        if field_name in {"initial_fractions", "fractions"}:
            if not isinstance(definition, dict):
                raise ValueError(f"'{field_name}' must be a mapping from species to fraction definitions")
            for raw_sp, raw_fraction_def in definition.items():
                sp = _ms_canonical_species_name(raw_sp)
                frac_def = _ms_coerce_param_definition(raw_fraction_def, f"{field_name}.{raw_sp}")
                entries.append((f"f_{sp}_0_field", frac_def))
            continue

        entries.append((field_name, _ms_coerce_param_definition(definition, field_name)))

    return entries


def _load_multispecies_params(yaml_path: Path, analysis_type: Optional[str] = None) -> Dict[str, Any]:
    registry = get_multispecies_registry()

    if not yaml_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "parameters" not in cfg or not isinstance(cfg["parameters"], dict):
        raise ValueError("YAML file must contain top-level 'parameters' mapping")

    params_cfg = cfg["parameters"]
    result: Dict[str, Any] = {}

    for field_name, definition in _ms_flatten_parameter_definitions(params_cfg):
        base = field_name[:-6] if field_name.endswith("_field") else field_name
        name = registry.resolve_alias(base)

        if name not in MULTISPECIES_PARAMETER_SCHEMA:
            raise ValueError(f"Unknown parameter '{base}' in YAML")
        if name in result:
            prev_field = result[name][2].get("field", name)
            raise ValueError(f"Parameter '{name}' is defined multiple times ('{prev_field}' and '{field_name}')")

        dtype = registry.get_dtype(name)
        source_unit = definition.get("unit")
        if source_unit is None:
            source_unit = registry.get_unit(name)

        values = _ms_values_from_definition(name, definition, dtype)

        if dtype == "float":
            converted, final_unit = registry.convert_to_default_unit(name, np.asarray(values, dtype=float), source_unit)
        elif dtype == "int":
            converted, final_unit = registry.convert_to_default_unit(name, np.asarray(values, dtype=float), source_unit)
            converted = np.rint(converted).astype(int)
        elif dtype == "bool":
            converted, final_unit = np.asarray(values, dtype=bool), source_unit
        else:
            converted, final_unit = np.asarray(values, dtype=object), source_unit

        result[name] = (
            converted,
            final_unit,
            {
                "name": name,
                "field": field_name,
                "type": definition.get("type", "scalar"),
                "description": definition.get("description", ""),
                "dtype": dtype,
            },
        )

    for name in registry.get_input_names(analysis_type):
        if name in result:
            continue
        default = registry.get_default(name)
        dtype = registry.get_dtype(name)
        if default is None and registry.is_required(name):
            raise ValueError(f"Missing required parameter: {name}")
        if dtype == "float":
            arr = np.array([float(default)], dtype=float)
        elif dtype == "int":
            arr = np.array([int(default)], dtype=int)
        elif dtype == "bool":
            arr = np.array([bool(default)], dtype=bool)
        else:
            arr = np.array([str(default)], dtype=object)

        result[name] = (
            arr,
            registry.get_unit(name),
            {
                "name": name,
                "field": f"{name}_field",
                "type": "default",
                "description": MULTISPECIES_PARAMETER_SCHEMA.get(name, {}).get("description", ""),
                "dtype": dtype,
            },
        )

    return result


def _ms_apply_config_controlled_inputs(input_data: Dict[str, np.ndarray], config: Optional[Dict[str, Any]]) -> None:
    if not config:
        return
    input_data["vector_length"] = np.array([int(config["vector_length"])], dtype=int)
    input_data["max_simulation_time"] = np.array([float(config["max_simulation_time"])], dtype=float)
    for key in (
        "enforce_constant_total_density",
        "allow_negative_auto_injection",
        "auto_injection_use_storage_limits",
    ):
        input_data[key] = np.array([bool(config[key])], dtype=bool)


def _prepare_multispecies_input_data(
    param_fields: Dict[str, Any],
    analysis_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    registry = get_multispecies_registry()
    if _normalize_analysis_type(analysis_type) != _MS_SUPPORTED_ANALYSIS_TYPE:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    input_data: Dict[str, np.ndarray] = {}
    for name in registry.get_input_names(analysis_type):
        values = param_fields[name][0]
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        input_data[name] = arr

    _ms_apply_config_controlled_inputs(input_data, config)
    return input_data


def _print_multispecies_configuration(
    config: Dict[str, Any],
    param_fields: Dict[str, Any],
    input_data: Dict[str, np.ndarray],
    param_file: Path,
    config_file: Path,
) -> None:
    registry = get_multispecies_registry()

    print("\n" + "=" * 72)
    print("DDSTARTUP MULTISPECIES CONFIGURATION")
    print("=" * 72)
    print(f"Parameter file: {param_file}")
    print(f"Config file:    {config_file}")
    print(f"Analysis type:  {config['analysis_type']}")
    print(f"Method:         {config['method']}")
    print(f"dd_startup_method: {config.get('dd_startup_method', 'none')}")
    print(f"Vector length:  {config['vector_length']}")
    print(f"Max sim time:   {config['max_simulation_time'] / (365.25 * 24 * 3600):.2f} years")
    print(f"Targets:        {config.get('targets', []) if config.get('targets') else 'none'}")
    print(f"n_jobs:         {config['n_jobs']}")
    print(f"chunk_size:     {config['chunk_size']}")
    print(f"batch_size:     {config['batch_size']}")
    if config.get("filter"):
        print(f"Filter:         {config['filter']}")

    param_shapes = [arr.shape[0] for arr in input_data.values()]
    n_combinations = int(np.prod(param_shapes)) if param_shapes else 0

    print("\nInput fields:")
    for name in registry.get_input_names(config["analysis_type"]):
        arr = input_data[name]
        if arr.dtype == object:
            example = arr[0] if arr.size else ""
            status = f"{arr.shape[0]} values, sample={example}"
        elif arr.dtype == bool:
            status = f"{arr.shape[0]} values, unique={sorted(set(arr.tolist()))}"
        else:
            status = f"{arr.shape[0]} values, min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}"
        print(f"  {name:35s}: {status}")

    print(f"\nTotal parameter combinations: {n_combinations:,}")
    print("=" * 72 + "\n")
