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
from .parameter_registry import PARAMETER_SCHEMA, get_registry
from .units_and_constants import u  # Pint UnitRegistry


PathLike = Union[str, Path]

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

    if analysis_type:
        provided_schema_names = [k for k in result.keys() if k in PARAMETER_SCHEMA]
        missing = registry.missing_required(provided_schema_names, analysis_type)
        if missing:
            raise ValueError(
                f"Missing required parameters for {analysis_type} analysis: {', '.join(missing)}"
            )
        # Fill computed-when-null entries if not provided
        for name in registry.get_input_names(analysis_type):
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

def prepare_input_data(param_fields: Dict[str, Any], analysis_type: str) -> Dict[str, np.ndarray]:
    """Build ordered input arrays for the requested analysis type."""
    registry = get_registry()
    if analysis_type not in {"lump", "T_seeded"}:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    input_names = registry.get_input_names(analysis_type)
    input_data: Dict[str, Optional[np.ndarray]] = {}

    for name in input_names:
        data = param_fields.get(name)
        if data is None:
            if registry.is_computed_when_null(name):
                input_data[name] = None
                continue
            raise ValueError(f"Missing parameter '{name}' for {analysis_type}")

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
    verbose: bool = True,
):
    """
    Generator that streams an HDF5 file into DataFrame chunks, preserving 1D vectors as per-row ndarrays.
    """
    def _to2d(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return a[:, None]
        if a.ndim == 2:
            return a
        return a.reshape(a.shape[0], int(np.prod(a.shape[1:], dtype=int)))

    def _append_col(builder: dict, name: str, a2d: np.ndarray, downcast_f32: bool) -> int:
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
                inn = _append_col(coldict, name, a2d, downcast_float32)
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
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stream an HDF5 file to a DataFrame, preserving 1D vectors as per-row ndarrays.
    No computed variables, no filters. Pure I/O.
    """
    parts: list[pd.DataFrame] = []
    inner_dims: dict[str, int] = {}

    for df_chunk in stream_h5_to_df(
        h5_path,
        columns=columns,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        verbose=verbose,
    ):
        inner_dims.update(df_chunk.attrs.get("_inner_dims", {}))
        parts.append(df_chunk)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.attrs["_inner_dims"] = inner_dims
    return df
