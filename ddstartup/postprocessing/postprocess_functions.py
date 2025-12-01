# ddstartup/postprocessing/postprocess_functions.py
from __future__ import annotations

import sys, re, ast, json, inspect, gc
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
import importlib

import numpy as np
import pandas as pd
import h5py
from contextlib import contextmanager

# Optional compression plugins (skip if missing)
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None

# -----------------------------------------------------------------------------
# Project registry (schema + units)
# -----------------------------------------------------------------------------
from ddstartup.utils.parameter_registry import get_registry, PARAMETER_SCHEMA
from ddstartup.utils.io_functions import resolve_file_path, resolve_h5_inputs, stream_h5_to_df


# -----------------------------------------------------------------------------
# Small config/CLI helper
# -----------------------------------------------------------------------------

def load_config_from_args(args, root: Path) -> Dict[str, Any]:
    """
    Resolve the config path (if provided), load YAML, or fall back to defaults.
    Resolution order handled by resolve_file_path:
      - exact path as given (absolute or relative)
      - <root>/inputs/<name>.yaml or .yml
    """
    default_config_dict = {
        "files": "latest",
        "target_variables": ["unrealized_profits", "t_startup"],
        "plots": {"generate_all": True},
        "output": {"directory": "default", "verbose": True},
        "runtime": {"downcast_float32": False}
    }

    # Check if the config arg is given
    cfg_arg = getattr(args, "config", None)
    if not cfg_arg:
        print("📋 Using default configuration (no config file specified)")
        return default_config_dict

    # Try: as-is; then <root>/inputs/<name>.yaml|.yml (helper also accepts names with extension)
    try:
        cfg_path = resolve_file_path(
            filename=str(cfg_arg),
            default_dir=str(root / "inputs"),
            extensions=[".yaml", ".yml", ""],
        )
    except FileNotFoundError as e:
        print(f"❌ Config not found: {cfg_arg}")
        print(str(e))
        sys.exit(1)

    print(f"📋 Loading configuration from: {Path(cfg_path).name}")
    try:
        import yaml  # type: ignore
    except ImportError:
        print("❌ PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


def apply_cli_overrides(config: Dict[str, Any], args) -> None:
    # Normalize output to a dict early so later code can assume dict
    out_spec = config.get("output", "default")
    if isinstance(out_spec, str):
        config["output"] = {"directory": out_spec}
    elif out_spec is None:
        config["output"] = {"directory": "default"}
        
    if getattr(args, "files", None):
        config["files"] = args.files if len(args.files) > 1 else args.files[0]
    if getattr(args, "targets", None):
        config["target_variables"] = args.targets
    if getattr(args, "output_dir", None):
        config.setdefault("output", {})["directory"] = args.output_dir
    if getattr(args, "plots", None):
        flags = ["kde","parcoords","pdf","importance","kmeans","contour","shap","ml_pairwise","strip"]
        p = config.setdefault("plots", {})
        if "all" in args.plots: p["generate_all"] = True
        else:
            p["generate_all"] = False
            for k in flags: p[k] = (k in args.plots)
    rt = config.setdefault("runtime", {})
    if getattr(args, "chunk_size", None) is not None: rt["chunk_size"] = int(args.chunk_size)
    if getattr(args, "n_jobs", None) is not None: rt["n_jobs"] = int(args.n_jobs)
    if getattr(args, "batch_size", None) is not None: rt["batch_size"] = int(args.batch_size)



def resolve_file_paths(config: Dict[str, Any], root: Path) -> List[Path]:
    spec = config.get("files", "latest")
    try:
        files, latest_folder = resolve_h5_inputs(spec, root)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)

    if isinstance(spec, str) and spec == "latest":
        if latest_folder and len(files) > 1:
            print(f"📁 Found {len(files)} file(s) in: {latest_folder.name}")
        else:
            print(f"📂 Using latest file: {files[0].name} (in outputs/)")
    else:
        # If the spec included directories, show a concise summary
        dirs = sorted({p.parent for p in files})
        if len(dirs) == 1:
            print(f"📁 Found {len(files)} file(s) in: {dirs[0].name}")
        else:
            print(f"📁 Found {len(files)} file(s) across {len(dirs)} folder(s)")

    if latest_folder is not None:
        config["_latest_folder"] = latest_folder
    return files


def _coerce_int(x):
    try:
        if x is None: return None
        if isinstance(x, (bytes, bytearray)): x = x.decode("utf-8","ignore")
        if isinstance(x, str): x = x.strip();  return None if not x else int(float(x))
        return int(x)
    except Exception:
        return None

def _attr_int(g: h5py.Group, k: str):
    return _coerce_int(g.attrs.get(k)) if k in g.attrs else None

def _runtime_from_h5(files: List[Path]) -> Dict[str, int]:
    from collections import Counter
    acc = {"chunk_size": [], "n_jobs": [], "batch_size": []}
    for fp in files:
        with h5py.File(fp, "r") as f:
            for k in acc:
                v = _attr_int(f, k)
                if v is None and "meta" in f: v = _attr_int(f["meta"], k)
                if v is None and "config" in f:
                    try:
                        raw = f["config"][()]
                        txt = raw.decode("utf-8","ignore") if isinstance(raw,(bytes,bytearray)) else str(raw)
                        try: data = json.loads(txt)
                        except Exception:
                            try:
                                import yaml  # noqa
                                data = yaml.safe_load(txt)
                            except Exception:
                                data = {}
                        v = _coerce_int((data or {}).get(k))
                    except Exception:
                        v = None
                if v is not None: acc[k].append(v)
    out = {}
    for k, vals in acc.items():
        if vals: out[k] = Counter(vals).most_common(1)[0][0]
    return out

def apply_h5_runtime_defaults(config: Dict[str, Any], files: List[Path]) -> Dict[str, int]:
    rt = config.setdefault("runtime", {})
    meta = _runtime_from_h5(files)
    for k in ("chunk_size","n_jobs","batch_size"):
        if rt.get(k) in (None, 0) and k in meta: rt[k] = meta[k]
    rt.setdefault("chunk_size", None)  # allow auto/heuristic
    rt.setdefault("n_jobs", 1)
    rt.setdefault("batch_size", 100_000)
    rt.setdefault("downcast_float32", False)
    def _fmt(v):
        return "auto" if v in (None, 0) else v
    print("\n🧰 Runtime: " + ", ".join(f"{k}={_fmt(rt[k])}" for k in ("chunk_size","n_jobs","batch_size","downcast_float32")))
    return rt


# -----------------------------------------------------------------------------
# Filters + additional parsing (compact)
# -----------------------------------------------------------------------------
_ALLOWED_FUNCS = {
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
    "exp": np.exp, "clip": np.clip, "isfinite": np.isfinite, "isnan": np.isnan,
    "round": np.round, "minimum": np.minimum, "maximum": np.maximum,
    "nan_to_num": np.nan_to_num, "pi": np.pi, "e": np.e,
}

def normalize_expr(expr: str) -> str:
    s = expr.strip()
    s = re.sub(r"\band\b", "&", s); s = re.sub(r"\bor\b", "|", s); s = re.sub(r"\bnot\b", "~", s)
    s = re.sub(r"(?<!\*)\^(?!=)", "**", s)
    return s

def _ast_vars(expr: str) -> set[str]:
    class V(ast.NodeVisitor):
        def __init__(self): self.n=set()
        def visit_Name(self,node): self.n.add(node.id)
    t=ast.parse(expr,mode="eval"); v=V(); v.visit(t)
    return {x for x in v.n if x not in {"True","False","None"} and x not in _ALLOWED_FUNCS}


def parse_filters_and_additional(config: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, str]], Set[str]]:
    """
    Parse filters and additional variables from config.
    Returns (filters_exprs, additional_map, additional_meta, passthrough_vars).
      - additional_map: name -> expression (will be evaluated)
      - passthrough_vars: names to load directly from H5 (expr blank or same name)
    """
    # 1) Filters
    filters_exprs = [
        normalize_expr(s)
        for s in (config.get("filters", []) or [])
        if isinstance(s, str) and s.strip()
    ]

    # 2) Additional (support dict or "expr, unit, symbol" string)
    raw = config.get("additional_variables", {}) or {}

    additional_map: Dict[str, str] = {}
    additional_meta: Dict[str, Dict[str, str]] = {}
    passthrough: Set[str] = set()
    for name, spec in raw.items():
        if isinstance(spec, dict):
            expr   = spec.get("expr", "") or ""
            unit   = spec.get("unit")
            symbol = spec.get("symbol")
        else:
            parts  = [p.strip() for p in str(spec).split(",")]
            expr   = parts[0] if parts else ""
            unit   = parts[1] if len(parts) >= 2 and parts[1] else None
            symbol = parts[2] if len(parts) >= 3 and parts[2] else None

        expr = expr.strip()
        if not expr or expr == name:
            passthrough.add(name)
            additional_meta[name] = {"unit": unit or "", "symbol": symbol or name}
        else:
            additional_map[name] = normalize_expr(expr)
            additional_meta[name] = {"unit": unit or "", "symbol": symbol or name}

    # 3) Validate symbols against schema + additional
    schema_names = set(PARAMETER_SCHEMA.keys())
    refs = set()
    if filters_exprs:
        refs |= set().union(*(_ast_vars(e) for e in filters_exprs))
    if additional_map:
        refs |= set().union(*(_ast_vars(e) for e in additional_map.values()))
    allowed_symbols = schema_names | set(additional_map.keys()) | passthrough
    unknown = refs - allowed_symbols
    if unknown:
        print(f"Unknown symbols in expressions: {sorted(unknown)}")
        sys.exit(1)

    # 4) Logging
    if filters_exprs:
        print("\n🔍 Filters:")
        for e in filters_exprs:
            print(f"  {e}")
    if additional_map:
        print("\n🧮 Additional variables (computed):")
        for k, v in additional_map.items():
            u = additional_meta.get(k, {}).get("unit")
            print(f"  {k} = {v}" + (f", Defined Unit: {u}" if u else ""))

    if passthrough:
        print("\n📦 Additional variables (loaded directly):")
        for k in sorted(passthrough):
            u = additional_meta.get(k, {}).get("unit")
            print(f"  {k}" + (f" [{u}]" if u else ""))

    return filters_exprs, additional_map, additional_meta, passthrough


# -----------------------------------------------------------------------------
# Streaming H5 -> filtered DataFrame (vectors preserved)
# -----------------------------------------------------------------------------
def load_h5_to_df(
    h5_path: Path,
    *,
    targets: List[str] | None = None,
    filters_exprs: List[str] | None = None,
    additional_map: Dict[str, str] | None = None,
    passthrough_vars: Set[str] | None = None,
    chunk_size: int | None = None,
    downcast_float32: bool = False,
    verbose: bool = True,
    keep: str = "slim",
    success_only: bool = False,
    plot_types: List[str] | None = None,
    strip_settings: Dict[str, Any] | None = None,
    surface3d_settings: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Read a minimal, plot-ready DataFrame:
      - All schema inputs/flexible parameters
      - Requested targets
      - Variables needed to compute *requested* additional targets
      - Variables referenced by filters
      - 'sol_success' if present
    Optionally keep only those columns ('slim'), default behavior for low memory.
    """
    # Normalize inputs
    additional_map   = additional_map   or {}
    passthrough_vars = set(passthrough_vars or [])
    filters_exprs    = filters_exprs    or []
    targets          = list(targets or [])
    plot_types       = set(plot_types or [])
    strip_settings   = strip_settings or {}
    surface3d_settings = surface3d_settings or {}

    # Discover which H5 columns exist
    with h5py.File(h5_path, "r") as f:
        file_keys = {k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].ndim >= 1}

    # Inputs from schema: those tagged as input/flexible
    def _schema_tag(meta: dict) -> str:
        if not isinstance(meta, dict):
            return ""
        for k in ("role", "kind", "tag", "source", "category"):
            v = meta.get(k)
            if isinstance(v, str):
                return v.lower()
        return ""

    schema_inputs = [
        name for name, meta in PARAMETER_SCHEMA.items()
        if name in file_keys and _schema_tag(meta) in ("input", "flexible")
    ]
    input_params = [n for n in schema_inputs if _schema_tag(PARAMETER_SCHEMA.get(n, {})) == "input"]
    flexible_params = [n for n in schema_inputs if _schema_tag(PARAMETER_SCHEMA.get(n, {})) == "flexible"]
    output_params = [
        name for name, meta in PARAMETER_SCHEMA.items()
        if name in file_keys and _schema_tag(meta) == "output"
    ]

    # Dependency helpers
    def _vars_in(exprs: List[str]) -> set[str]:
        if not exprs:
            return set()
        def _ast_vars(expr: str) -> set[str]:
            class V(ast.NodeVisitor):
                def __init__(self): self.n=set()
                def visit_Name(self, node): self.n.add(node.id)
            t = ast.parse(expr, mode="eval"); v = V(); v.visit(t);
            return {x for x in v.n if x not in {"True","False","None"} and x not in _ALLOWED_FUNCS}
        return set().union(*(_ast_vars(e) for e in exprs))

    filter_deps = _vars_in(filters_exprs)
    all_needed_additional: set[str] = set(additional_map.keys())

    def _gather_additional_chain(name: str, seen: set[str] | None = None) -> set[str]:
        seen = seen or set()
        if name in seen or name not in additional_map:
            return set()
        seen.add(name)
        deps = _vars_in([additional_map[name]])
        acc = {name}
        for d in deps:
            if d in additional_map:
                acc |= _gather_additional_chain(d, seen)
        return acc

    def _gather_base_deps(name: str, seen: set[str] | None = None) -> set[str]:
        seen = seen or set()
        if name in seen or name not in additional_map:
            return set()
        seen.add(name)
        deps = _vars_in([additional_map[name]])
        base: set[str] = set()
        for d in deps:
            base |= _gather_base_deps(d, seen) if d in additional_map else {d}
        return base

    for name in list(all_needed_additional):
        all_needed_additional |= _gather_additional_chain(name)

    additional_base_deps = set()
    for name in all_needed_additional:
        additional_base_deps |= _gather_base_deps(name)

    # Plot-specific columns (e.g., strip metrics, surface3d axes)
    extra_cols: set[str] = set()
    if "strip" in plot_types:
        ys = strip_settings.get("y_metrics", [])
        if isinstance(ys, (str, bytes)):
            ys = [ys]
        extra_cols |= {y for y in ys if y}
        sort_by = strip_settings.get("sort_by")
        if sort_by:
            extra_cols.add(sort_by)
    if "surface3d" in plot_types:
        axes = surface3d_settings.get("axes")
        if isinstance(axes, str):
            axes = [axes]
        extra_cols |= {a for a in (axes or []) if a}

    # Column wish-list (keep vectors intact for future plots)
    wanted = (
        set(schema_inputs)
        | set(targets)
        | additional_base_deps
        | filter_deps
        | set(additional_map.keys())
        | passthrough_vars
        | extra_cols
        | {"sol_success"}
    )
    read_cols = sorted(wanted & file_keys)
    if verbose:
        print(f"   Columns selected to read: {len(read_cols)} "
              f"(inputs={len(schema_inputs)}, targets={len(set(targets))}, "
              f"deps={len((additional_base_deps|filter_deps) & file_keys)})")

    # Helpers for additional + filters (chunk-local)
    def _col_to_2d(s: pd.Series) -> np.ndarray:
        if s.dtype == object:
            arr = np.array([v[-1] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else np.nan for v in s])
            return arr[:, None]
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        return arr[:, None]

    def _env_from_df(df_) -> dict:
        env = {**_ALLOWED_FUNCS}
        for c in df_.columns:
            env[c] = _col_to_2d(df_[c])
        return env

    def _reduce_mask(val: np.ndarray) -> np.ndarray:
        if val.dtype != bool:
            with np.errstate(all="ignore"):
                vv = val.astype(float)
            val = np.isfinite(vv) & (vv != 0)
        return val.any(axis=1) if val.ndim == 2 else val

    def _order_columns(df_: pd.DataFrame) -> list[str]:
        def _add_unique(dst: list[str], names: list[str]):
            for n in names:
                if n in df_.columns and n not in dst:
                    dst.append(n)
        ordered_cols: list[str] = []
        _add_unique(ordered_cols, input_params)
        _add_unique(ordered_cols, flexible_params)
        outputs_needed = [
            n for n in output_params
            if n in df_.columns and (n in additional_base_deps or n in filter_deps or n in targets)
        ]
        _add_unique(ordered_cols, outputs_needed)
        if "sol_success" in df_.columns and "sol_success" not in ordered_cols:
            ordered_cols.append("sol_success")
        additional_order = [name for name in additional_map.keys() if name in df_.columns]
        for name in passthrough_vars:
            if name in df_.columns and name not in additional_order:
                additional_order.append(name)
        _add_unique(ordered_cols, additional_order)
        remaining = [c for c in df_.columns if c not in ordered_cols]
        _add_unique(ordered_cols, remaining)
        return ordered_cols

    # Precompute per-run helpers
    compiled_filters = [
        {
            "expr": expr,
            "code": compile(expr, "<filter>", "eval"),
            "deps": _vars_in([expr]),
            "applied": False,
            "missing": False,
            "nonfinite": False,
        }
        for expr in filters_exprs
    ]
    compiled_additional = {name: (expr, compile(expr, f"<add:{name}>", "eval")) for name, expr in additional_map.items()}
    dep_graph = {name: _vars_in([expr]) for name, expr in additional_map.items()}
    parts: list[pd.DataFrame] = []
    inner_dims_all: dict[str, int] = {}
    cached_order: list[str] | None = None

    for df_chunk in stream_h5_to_df(
        h5_path,
        columns=read_cols,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        verbose=verbose,
    ):
        if df_chunk.empty:
            continue

        inner = dict(df_chunk.attrs.get("_inner_dims", {}))

        # Light downcast on numeric floats
        if downcast_float32:
            for col in df_chunk.columns:
                s = df_chunk[col]
                if pd.api.types.is_float_dtype(s):
                    df_chunk[col] = s.astype(np.float32, copy=False)

        # Computed columns (chunk-local)
        if compiled_additional:
            env = _env_from_df(df_chunk)
            remaining = set(compiled_additional.keys())
            skipped: dict[str, list[str]] = {}
            while remaining:
                progress = False
                for cname in list(remaining):
                    deps = dep_graph.get(cname, set())
                    if all((d in env) for d in deps):
                        expr, code = compiled_additional[cname]
                        out = eval(code, {"__builtins__": {}}, env)
                        a = np.asarray(out)
                        if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1):
                            df_chunk[cname] = a if a.ndim == 1 else a[:, 0]
                            inner[cname] = 1
                        else:
                            df_chunk[cname] = [a[i].copy() for i in range(a.shape[0])]
                            inner[cname] = int(a.shape[1])
                        env[cname] = _col_to_2d(df_chunk[cname])
                        remaining.remove(cname)
                        progress = True
                if not progress:
                    for c in list(remaining):
                        missing = sorted(dep_graph.get(c, set()) - set(env.keys()))
                        skipped[c] = missing
                        remaining.remove(c)
                    if verbose and skipped:
                        print(f"   ⚠️  Skipping additional variables with missing dependencies: {skipped}")
                    break
            if compiled_filters:
                env = _env_from_df(df_chunk)
        else:
            env = _env_from_df(df_chunk) if compiled_filters else {}

        # Filters per chunk
        if compiled_filters:
            mask = np.ones(len(df_chunk), dtype=bool)
            for f in compiled_filters:
                expr, code, deps = f["expr"], f["code"], f["deps"]
                missing = [d for d in deps if d not in env]
                if missing:
                    f["missing"] = True
                    continue
                dep_finite = []
                for d in deps:
                    arr = env.get(d)
                    if arr is None:
                        dep_finite.append(False)
                        continue
                    a = np.asarray(arr)
                    if a.size == 0:
                        dep_finite.append(False)
                        continue
                    if a.dtype == object:
                        flat = []
                        for v in a.ravel():
                            vv = np.asarray(v)
                            if vv.size:
                                flat.append(vv.ravel())
                        a = np.concatenate(flat) if flat else np.array([])
                        if a.size == 0:
                            dep_finite.append(False)
                            continue
                    else:
                        a = a.ravel()
                    a = pd.to_numeric(a, errors="coerce")
                    dep_finite.append(np.isfinite(a).any())
                if not any(dep_finite):
                    f["nonfinite"] = True
                    continue
                val = np.asarray(eval(code, {"__builtins__": {}}, env))
                num = pd.to_numeric(val.ravel(), errors="coerce")
                finite = num[np.isfinite(num)]
                if num.size == 0 or finite.size == 0:
                    f["nonfinite"] = True
                    continue
                mask &= _reduce_mask(val)
                f["applied"] = True
            if not mask.all():
                df_chunk = df_chunk.loc[mask].reset_index(drop=True)
            if df_chunk.empty:
                continue

        if success_only and "sol_success" in df_chunk.columns:
            df_chunk = df_chunk.loc[df_chunk["sol_success"].astype(bool)].reset_index(drop=True)
            if df_chunk.empty:
                continue

        # Keep only needed columns (chunk-local)
        if keep == "slim":
            if cached_order is None:
                cached_order = _order_columns(df_chunk)
            ordered_cols = [c for c in cached_order if c in df_chunk.columns]
            df_chunk = df_chunk.loc[:, ordered_cols]
            inner = {k: v for k, v in inner.items() if k in df_chunk.columns}

        df_chunk.attrs["_inner_dims"] = inner
        for k, v in inner.items():
            inner_dims_all.setdefault(k, v)
        parts.append(df_chunk)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.attrs["_inner_dims"] = inner_dims_all
    if verbose and not df.empty:
        print_columns_overview(df, title="Columns kept")
        for f in compiled_filters:
            if f["applied"]:
                continue
            if f["missing"]:
                print(f"   ⚠️  Filter '{f['expr']}' skipped (missing dependency columns)")
            elif f["nonfinite"]:
                print(f"   ⚠️  Filter '{f['expr']}' skipped (all dependency values non-finite)")
    return df

# -----------------------------------------------------------------------------
# Compact console overview
# -----------------------------------------------------------------------------
def _fmt_scalar(x) -> str:
    if isinstance(x, (bytes, bytearray)): return repr(x)
    if isinstance(x, (np.floating, float)):
        if not np.isfinite(x): return "nan"
        a = abs(x)
        if a >= 1e6 or (a != 0 and a < 1e-3): return f"{x:.3e}"
        if a >= 1e4: return f"{x:.0f}"
        return f"{x:.4g}"
    if isinstance(x, (np.integer, int)): return str(int(x))
    return str(x)

def print_columns_overview(df: pd.DataFrame, *, title: str) -> None:
    rows = len(df)
    inner = (df.attrs or {}).get("_inner_dims", {})

    def _dtype_label(col: pd.Series, inn: int) -> str:
        if inn == 1 and col.dtype != object:
            return f"scalar<{col.dtype}>"
        sample_dtype = None
        for v in col:
            if v is None:
                continue
            arr = np.asarray(v)
            if arr.size == 0:
                continue
            sample_dtype = str(arr.dtype)
            break
        return f"vector<{sample_dtype or col.dtype}>"

    def _trim(text: str, limit: int = 48) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    table_rows: list[dict[str, str]] = []
    for c in df.columns:
        inn = int(inner.get(c, 1))
        col = df[c]
        dtype = _dtype_label(col, inn)
        missing = rows - int(col.notna().sum()) if rows else 0
        missing_pct = (missing / rows * 100) if rows else 0.0
        missing_str = f"{missing}" if not rows else f"{missing} ({missing_pct:4.1f}%)"

        # Scalar stats
        if inn == 1 and col.dtype != object:
            vals = pd.to_numeric(col, errors="coerce").to_numpy()
            if np.isnan(vals).all():
                stats = "min=nan, avg=nan, max=nan"
            else:
                stats = f"min={_fmt_scalar(np.nanmin(vals))}, avg={_fmt_scalar(np.nanmean(vals))}, max={_fmt_scalar(np.nanmax(vals))}"
            preview = _fmt_scalar(col.iloc[0]) if rows else ""
        else:
            # Vector: use last element for stats; preview as [..., second_last, last]
            last_vals = []
            second_vals = []
            for v in col:
                arr = np.asarray(v)
                if arr.size == 0:
                    last_vals.append(np.nan); second_vals.append(np.nan)
                else:
                    last_vals.append(arr[-1])
                    second_vals.append(arr[-2] if arr.size > 1 else arr[-1])
            larr = pd.to_numeric(last_vals, errors="coerce")
            if np.isnan(larr).all():
                stats = "min=nan, avg=nan, max=nan"
            else:
                stats = f"min={_fmt_scalar(np.nanmin(larr))}, avg={_fmt_scalar(np.nanmean(larr))}, max={_fmt_scalar(np.nanmax(larr))}"
            sec0 = second_vals[0] if rows else np.nan
            last0 = last_vals[0] if rows else np.nan
            preview = f"[..., {_fmt_scalar(sec0)}, {_fmt_scalar(last0)}]"

        table_rows.append(
            {
                "column": c,
                "type": dtype,
                "inner": str(inn),
                "missing": missing_str,
                "stats": stats,
                "preview": _trim(str(preview)),
            }
        )

    headers = {
        "column": "Column",
        "type": "Type",
        "inner": "Inner",
        "missing": "Missing",
        "stats": "Stats (min/avg/max)",
        "preview": "Preview",
    }
    order = ["column", "type", "inner", "missing", "stats", "preview"]

    widths = {k: len(v) for k, v in headers.items()}
    for row in table_rows:
        for k, v in row.items():
            widths[k] = max(widths[k], len(v))

    def _fmt_row(row: dict[str, str], header: bool = False) -> str:
        align_left = {"column", "type", "stats", "preview"}
        parts = []
        for key in order:
            val = headers[key] if header else row[key]
            if key in align_left:
                parts.append(f"{val:<{widths[key]}}")
            else:
                parts.append(f"{val:>{widths[key]}}")
        return "  ".join(parts)

    print()
    print(f"{title} (rows={rows})")
    print(_fmt_row({}, header=True))
    print("  ".join("-" * widths[key] for key in order))
    for row in table_rows:
        print(_fmt_row(row))


# -----------------------------------------------------------------------------
# Registry wrapper (keep for plot label/unit compatibility)
# -----------------------------------------------------------------------------
class AdditionalAwareRegistry:
    def __init__(self, base, additional_meta: dict|None):
        self._base = base; self._meta = additional_meta or {}
    def get_param_label(self, name: str, *a, **k):
        m = self._meta.get(name)
        if m and m.get("symbol") and not k.get("prefer_base", False): return m["symbol"]
        return self._base.get_param_label(name, *a, **k) if hasattr(self._base,"get_param_label") else name
    def get_param_unit(self, name: str, *a, **k):
        m = self._meta.get(name)
        if m and m.get("unit"): return m["unit"]
        return self._base.get_param_unit(name, *a, **k) if hasattr(self._base,"get_param_unit") else None
    def __getattr__(self, attr): return getattr(self._base, attr)


@contextmanager
def plot_style_context(show_titles: bool = True, font_scale: float | None = None):
    """
    Temporarily apply plotting style:
      - Disable titles/supertitles when show_titles=False
      - Scale common font rcParams when font_scale is provided
    Restores original settings afterwards.
    """
    import matplotlib
    import matplotlib.pyplot as plt  # noqa: F401

    saved_rc: dict[str, Any] = {}
    if font_scale is not None:
        try:
            scale = float(font_scale)
        except Exception:
            scale = None
        if scale and scale > 0:
            for key in (
                "font.size",
                "axes.titlesize",
                "axes.labelsize",
                "xtick.labelsize",
                "ytick.labelsize",
                "legend.fontsize",
                "figure.titlesize",
            ):
                saved_rc[key] = matplotlib.rcParams.get(key)
                val = saved_rc[key]
                if isinstance(val, (int, float)):
                    matplotlib.rcParams[key] = val * scale
        # also scale tick padding/label padding so spacing grows with font size
        saved_rc["xtick.major.pad"] = matplotlib.rcParams.get("xtick.major.pad")
        saved_rc["ytick.major.pad"] = matplotlib.rcParams.get("ytick.major.pad")
        saved_rc["axes.labelpad"] = matplotlib.rcParams.get("axes.labelpad")
        if scale and scale > 0:
            try:
                matplotlib.rcParams["xtick.major.pad"] = float(saved_rc["xtick.major.pad"]) * scale
                matplotlib.rcParams["ytick.major.pad"] = float(saved_rc["ytick.major.pad"]) * scale
                matplotlib.rcParams["axes.labelpad"] = float(saved_rc["axes.labelpad"]) * scale
            except Exception:
                pass

    saved_set_title = None
    saved_suptitle = None
    if not show_titles:
        saved_set_title = matplotlib.axes.Axes.set_title
        saved_suptitle = matplotlib.figure.Figure.suptitle

        def _no_title(self, *args, **kwargs):
            return None

        def _no_suptitle(self, *args, **kwargs):
            return None

        matplotlib.axes.Axes.set_title = _no_title  # type: ignore
        matplotlib.figure.Figure.suptitle = _no_suptitle  # type: ignore

    try:
        yield
    finally:
        if saved_set_title:
            matplotlib.axes.Axes.set_title = saved_set_title  # type: ignore
        if saved_suptitle:
            matplotlib.figure.Figure.suptitle = saved_suptitle  # type: ignore
        for key, val in saved_rc.items():
            if val is None:
                continue
            matplotlib.rcParams[key] = val

# colorscale kept for older plot modules
def get_discrete_colorscale(n_chunks: int):
    import matplotlib, matplotlib.colors as mcolors, numpy as _np
    base = ["#2166AC","#4393C3","#92C5DE","#FFFFBF","#FDAE61","#F46D43","#D73027"]
    if n_chunks <= len(base):
        rgb = [mcolors.to_rgb(c) for c in base]
        pos = _np.linspace(0,1,len(rgb)); tgt = _np.linspace(0,1,n_chunks)
        interp = _np.array([_np.interp(tgt,pos,[r[i] for r in rgb]) for i in range(3)]).T
        colors = [mcolors.to_hex(c) for c in interp]
    else:
        cmap = matplotlib.colormaps.get_cmap('RdYlBu_r')
        colors = [mcolors.to_hex(cmap(t)) for t in _np.linspace(0,1,n_chunks)]
    out=[]
    for i,c in enumerate(colors):
        a=i/n_chunks; b=(i+1)/n_chunks
        out.append([a,c]); out.append([b,c])
    return out


# -----------------------------------------------------------------------------
# Plot wiring (generalized dispatch)
# -----------------------------------------------------------------------------
def collect_plot_settings(config: Dict[str, Any], args, targets: List[str], plot_types: List[str]):
    plots = config.get("plots", {})
    shap = plots.get("shap_settings", {})
    pdf  = plots.get("pdf_settings", {})
    shap_interp = shap.get("interpolate", False) or bool(getattr(args, "shap_interpolate", False))
    pdf_smooth  = pdf.get("smooth", False) or bool(getattr(args, "pdf_smooth", False))
    if "shap" in plot_types: print(f"🔷 SHAP interpolation: {'ENABLED (smooth density plots)' if shap_interp else 'DISABLED (scatter plots)'}")
    if "pdf"  in plot_types: print(f"🔷 PDF smoothing: {'ENABLED (KDE)' if pdf_smooth else 'DISABLED (histogram bins)'}")
    ml_pair = plots.get("ml_pairwise_settings", {})
    if "ml_pairwise" in plot_types: print(f"🔷 ML pairwise mode: {ml_pair.get('pairs','auto')}")
    strip = plots.get("strip_settings", {})
    if "strip" in plot_types:
        ys = strip.get("y_metrics", targets[:min(3,len(targets))])
        strip = {**strip, "y_metrics": ys}
        print(f"🔷 Strip plot metrics: {', '.join(ys)}")
    style_cfg = plots.get("style", {}) or {}
    show_titles = bool(style_cfg.get("show_titles", True))
    font_scale = style_cfg.get("font_scale")
    surface3d = plots.get("surface3d_settings", {})
    axes = surface3d.get("axes")
    if "surface3d" in plot_types and axes:
        if isinstance(axes, str):
            axes = [axes]
        print(f"🔷 Surface3D axes: {', '.join(axes[:3])}")
    return shap_interp, pdf_smooth, ml_pair, strip, show_titles, font_scale, surface3d

def generate_plots_for_file(
    path: Path,
    *,
    targets: List[str],
    filters_exprs: List[str],
    additional_map: Dict[str, str],
    passthrough_vars: Set[str],
    plot_types: List[str],
    output_dir: Path,
    additional_meta: Dict[str, Dict[str, str]] | None = None,
    shap_interpolate: bool = False,
    pdf_smooth: bool = False,
    ml_pairwise_settings: Dict[str, Any] | None = None,
    strip_settings: Dict[str, Any] | None = None,
    surface3d_settings: Dict[str, Any] | None = None,
    chunk_size: int | None = None,
    n_jobs: int = 1,
    batch_size: int = 100_000,
    downcast_float32: bool = False,
    show_titles: bool = True,
    font_scale: float | None = None,
) -> None:
    # Registry (respect additional labels/units if provided)
    registry = get_registry()
    if additional_meta:
        registry = AdditionalAwareRegistry(registry, additional_meta)

    print(f"\n📁 Processing: {path.name}")

    # Try to read total_combinations (optional; not used directly here)
    total_combos = None
    try:
        with h5py.File(path, "r") as f:
            total_combos = (
                _attr_int(f, "total_combinations")
                or (_attr_int(f["meta"], "total_combinations") if "meta" in f else None)
            )
    except Exception as _e:
        print(f"   ⚠️ Could not read total_combinations: {_e}")

    # File type label
    name = path.name.lower()
    file_type = "lump" if "lump" in name else ("Tseeded" if ("t_seeded" in name or "tseeded" in name) else "unknown")

    # Stream → DataFrame (vectors preserved)
    df = load_h5_to_df(
        path,
        targets=targets,
        filters_exprs=filters_exprs,
        additional_map=additional_map,
        passthrough_vars=passthrough_vars,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        verbose=True,
        plot_types=plot_types,
        strip_settings=strip_settings,
        surface3d_settings=surface3d_settings,
    )
    if df.empty:
        print("   ⚠️  No data after filtering. Skipping file.")
        return

    # Keep ALL rows (after config filters) to include failures for the "5th quartile"
    df_all = df.copy()

    # Dedupe targets preserving order, keep only those present in this file
    seen = set()
    all_targets = [t for t in targets if not (t in seen or seen.add(t))]
    available_targets = [t for t in all_targets if t in df.columns]
    missing_targets = [t for t in all_targets if t not in df.columns]
    if missing_targets:
        print(f"   ⚠️  Skipping missing targets for this file: {', '.join(missing_targets)}")
    if not available_targets:
        print("   ⚠️  No requested targets available in this file. Skipping.")
        return

    # Dispatcher
    def _call(mod_path: str, fn_name: str, **kwargs):
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, fn_name, None)
            if fn is None:
                print(f"   ⚠️  {fn_name} not found in {mod_path}. Skipping.")
                return
            sig = inspect.signature(fn)
            allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return fn(**allowed)
        except Exception as e:
            print(f"   ❌ Error in {fn_name}: {e}")

    pipeline = [
        ("kde",         "ddstartup.postprocessing.plot_kde_functions",                 "kde_quartile_plot"),
        ("importance",  "ddstartup.postprocessing.plot_importance_matrix",             "plot_effect_size_matrix"),
        ("kmeans",      "ddstartup.postprocessing.plot_kmeans_functions",              "cluster_and_quartile_bar"),
        ("contour",     "ddstartup.postprocessing.plot_contour_functions",             "plot_interactive_pairwise_contours"),
        ("parcoords",   "ddstartup.postprocessing.plot_parcoords_functions",           "generate_parcoords_plot"),
        ("pdf",         "ddstartup.postprocessing.plot_pdf_functions",                 "generate_pdf_plot"),
        ("shap",        "ddstartup.postprocessing.plot_shap_functions",                "generate_shap_plots"),
        ("ml_pairwise", "ddstartup.postprocessing.plot_ML_pairwise_functions",         "generate_ml_pairwise_plots"),
        ("strip",       "ddstartup.postprocessing.plot_strips",                        "generate_strip_plot"),
        ("quartprob",   "ddstartup.postprocessing.plot_quartile_probability_functions","quartile_probability_plot"),
        ("surface3d",   "ddstartup.postprocessing.plot_surface3d",                     "generate_surface3d_plot"),
    ]

    inner_dims = (df.attrs or {}).get("_inner_dims", {})

    with plot_style_context(show_titles=show_titles, font_scale=font_scale):
        for target in available_targets:

            # scalar-only targets
            if int(inner_dims.get(target, 1)) != 1:
                print(f"   ⚠️  Target '{target}' is vector-valued. Skipping scalar plots.")
                continue

            # Success-only frame for quartiles: finite target
            y = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df_t = df.loc[y.notna()]
            if df_t.empty:
                print(f"   ⚠️  No finite data for '{target}'. Skipping.")
                continue

            inputs = [c for c in df_t.columns if c != target]

            # Target unit
            if additional_meta and additional_meta.get(target, {}).get("unit"):
                tunit = additional_meta[target]["unit"]
            elif isinstance(PARAMETER_SCHEMA.get(target), dict):
                tunit = PARAMETER_SCHEMA[target].get("unit", "") or ""
            else:
                tunit = ""

            common = dict(
                df=df_t,
                df_filtered=df_t,
                target=target,
                inputs=inputs,
                target_unit=tunit,
                output_dir=output_dir,
                file_type=file_type,
                registry=registry,
                n_jobs=n_jobs,
                batch_size=batch_size,
                pdf_smooth=pdf_smooth,
                shap_interpolate=shap_interpolate,
                ml_pairwise_settings=ml_pairwise_settings or {},
                strip_settings=strip_settings or {},
                surface3d_settings=surface3d_settings or {},
                plot_name_prefix=path.stem,
                show_titles=show_titles,
            )

            
            for key, mod, fn in pipeline:
                if key in plot_types:
                    if key == "quartprob":
                        # pass the FULL df (includes failures) so the plotter can compute FAILED per-bin
                        _call(mod, fn, df=df,           # << full DF here
                            target=target,
                            inputs=inputs,
                            target_unit=tunit,
                            output_dir=output_dir,
                            file_type=file_type,
                            registry=registry,
                            plot_name_prefix=path.stem,
                            COUNT_FAILED=True, AVG_POINTS=10, PLOT_STYLE="line")
                    else:
                        _call(mod, fn, **common)

            gc.collect()

    print("   🧹 Memory cleaned for next file")
