# ddstartup/postprocessing/postprocess_functions.py
from __future__ import annotations
from itertools import chain

import sys, re, ast, json, inspect, gc
from pathlib import Path
from typing import Any, Dict, List, Tuple
import importlib, inspect, gc

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import hdf5plugin  # noqa: F401


# -----------------------------------------------------------------------------
# Project registry (schema + units)
# -----------------------------------------------------------------------------
from ddstartup.utils.parameter_registry import get_registry, PARAMETER_SCHEMA
from ddstartup.utils.io_functions import resolve_file_path, resolve_h5_inputs, h5_to_df_core


# -----------------------------------------------------------------------------
# Small config/CLI plumbing
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
    rt.setdefault("chunk_size", 500_000)
    rt.setdefault("n_jobs", 1)
    rt.setdefault("batch_size", 100_000)
    rt.setdefault("downcast_float32", False)
    print("\n🧰 Runtime: " + ", ".join(f"{k}={rt[k]}" for k in ("chunk_size","n_jobs","batch_size","downcast_float32")))
    return rt


# -----------------------------------------------------------------------------
# Filters + computed parsing (compact)
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


def parse_filters_and_computed(config: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Parse filters and computed variables from config.
    Returns (filters_exprs, computed_map, computed_meta).
    """
    # 1) Filters
    filters_exprs = [
        normalize_expr(s)
        for s in (config.get("filters", []) or [])
        if isinstance(s, str) and s.strip()
    ]

    # 2) Computed (support dict or "expr, unit, symbol" string)
    raw = config.get("computed_variables", {}) or {}
    computed_map: Dict[str, str] = {}
    computed_meta: Dict[str, Dict[str, str]] = {}
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

        if expr.strip():
            computed_map[name] = normalize_expr(expr)
            computed_meta[name] = {"unit": unit or "", "symbol": symbol or name}

    # 3) Validate symbols against schema + computed
    schema_names = set(PARAMETER_SCHEMA.keys())
    refs = set()
    if filters_exprs:
        refs |= set().union(*(_ast_vars(e) for e in filters_exprs))
    if computed_map:
        refs |= set().union(*(_ast_vars(e) for e in computed_map.values()))
    unknown = refs - (schema_names | set(computed_map.keys()))
    if unknown:
        print(f"Unknown symbols in expressions: {sorted(unknown)}")
        sys.exit(1)

    # 4) Logging
    if filters_exprs:
        print("\n🔍 Filters:")
        for e in filters_exprs:
            print(f"  {e}")
    if computed_map:
        print("\n🧮 Computed variables:")
        for k, v in computed_map.items():
            u = computed_meta.get(k, {}).get("unit")
            print(f"  {k} = {v}" + (f", Defined Unit: {u}" if u else ""))

    return filters_exprs, computed_map, computed_meta


# -----------------------------------------------------------------------------
# Streaming H5 -> filtered DataFrame (vectors preserved)
# -----------------------------------------------------------------------------
def load_h5_to_df(
    h5_path: Path,
    *,
    targets: List[str] | None = None,
    filters_exprs: List[str] | None = None,
    computed_map: Dict[str, str] | None = None,
    chunk_size: int = 500_000,
    downcast_float32: bool = False,
    verbose: bool = True,
    keep: str = "slim",
    success_only: bool = False,         # << NEW
) -> pd.DataFrame:
    """
    Read a minimal, plot-ready DataFrame:
      - All schema inputs/flexible parameters
      - Requested targets
      - Variables needed to compute *requested* computed targets
      - Variables referenced by filters
      - 'sol_success' if present
    Optionally keep only those columns ('slim'), default behavior for low memory.
    """
    computed_map   = computed_map   or {}
    filters_exprs  = filters_exprs  or []
    targets        = list(targets or [])

    # ---- Discover which H5 columns exist
    with h5py.File(h5_path, "r") as f:
        file_keys = {k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].ndim >= 1}

    # ---- Inputs from schema: those tagged as input/flexible
    # be tolerant to different schema field names: role/kind/tag/source/category
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

    # ---- Dependencies
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

    # only deps for computed targets you actually requested
    computed_target_deps = set()
    for t in targets:
        if t in computed_map:
            computed_target_deps |= _vars_in([computed_map[t]])

    filter_deps = _vars_in(filters_exprs)

    # ---- Column wish-list
    wanted = set(schema_inputs) | set(targets) | computed_target_deps | filter_deps | {"sol_success"}
    read_cols = sorted(wanted & file_keys)

    if verbose:
        print(f"   Columns selected to read: {len(read_cols)} "
              f"(inputs={len(schema_inputs)}, targets={len(set(targets))}, "
              f"deps={len((computed_target_deps|filter_deps) & file_keys)})")

    # ---- Core read (zero extra logic; vector columns preserved)
    df = h5_to_df_core(
        h5_path,
        columns=read_cols,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        verbose=verbose,
    )
    
    if df.empty:
        return df

    # ---- Computed columns (evaluate only those requested OR referenced by filters)
    # If you also want computed variables not in targets but referenced by filters, keep them here.
    need_computed = {t for t in targets if t in computed_map}
    # add computed variables directly referenced in filters (rare, but supported)
    for name, expr in (computed_map or {}).items():
        if name in _vars_in(filters_exprs):
            need_computed.add(name)

    if need_computed:
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

        env = _env_from_df(df)
        inner = (df.attrs or {}).setdefault("_inner_dims", {})
        for cname in need_computed:
            expr = computed_map[cname]
            out = eval(expr, {"__builtins__": {}}, env)
            a = np.asarray(out)
            if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1):
                df[cname] = a if a.ndim == 1 else a[:, 0]
                inner[cname] = 1
            else:
                df[cname] = [a[i].copy() for i in range(a.shape[0])]
                inner[cname] = int(a.shape[1])
        # recompute env only if you’ll apply filters below
        if filters_exprs:
            env = _env_from_df(df)

    # ---- Post filters (reduce vector masks with any(axis=1))
    if filters_exprs:
        def _reduce_mask(val: np.ndarray) -> np.ndarray:
            if val.dtype != bool:
                with np.errstate(all="ignore"):
                    vv = val.astype(float)
                val = np.isfinite(vv) & (vv != 0)
            return val.any(axis=1) if val.ndim == 2 else val
        env = locals().get("env") or {**_ALLOWED_FUNCS, **{c: (np.stack(df[c].values) if df[c].dtype==object else pd.to_numeric(df[c], errors="coerce").to_numpy()[:,None]) for c in df.columns}}
        mask = np.ones(len(df), dtype=bool)
        for expr in filters_exprs:
            val = np.asarray(eval(expr, {"__builtins__": {}}, env))
            mask &= _reduce_mask(val)
        if not mask.all():
            df = df.loc[mask].reset_index(drop=True)

    # ---- Keep only the columns you asked for (inputs + targets + deps [+ computed targets]) unless keep=="all"
    if keep == "slim":
        keep_cols = set(schema_inputs) | set(targets) | computed_target_deps | filter_deps | {"sol_success"}
        # also keep computed targets we just created
        keep_cols |= (need_computed if need_computed else set())
        keep_cols = [c for c in df.columns if c in keep_cols]
        df = df[keep_cols].copy()
    if success_only and "sol_success" in df.columns:
        df = df.loc[df["sol_success"].astype(bool)].reset_index(drop=True)
    if verbose:
        print_columns_overview(df, title="Columns kept")
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

def _preview_cell(v) -> str:
    a = np.asarray(v)
    if a.ndim == 0: return _fmt_scalar(a.item())
    if a.ndim == 1:
        m = min(3, a.size)
        head = ", ".join(_fmt_scalar(float(a[i])) for i in range(m))
        return f"[{head}{', …' if a.size>m else ''}]"
    return f"arr{a.shape}"

def print_columns_overview(df: pd.DataFrame, *, title: str) -> None:
    rows = len(df)
    inner = (df.attrs or {}).get("_inner_dims", {})
    print()
    print(f"{title:>18}   {'Rows':<8} {'Inner':<5} {'first 5 values'}")
    for c in df.columns:
        inn = int(inner.get(c, 1))
        head = df[c].iloc[:5].tolist()
        prev = ", ".join(_preview_cell(v) for v in head)
        print(f"{c:>18} {rows:<8d} {inn:<5d} {prev}")


# -----------------------------------------------------------------------------
# Registry wrapper (keep for plot label/unit compatibility)
# -----------------------------------------------------------------------------
class ComputedAwareRegistry:
    def __init__(self, base, computed_meta: dict|None):
        self._base = base; self._meta = computed_meta or {}
    def get_param_label(self, name: str, *a, **k):
        m = self._meta.get(name)
        if m and m.get("symbol") and not k.get("prefer_base", False): return m["symbol"]
        return self._base.get_param_label(name, *a, **k) if hasattr(self._base,"get_param_label") else name
    def get_param_unit(self, name: str, *a, **k):
        m = self._meta.get(name)
        if m and m.get("unit"): return m["unit"]
        return self._base.get_param_unit(name, *a, **k) if hasattr(self._base,"get_param_unit") else None
    def __getattr__(self, attr): return getattr(self._base, attr)

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
    return shap_interp, pdf_smooth, ml_pair, strip

def _read_minimal_all(path: Path, cols: list[str], inner_dims: dict | None = None) -> pd.DataFrame:
    import h5py
    try:
        import hdf5plugin  # if compressed; no-op if missing
    except ImportError:
        pass

    out: dict[str, pd.Series] = {}
    with h5py.File(path, "r") as f:
        for c in cols:
            if c == "sol_success":
                continue  # handle after loop

            # keep only scalar inputs
            if inner_dims is not None and int(inner_dims.get(c, 1)) != 1:
                continue
            if c not in f:
                continue

            d = f[c][...]
            # squeeze (N,1) / (1,N); skip true vectors (N,K>1)
            if isinstance(d, np.ndarray) and d.ndim > 1:
                if d.shape[1:] == (1,):
                    d = d.reshape(-1)
                elif d.shape[0] == 1:
                    d = d.reshape(-1)
                else:
                    # vector-valued input -> skip
                    continue

            if np.issubdtype(np.asarray(d).dtype, np.number):
                s = pd.Series(d, dtype="float64")
            else:
                s = pd.to_numeric(pd.Series(d), errors="coerce")
            out[c] = s

        # sol_success (robust)
        if "sol_success" in f:
            s = f["sol_success"][...]
            if isinstance(s, np.ndarray) and s.ndim > 1:
                if s.shape[1:] == (1,):
                    s = s.reshape(-1)
                elif s.shape[0] == 1:
                    s = s.reshape(-1)
                else:
                    s = s.ravel()

            if np.issubdtype(np.asarray(s).dtype, np.bool_) or np.issubdtype(np.asarray(s).dtype, np.integer):
                sol = pd.Series(s).astype(bool)
            else:
                def _truthy(v):
                    if pd.isna(v): return False
                    if isinstance(v, (bool, np.bool_, int, np.integer)): return bool(v)
                    return str(v).strip().lower() in {"1","true","t","y","yes"}
                sol = pd.Series(list(map(_truthy, s)), dtype=bool)
            out["sol_success"] = sol

    if not out:
        return pd.DataFrame()

    # align lengths (truncate to min length to keep indices consistent)
    nmin = min(len(s) for s in out.values())
    for k in list(out.keys()):
        out[k] = out[k].iloc[:nmin].reset_index(drop=True)

    return pd.DataFrame(out)

def compute_bin_stats_from_h5(
    path: Path,
    params: list[str],
    *,
    inner_dims: dict | None = None,
    avg_points: int = 10,
    registry=None,
    chunk: int = 500_000,
) -> dict[str, dict]:
    """
    Return per-parameter evenly spaced bin edges and counts over ALL runs:
      bin_stats[p] = {"edges": np.ndarray[M+1], "total": np.ndarray[M], "failed": np.ndarray[M]}
    Only scalar inputs (inner_dim==1) are processed. t_startup -> days, c_kWh* (or 1/J units) -> $/kWh.
    """
    import h5py
    try:
        import hdf5plugin  # if present enables compressed reads
    except ImportError:
        pass

    # minimal unit/name-based transform (kept in sync with the plotter)
    def _x_transform(name: str, arr: np.ndarray) -> tuple[np.ndarray, str | None]:
        lname = name.lower()
        unit = getattr(registry, "get_param_unit", lambda n, **k: None)(name) or ""
        u = unit.replace(" ", "").lower()
        if name in {"t_startup", "tStartup"} or "t_startup" in lname:
            return arr / 86400.0, "days"
        if ("kwh" in lname) or ("/j" in u or "1/j" in u or "j^-1" in u):
            return arr * 3.6e6, "$/kWh"
        return arr, None

    # filter scalar params
    scalar_params = [p for p in params if (inner_dims is None or int(inner_dims.get(p, 1)) == 1)]
    out: dict[str, dict] = {}
    if not scalar_params:
        return out

    with h5py.File(path, "r") as f:
        # Determine N and ensure sol_success exists
        if "sol_success" not in f:
            # no failures info; we will still return totals (failed=0)
            N = None
            for p in scalar_params:
                if p in f:
                    N = f[p].shape[0]
                    break
            has_sol = False
        else:
            N = f["sol_success"].shape[0]
            has_sol = True
        if N is None:
            return out

        # --- PASS 1: min/max per parameter (streaming) ---
        mins = {p: np.inf for p in scalar_params}
        maxs = {p: -np.inf for p in scalar_params}

        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            for p in scalar_params:
                if p not in f:
                    continue
                a = f[p][start:end]
                a = np.asarray(a)
                if a.ndim > 1:
                    # squeeze (N,1)/(1,N); skip true vectors
                    if a.shape[1:] == (1,):
                        a = a.reshape(-1)
                    elif a.shape[0] == 1:
                        a = a.reshape(-1)
                    else:
                        continue
                a = a.astype("float64", copy=False)
                a[np.isinf(a)] = np.nan
                a, _ = _x_transform(p, a)
                finite = np.isfinite(a)
                if not finite.any():
                    continue
                mins[p] = min(mins[p], float(np.nanmin(a[finite])))
                maxs[p] = max(maxs[p], float(np.nanmax(a[finite])))

        # prepare edges and alloc counters
        for p in scalar_params:
            vmin, vmax = mins[p], maxs[p]
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                continue
            if vmin == vmax:
                edges = np.array([vmin, vmax], dtype="float64")
            else:
                edges = np.linspace(vmin, vmax, avg_points + 1, dtype="float64")
            out[p] = {
                "edges": edges,
                "total": np.zeros(edges.size - 1, dtype=np.int64),
                "failed": np.zeros(edges.size - 1, dtype=np.int64),
            }

        if not out:
            return out

        # helper: safe truth conversion
        def _to_bool(arr) -> np.ndarray:
            a = np.asarray(arr)
            if a.dtype.kind in "biu":
                return a.astype(bool, copy=False)
            if a.dtype.kind in "f":
                return (a != 0).astype(bool)
            # string-like
            v = np.vectorize(lambda x: str(x).strip().lower() in {"1", "true", "t", "y", "yes"})
            return v(a)

        # --- PASS 2: accumulate totals/failed per bin ---
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            if has_sol:
                sol = _to_bool(f["sol_success"][start:end])
            else:
                sol = np.ones(end - start, dtype=bool)  # no info => assume all success (failed stays 0)

            for p, stats in out.items():
                if p not in f:
                    continue
                a = f[p][start:end]
                a = np.asarray(a)
                if a.ndim > 1:
                    if a.shape[1:] == (1,):
                        a = a.reshape(-1)
                    elif a.shape[0] == 1:
                        a = a.reshape(-1)
                    else:
                        continue
                a = a.astype("float64", copy=False)
                a[np.isinf(a)] = np.nan
                a, _ = _x_transform(p, a)
                finite = np.isfinite(a)
                if not finite.any():
                    continue

                # digitize to even-width bins (right-closed like pd.cut default)
                edges = stats["edges"]
                idx = np.digitize(a[finite], edges, right=True) - 1
                # clamp to [0, nbins-1]
                idx[idx < 0] = 0
                nb = edges.size - 1
                idx[idx >= nb] = nb - 1

                # update totals/failed for rows with finite a
                np.add.at(stats["total"], idx, 1)
                if has_sol:
                    np.add.at(stats["failed"], idx, (~sol[finite]).astype(np.int64))

    return out

def generate_plots_for_file(
    path: Path,
    *,
    targets: List[str],
    filters_exprs: List[str],
    computed_map: Dict[str, str],
    plot_types: List[str],
    output_dir: Path,
    computed_meta: Dict[str, Dict[str, str]] | None = None,
    shap_interpolate: bool = False,
    pdf_smooth: bool = False,
    ml_pairwise_settings: Dict[str, Any] | None = None,
    strip_settings: Dict[str, Any] | None = None,
    chunk_size: int = 500_000,
    n_jobs: int = 1,
    batch_size: int = 100_000,
    downcast_float32: bool = False,
) -> None:
    # Registry (respect computed labels/units if provided)
    registry = get_registry()
    if computed_meta:
        registry = ComputedAwareRegistry(registry, computed_meta)

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
        computed_map=computed_map,
        chunk_size=chunk_size,
        downcast_float32=downcast_float32,
        verbose=True,
    )
    if df.empty:
        print("   ⚠️  No data after filtering. Skipping file.")
        return

    # Keep ALL rows (after config filters) to include failures for the "5th quartile"
    df_all = df.copy()

    # Dedupe targets preserving order
    seen = set()
    all_targets = [t for t in targets if not (t in seen or seen.add(t))]

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
        ("strip",       "ddstartup.postprocessing.plot_strip_functions",               "generate_strip_plot"),
        ("quartprob",   "ddstartup.postprocessing.plot_quartile_probability_functions","quartile_probability_plot"),
    ]

    inner_dims = (df.attrs or {}).get("_inner_dims", {})

    for target in all_targets:
        if target not in df.columns:
            print(f"   ⚠️  Target '{target}' not present. Skipping.")
            continue

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
        if computed_meta and computed_meta.get(target, {}).get("unit"):
            tunit = computed_meta[target]["unit"]
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
            plot_name_prefix=path.stem,
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
