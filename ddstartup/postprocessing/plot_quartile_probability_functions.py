from pathlib import Path
import numpy as np
import pandas as pd
import os, re

# Headless by default
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---- MathText: always render $...$ instead of showing literally ----
try:
    mpl.rcParams["text.usetex"] = False
    if "text.parse_math" in mpl.rcParams:
        mpl.rcParams["text.parse_math"] = True  # Matplotlib ≥3.8
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
except Exception:
    pass

# ---- robust “per Joule” detector (matches 1/J, J^-1, per J, etc.; case-insensitive) ----
PER_J_RE = re.compile(r'(?i)(?:^|[^a-z])(?:1\s*/\s*j|j\s*(?:\^|-)?\s*-?1|per\s*j)(?:$|[^a-z])')

def _norm_unit(u: str) -> str:
    """Normalize a unit string for matching (lower; map 'joule'->'j'; strip brackets/odd chars)."""
    return re.sub(r"[^a-z0-9/^\-\s]", "", str(u or "").lower().replace("joule", "j"))

def _clean_symbol_label(s: str | None) -> str:
    """
    Make registry-provided symbols safe for Matplotlib mathtext:
      - Strip optional r/u/ur prefix + surrounding quotes (r"...", '...').
      - Trim whitespace.
      - Leave $...$ intact so mathtext renders.
    """
    if not s:
        return ""
    s = str(s).strip()
    m = re.fullmatch(r'(?is)\s*(?:ur|ru|r|u)?\s*([\'"])(.*)\1\s*', s)
    return m.group(2) if m else s

def _strip_trailing_unit(label: str) -> str:
    """Remove a trailing ' [ ... ]' unit suffix from a label if present."""
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", label or "")

def _escape_dollars(s: str | None) -> str:
    """Escape $ so units like $/kWh don't break mathtext parsing."""
    return "" if not s else s.replace("$", r"\$")


def quartile_probability_plot(
    *,
    df: pd.DataFrame,
    target: str,
    inputs=None,
    input_parameters=None,
    target_unit: str | None = None,
    output_dir=None,
    outputs_dir=None,
    file_type: str = "",
    plot_name_prefix: str | None = None,
    plot_name: str | None = None,
    registry=None,
    COUNT_FAILED: bool = True,             # add grey series P(FAILED | x)
    PLOT_STYLE: str = "line",              # "line" | "bar"
    min_per_bin: int = 1,
    AVG_POINTS: int = 10,                  # for continuous params
    discrete_unique_threshold: int = 128,  # treat as exact-value plot if <= this
    FORCE_DISCRETE: set[str] = frozenset({"I_target"}),
    XTICK_CAP: int = 10,                   # show all discrete ticks up to this many
    **_,
):
    if df is None or len(df) == 0:
        print("   No data. Skipping quartile-probability plot.")
        return

    _inputs = input_parameters if inputs is None else inputs
    outdir = Path(outputs_dir if output_dir is None else output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{plot_name_prefix}__quartile_probs"
        if plot_name_prefix
        else (Path(plot_name).stem if plot_name else "quartile_probs")
    )

    PLOT_STYLE = (PLOT_STYLE or "line").lower()
    if PLOT_STYLE not in {"line", "bar"}:
        PLOT_STYLE = "line"

    if registry is None:
        from ddstartup.utils.parameter_registry import get_registry as _get_registry
        registry = _get_registry()

    # ---------- target quartiles on successes ----------
    t = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    sol = df["sol_success"].astype(bool) if "sol_success" in df.columns else pd.Series(True, index=df.index)
    succ = sol & t.notna()
    if not succ.any():
        print(f"   No successful finite '{target}'. Skipping.")
        return

    qbins = pd.qcut(t[succ], q=4, duplicates="drop")
    qcodes = qbins.cat.codes.to_numpy()
    qcats = list(qbins.cat.categories)  # IntervalIndex
    kq = len(qcats)
    qlabels = [f"Q{i+1}: {iv.left:.2e}–{iv.right:.2e}" for i, iv in enumerate(qcats)]

    # ---------- select scalar, varying inputs ----------
    inner = (df.attrs or {}).get("_inner_dims", {})
    cand = [
        c for c in list(_inputs or [])
        if c != "sol_success" and c in df.columns
        and int(inner.get(c, 1)) == 1
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    varying = []
    for c in cand:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.dropna().empty:
            continue
        std = float(s.std()); mean = abs(float(s.mean()))
        if std > 1e-10 and (mean == 0 or std / max(mean, 1e-30) > 1e-6):
            varying.append(c)
    if not varying:
        print("   No varying scalar inputs to plot. Skipping.")
        return

    # ---------- colors & legend order ----------
    quart_colors = ["#2563EB", "#059669", "#D97706", "#DC2626"][:kq]  # Q1..Q4
    cat_colors = {qlabels[i]: quart_colors[i] for i in range(kq)}
    FAILED_COLOR, FAILED_ALPHA = "#808080", 0.3
    if PLOT_STYLE == "bar":
        legend_proxies = [Patch(facecolor=cat_colors[ql], edgecolor="none", label=ql) for ql in qlabels]
        if COUNT_FAILED:
            legend_proxies.append(Patch(facecolor=FAILED_COLOR, edgecolor="none", alpha=FAILED_ALPHA, label="FAILED"))
    else:
        legend_proxies = [Line2D([0],[0], color=cat_colors[ql], marker="o", linewidth=1.5, label=ql) for ql in qlabels]
        if COUNT_FAILED:
            legend_proxies.append(Line2D([0],[0], color=FAILED_COLOR, alpha=FAILED_ALPHA, marker="o", linewidth=1.5, label="FAILED"))

    def _fmt(v: float) -> str:
        if not np.isfinite(v) or v == 0:
            return "0"
        a = abs(v)
        return f"{v:.2e}" if (a >= 1e6 or a < 1e-3) else f"{v:.3g}"

    # ---------- transforms (t_startup→days, 1/J→$/kWh; label cleaning) ----------
    FORCE_KWH_NAMES = {"price_of_electricity", "c_kwh", "ckwh", "c_kwhn"}

    def _x_transform(name: str, unit_str: str | None, label_str: str | None, x: np.ndarray) -> tuple[np.ndarray, str | None]:
        lname = (name or "").lower()
        utok  = _norm_unit(unit_str or "")
        ltok  = (label_str or "").lower()

        # t_startup -> days
        if lname in {"t_startup", "tstartup"} or "t_startup" in lname:
            return x / 86400.0, "days"

        # name/label hints for kWh
        if (lname in FORCE_KWH_NAMES) or ("kwh" in ltok):
            if PER_J_RE.search(utok):  # stored as $/J
                return x * 3.6e6, "$/kWh"
            return x, "$/kWh"

        # otherwise: infer from unit
        if "kwh" in utok:
            return (x * 3.6e6, "$/kWh") if PER_J_RE.search(utok) else (x, "$/kWh")
        if PER_J_RE.search(utok):
            return x * 3.6e6, "$/kWh"

        return x, None

    # ---------- layout ----------
    n = len(varying)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.0 * nrows + 2.0))
    axes = np.atleast_1d(axes).ravel()

    fail_mask = (~sol) | (~t.notna())
    csv_rows = []
    rpow = 9  # rounding precision for discrete grouping

    for i, param in enumerate(varying):
        ax = axes[i]

        # get & clean label and unit up front
        raw_label = getattr(registry, "get_param_label", lambda n, **k: n)(param) or ""
        label0    = _clean_symbol_label(raw_label)  # keep $...$ for mathtext
        unit0     = getattr(registry, "get_param_unit",  lambda n, **k: None)(param) or ""

        # numeric vectors
        x_all  = pd.to_numeric(df[param], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64, copy=False)
        x_succ = pd.to_numeric(df.loc[succ, param], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64, copy=False)

        # primary transform
        x_all,  unit_override = _x_transform(param, unit0, label0, x_all)
        x_succ, _             = _x_transform(param, unit0, label0, x_succ)

        # FORCE $/kWh for electricity price (by name/label/unit/value scale)
        name_l  = (param or "").lower()
        label_l = (label0 or "").lower()
        unit_l  = _norm_unit(unit0)
        looks_kwh   = ("kwh" in name_l) or ("kwh" in label_l)
        looks_per_j = PER_J_RE.search(unit_l) is not None or ("[1/j]" in label_l.replace(" ", "")) or ("1/j" in unit_l)
        must_be_kwh = looks_kwh or looks_per_j or (name_l in FORCE_KWH_NAMES)
        if must_be_kwh:
            need_scale = looks_per_j or (np.nanmedian(np.abs(x_all)) < 1e-3 and np.nanmax(np.abs(x_all)) < 1.0)
            if need_scale:
                x_all  = x_all  * 3.6e6
                x_succ = x_succ * 3.6e6
            unit_override = "$/kWh"

        # compute after any scaling
        finite_all = np.isfinite(x_all)
        xa = x_all[finite_all]
        if xa.size == 0:
            ax.set_visible(False); continue

        # clean label to avoid trailing "[...]" duplication and keep mathtext
        p_label = _strip_trailing_unit(label0)
        p_unit  = _escape_dollars(unit_override or unit0)  # ESCAPE $ IN UNIT ONLY

        # ---------- discrete vs continuous ----------
        xr = np.round(xa, rpow)
        levels = np.unique(xr)
        nlevels = levels.size
        discrete_mode = (nlevels <= discrete_unique_threshold) or (nlevels <= 256 and nlevels / xa.size <= 1e-3)
        if param in FORCE_DISCRETE:
            discrete_mode = True

        if discrete_mode:
            # exact levels
            levels = np.sort(levels)
            idx_map = {v: j for j, v in enumerate(levels)}

            xr_all = np.round(x_all[finite_all], rpow)
            idx_all = np.fromiter((idx_map[v] for v in xr_all), dtype=np.int64, count=xr_all.size)
            totals = np.bincount(idx_all, minlength=len(levels))

            if COUNT_FAILED:
                idx_fail = idx_all[fail_mask.values[finite_all]]
                fails = np.bincount(idx_fail, minlength=len(levels))
            else:
                fails = np.zeros(len(levels), dtype=np.int64)

            finite_succ = np.isfinite(x_succ)
            qr = np.round(x_succ[finite_succ], rpow)
            valid = np.isin(qr, levels)
            idx_s = np.fromiter((idx_map[v] for v in qr[valid]), dtype=np.int64, count=int(valid.sum()))
            q_s = qcodes[finite_succ][valid]

            counts_q = np.zeros((len(levels), kq), dtype=np.int64)
            for q in range(kq):
                sel = (q_s == q)
                if sel.any():
                    np.add.at(counts_q[:, q], idx_s[sel], 1)

            keep = totals >= max(1, min_per_bin)
            if not np.any(keep):
                ax.set_visible(False); continue

            x_pos   = levels[keep]   # exact values (already converted if needed)
            totals  = totals[keep]
            counts_q = counts_q[keep, :]
            fails   = fails[keep] if COUNT_FAILED else None

        else:
            # continuous: even-width bins on ALL rows; x plotted at per-bin mean
            vmin, vmax = float(np.min(xa)), float(np.max(xa))
            if vmin == vmax:
                eps = (abs(vmin) + 1.0) * 1e-12
                edges = np.array([vmin - eps, vmax + eps], dtype=np.float64)
            else:
                edges = np.linspace(vmin, vmax, AVG_POINTS + 1, dtype=np.float64)
            nb = edges.size - 1

            idx_all = np.digitize(xa, edges, right=True) - 1
            idx_all[idx_all < 0] = 0; idx_all[idx_all >= nb] = nb - 1
            totals = np.bincount(idx_all, minlength=nb)

            sumx = np.bincount(idx_all, weights=xa, minlength=nb)
            means = np.divide(sumx, totals, out=np.full(nb, np.nan, dtype=np.float64), where=totals > 0)

            if COUNT_FAILED:
                idx_fail = idx_all[fail_mask.values[finite_all]]
                fails = np.bincount(idx_fail, minlength=nb)
            else:
                fails = np.zeros(nb, dtype=np.int64)

            finite_succ = np.isfinite(x_succ)
            xs = x_succ[finite_succ]
            qs = qcodes[finite_succ].astype(np.int32, copy=False)
            idx_s = np.digitize(xs, edges, right=True) - 1
            idx_s[idx_s < 0] = 0; idx_s[idx_s >= nb] = nb - 1

            counts_q = np.zeros((nb, kq), dtype=np.int64)
            for q in range(kq):
                sel = (qs == q)
                if sel.any():
                    np.add.at(counts_q[:, q], idx_s[sel], 1)

            keep = totals >= max(1, min_per_bin)
            if not np.any(keep):
                ax.set_visible(False); continue

            x_pos   = means[keep]
            totals  = totals[keep]
            counts_q = counts_q[keep, :]
            fails   = fails[keep] if COUNT_FAILED else None

        # ---------- probabilities ----------
        denom = totals.astype(np.float64)
        denom[denom == 0] = np.nan
        probs_q = np.nan_to_num(counts_q / denom[:, None], nan=0.0)
        prob_failed = (np.nan_to_num(fails / denom, nan=0.0) if (COUNT_FAILED and fails is not None) else None)

        # ---------- draw ----------
        if PLOT_STYLE == "bar":
            xpos = np.arange(x_pos.size)
            bottom = np.zeros(x_pos.size, float)
            for q, lab in enumerate(qlabels):
                ax.bar(xpos, probs_q[:, q], bottom=bottom, width=0.9, align="center",
                       edgecolor="none", color=cat_colors[lab], label=lab)
                bottom += probs_q[:, q]
            if COUNT_FAILED and prob_failed is not None:
                ax.bar(xpos, prob_failed, bottom=bottom, width=0.9, align="center",
                       edgecolor="none", color=FAILED_COLOR, alpha=FAILED_ALPHA, label="FAILED")
            tick_pos = xpos
            tick_labels = [_fmt(v) for v in x_pos]
        else:
            for q, lab in enumerate(qlabels):
                ax.plot(x_pos, probs_q[:, q], marker="o", linestyle="-", linewidth=1.5, markersize=4,
                        color=cat_colors[lab], label=lab)
            if COUNT_FAILED and prob_failed is not None:
                ax.plot(x_pos, prob_failed, marker="o", linestyle="-", linewidth=1.5, markersize=4,
                        color=FAILED_COLOR, alpha=FAILED_ALPHA, label="FAILED")
            tick_pos = x_pos
            tick_labels = [_fmt(v) for v in x_pos]

        # ticks: show all discrete ticks if <= XTICK_CAP; otherwise thin
        if discrete_mode and len(tick_labels) <= XTICK_CAP:
            pass  # keep all
        else:
            cap = XTICK_CAP if discrete_mode else 10
            if len(tick_labels) > cap:
                step = int(np.ceil(len(tick_labels) / cap))
                tick_labels = [lbl if (j % step == 0) else "" for j, lbl in enumerate(tick_labels)]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability", fontsize=9)

        # title: cleaned mathtext label + our (escaped) unit
        ax.set_title(p_label if not p_unit else f"{p_label} [{p_unit}]", fontsize=10)

        # CSV
        for j in range(x_pos.size):
            row = {
                "parameter": param,
                "bin_index": int(j),
                "x_value_or_mean": float(x_pos[j]),
                "n_total_bin": int(totals[j]),
            }
            for q, lab in enumerate(qlabels):
                row[f"prob_{lab}"] = float(probs_q[j, q])
            if COUNT_FAILED and prob_failed is not None:
                row["prob_FAILED"] = float(prob_failed[j])
            csv_rows.append(row)

    # hide unused axes
    for j in range(len(varying), len(axes)):
        axes[j].set_visible(False)

    # figure title: clean mathtext for the target symbol; escape dollars in unit only
    raw_tlabel = getattr(registry, "get_param_label", lambda n, **k: n)(target)
    t_label = _strip_trailing_unit(_clean_symbol_label(raw_tlabel))
    t_unit  = _escape_dollars(target_unit or getattr(registry, "get_param_unit", lambda n, **k: None)(target) or "")
    title = f"P(quartile | parameter value) wrt {t_label if not t_unit else f'{t_label} [{t_unit}]'}"
    if file_type:
        title += f" • {file_type}"
    fig.suptitle(title, fontsize=14)

    legend_labels = qlabels + (["FAILED"] if COUNT_FAILED else [])
    fig.legend(legend_proxies, legend_labels, loc="lower center",
               ncol=min(5, len(legend_labels)), bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_png = outdir / f"{stem}__{target}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"   Quartile-probability plot saved: {out_png.name}")

    if csv_rows:
        out_csv = outdir / f"{stem}__{target}.csv"
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
        print(f"   Quartile-probability CSV saved: {out_csv.name}")
