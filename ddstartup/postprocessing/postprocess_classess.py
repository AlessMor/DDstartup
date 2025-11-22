from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List



# -----------------------------------------------------------------------------
# Project registry (schema + units)
# -----------------------------------------------------------------------------
from ddstartup.utils.parameter_registry import get_registry, PARAMETER_SCHEMA
from ddstartup.utils.io_functions import resolve_file_path, resolve_h5_inputs, h5_to_df_core

class PlotOrchestrator:
    """
    Minimal orchestrator that:
      - resolves files
      - applies HDF5/runtime defaults
      - parses filters & computed variables
      - runs generate_plots_for_file for selected plot types
    Keeps CLI thin and testable.
    """
    DEFAULT_PLOTS = ["kde","parcoords","pdf","importance","kmeans",
                     "contour","shap","ml_pairwise","strip","quartprob"]

    def __init__(self, config: Dict[str, Any], root: Path, debug: bool = False):
        self.config = config
        self.root = root
        self.debug = bool(debug)
        self.files: List[Path] | None = None
        self.runtime: Dict[str, Any] | None = None
        self.filters_exprs: List[str] | None = None
        self.computed_map: Dict[str, str] | None = None
        self.computed_meta: Dict[str, Dict[str, str]] | None = None
        self.targets: List[str] = list(self.config.get("target_variables", ["unrealized_profits", "t_startup"]))
        # derive default plot types from config if present, otherwise use DEFAULT_PLOTS
        plots_cfg = self.config.get("plots", {}) or {}
        if plots_cfg.get("generate_all", True):
            self.plot_types = list(self.DEFAULT_PLOTS)
        else:
            allowed = self.DEFAULT_PLOTS
            self.plot_types = [p for p in allowed if plots_cfg.get(p, False)]

    def available_plots(self) -> List[str]:
        # single source of truth for CLI/display
        return list(self.DEFAULT_PLOTS)

    def prepare(self) -> None:
        # resolve files, runtime defaults, and parse filters/computed
        self.files = resolve_file_paths(self.config, self.root)
        self.runtime = apply_h5_runtime_defaults(self.config, self.files)
        self.filters_exprs, self.computed_map, self.computed_meta = parse_filters_and_computed(self.config)

    def run(
        self,
        output_dir: Path,
        plot_types: List[str] | str | None = None,
        shap_interpolate: bool = False,
        pdf_smooth: bool = False,
        ml_pairwise_settings: Dict[str, Any] | None = None,
        strip_settings: Dict[str, Any] | None = None,
    ) -> None:
        if self.files is None or self.runtime is None or self.filters_exprs is None:
            self.prepare()

        # normalize plot_types
        if plot_types is None or plot_types == "all":
            plot_types = self.plot_types
        elif isinstance(plot_types, str):
            plot_types = [plot_types]
        # sanitize against available
        plot_types = [p for p in plot_types if p in self.available_plots()]

        rt = self.runtime or {}
        chunk_def = int(rt.get("chunk_size", 500_000))
        n_jobs_def = int(rt.get("n_jobs", 1))
        batch_def = int(rt.get("batch_size", 100_000))
        downcast_def = bool(rt.get("downcast_float32", False))

        ml_pairwise_settings = ml_pairwise_settings or {}
        ml_pairwise_settings.setdefault("n_jobs", n_jobs_def)
        ml_pairwise_settings.setdefault("batch_size", batch_def)

        for path in self.files:
            if self.debug:
                print(f"Processing {path.name} (plots={plot_types})")
            generate_plots_for_file(
                path,
                targets=self.targets,
                filters_exprs=self.filters_exprs or [],
                computed_map=self.computed_map or {},
                computed_meta=self.computed_meta,
                plot_types=plot_types,
                output_dir=output_dir,
                shap_interpolate=shap_interpolate,
                pdf_smooth=pdf_smooth,
                ml_pairwise_settings=ml_pairwise_settings,
                strip_settings=strip_settings or {},
                chunk_size=chunk_def,
                n_jobs=n_jobs_def,
                batch_size=batch_def,
                downcast_float32=downcast_def,
            )
