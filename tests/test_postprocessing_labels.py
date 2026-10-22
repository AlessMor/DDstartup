"""Plot labels and display-unit conversions for legacy paper results."""

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from src.registry import parameter_registry as registry
from src.postprocessing.plot_quartile_probability_functions import quartile_probability_plot
from src.postprocessing.plot_utils_functions import get_param_label


@pytest.mark.parametrize("name, symbol", [
    ("V_plasma", r"$V_{\mathrm{plasma}}$"),
    ("eta_th", r"$\eta_{\mathrm{th}}$"),
    ("tau_p_T", r"$\tau_{p,\mathrm{T}}$"),
    ("P_DT_eq", r"$P_{\mathrm{DT,eq}}$"),
    ("price_of_electricity", r"$C_{\mathrm{kWh}}$"),
])
def test_scientific_symbols(name, symbol):
    assert registry.get_symbol(name) == symbol
    assert get_param_label(name).startswith(symbol)


@pytest.mark.parametrize("source_unit, factor", [("1/J", 3.6e6), ("$/J", 3.6e6), ("$/kWh", 1.)])
def test_quartile_electricity_price_units(tmp_path, monkeypatch, source_unit, factor):
    class PriceRegistry:
        get_symbol = staticmethod(registry.get_symbol)

        @staticmethod
        def get_unit(name):
            return source_unit if name == "price_of_electricity" else "s"

    prices = np.array([.1, .2, .3, .4])
    df = pd.DataFrame({
        "price_of_electricity": np.repeat(prices / factor, 4),
        "t_startup": np.tile([1., 2., 3., 4.], 4),
    })
    failed = pd.DataFrame({"price_of_electricity": np.repeat(prices / factor, [2, 4, 6, 8])})
    labels = []
    savefig = matplotlib.figure.Figure.savefig

    def capture_savefig(fig, *args, **kwargs):
        labels.extend(text.get_text() for ax in fig.axes for text in ax.get_legend().get_texts())
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_savefig)
    quartile_probability_plot(
        df=df, target="t_startup", inputs=["price_of_electricity"],
        registry=PriceRegistry(), output_dir=tmp_path,
        failed_counts_summary={"_failed_df": failed}, show_titles=False,
    )
    result = pd.read_csv(tmp_path / "quartile_probs__t_startup.csv")
    np.testing.assert_allclose(result["x_value_or_mean"], prices)
    np.testing.assert_array_equal(result["n_total_bin"], [6, 8, 10, 12])
    assert labels == [r"$C_{\mathrm{kWh}}$ [\$/kWh]"]
    assert (tmp_path / "quartile_probs__t_startup.png").stat().st_size > 0
