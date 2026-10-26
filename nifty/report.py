"""README charts and the CSV extracts behind the Tableau dashboard."""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import data  # noqa: E402
from .data import ETF, IMAGES, INDEX, OUT, TABLEAU  # noqa: E402
from .strategies import BENCH  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b2", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=160)
    plt.close(fig)


def growth_chart(daily):
    show = [BENCH, "Equal weight", "Momentum 12-1, top 10", "SMA 50/200"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, g in daily.groupby("strategy"):
        if name not in show:
            ax.plot(g.date, (1 + g.ret).cumprod(), color=MUTED, lw=1, zorder=1)
    for color, name in zip(SERIES, show):
        g = daily[daily.strategy == name]
        w = (1 + g.ret).cumprod()
        ax.plot(g.date, w, color=color, lw=2, label=name, zorder=3)
        ax.annotate(f"{name}  {w.iloc[-1]:.1f}x", (g.date.iloc[-1], w.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v:g}"))
    ax.set_xlim(right=daily.date.max() + pd.Timedelta(days=900))  # room for end labels
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2017, daily.date.max().year + 1, 3)]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Growth of ₹1 since Jan 2017, after costs (grey = other portfolios)")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "growth.png")


def sharpe_chart(table):
    t = table.sort_values("sharpe")
    y = np.arange(len(t))
    colors = np.where(t.strategy.eq(BENCH), SERIES[1], SERIES[0])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hlines(y, t.sharpe_lo, t.sharpe_hi, color=colors, lw=2, alpha=0.45)
    ax.scatter(t.sharpe, y, color=colors, s=60, zorder=3, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, t.sharpe):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_2)
    ax.axvline(t.loc[t.strategy.eq(BENCH), "sharpe"].iloc[0], color=SERIES[1], lw=1, ls="--")
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(y, t.strategy)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Sharpe ratio (dot) with 95% bootstrap range (bar), Jan 2017 to Oct 2026")
    ax.set_title("Most of the gaps sit inside the noise")
    save(fig, "sharpe_ranges.png")


def regime_chart(reg, close, avg_corr):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, height_ratios=[3, 2])
    stressed = reg.regime.eq("Stressed")
    spells = (stressed != stressed.shift()).cumsum()[stressed]
    for _, g in spells.groupby(spells):
        for ax in (top, bottom):
            ax.axvspan(g.index.min(), g.index.max(), color=SERIES[1], alpha=0.18, lw=0)
    top.plot(close.index, close.values, color=SERIES[0], lw=1.6)
    top.set_title("Nifty 50 with stressed regimes shaded (Markov switching model)")
    top.set_ylabel("Index level")
    bottom.plot(avg_corr.index, avg_corr.values, color=SERIES[0], lw=1.6)
    bottom.set_ylabel("Avg correlation\n(63 days)")
    bottom.set_title("Average correlation between stocks: diversification fades in stress", fontsize=11)
    bottom.xaxis.set_major_locator(mdates.YearLocator(2))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "regimes.png")


def correlation_chart(pairs, groups):
    order = groups.sort_values("order").ticker.tolist()
    m = pairs.pivot(index="ticker_a", columns="ticker_b", values="rho").loc[order, order]
    labels = [t.removesuffix(".NS") for t in order]
    fig, ax = plt.subplots(figsize=(9, 8.4))
    im = ax.imshow(m.values, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.grid(False)
    for e in np.flatnonzero(np.diff(groups.set_index("ticker").loc[order, "cluster"].values)) + 0.5:
        ax.axhline(e, color=SURFACE, lw=1.5)
        ax.axvline(e, color=SURFACE, lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Correlation of daily returns, last 3 years")
    cb.outline.set_visible(False)
    ax.set_title("Nifty 50 stocks fall into 10 clusters (Ward linkage on correlation distance)")
    save(fig, "correlation.png")


def tableau_extracts():
    TABLEAU.mkdir(exist_ok=True)
    P = lambda name: pd.read_parquet(OUT / f"{name}.parquet")  # noqa: E731
    name = lambda t: {INDEX: "Nifty 50 index", ETF: "Nifty 50 ETF"}.get(t, t.removesuffix(".NS"))  # noqa: E731

    m = data.query("SELECT * FROM stock_metrics").merge(P("capm"), on="ticker", how="left") \
        .merge(P("clusters"), on="ticker", how="left")
    m["symbol"] = m.ticker.map(name)
    m["industry"] = m.industry.fillna("Benchmark")
    m.to_csv(TABLEAU / "stocks.csv", index=False)

    d = data.query("""SELECT r.date, r.ticker, r.ret, r.wealth, d.drawdown, v.vol_21d
                      FROM returns r JOIN drawdowns d USING (date, ticker)
                      LEFT JOIN rolling_vol v USING (date, ticker)""")
    d["date"] = pd.to_datetime(d.date)
    d = d.merge(P("rolling_beta"), on=["date", "ticker"], how="left").assign(symbol=lambda x: x.ticker.map(name))
    d.round(6).to_csv(TABLEAU / "stocks_daily.csv", index=False)

    c = P("corr").merge(P("clusters").add_suffix("_a"), on="ticker_a").merge(P("clusters").add_suffix("_b"), on="ticker_b")
    c.assign(symbol_a=c.ticker_a.map(name), symbol_b=c.ticker_b.map(name)).round(4).to_csv(TABLEAU / "correlation.csv", index=False)

    P("var_backtest").round(6).to_csv(TABLEAU / "var_daily.csv", index=False)
    v = P("var_summary")
    v["result"] = np.where((v.kupiec_p > 0.05) & (v.christoffersen_p > 0.05), "Pass", "Fail")
    v.round(4).to_csv(TABLEAU / "var_tests.csv", index=False)

    s = P("strategies").sort_values(["strategy", "date"])
    s["wealth"] = s.groupby("strategy").ret.transform(lambda r: (1 + r).cumprod())
    s["drawdown"] = s.groupby("strategy").wealth.transform(lambda w: w / w.cummax() - 1)
    s.merge(P("regimes").regime.reset_index(), on="date", how="left").round(6) \
        .to_csv(TABLEAU / "strategies_daily.csv", index=False)
    P("summary").round(4).to_csv(TABLEAU / "strategies.csv", index=False)
    P("by_regime").melt("strategy", var_name="regime", value_name="ann_return").round(4) \
        .to_csv(TABLEAU / "strategies_by_regime.csv", index=False)
    P("sensitivity").round(4).to_csv(TABLEAU / "sensitivity.csv", index=False)

    w = P("weights").merge(data.query("SELECT ticker, industry FROM constituents"), on="ticker", how="left")
    w.assign(symbol=w.ticker.map(name)).round(6).to_csv(TABLEAU / "weights.csv", index=False)

    r = P("regimes").join(data.index_close().rename("nifty_close")).join(P("avg_corr"))
    r["avg_corr_63d"] = r.avg_corr_63d.ffill()
    r.reset_index().round(6).to_csv(TABLEAU / "regimes.csv", index=False)
    P("regime_summary").round(4).to_csv(TABLEAU / "regime_summary.csv", index=False)


def run():
    IMAGES.mkdir(exist_ok=True)
    growth_chart(pd.read_parquet(OUT / "strategies.parquet"))
    sharpe_chart(pd.read_parquet(OUT / "summary.parquet"))
    regime_chart(pd.read_parquet(OUT / "regimes.parquet"), data.index_close(),
                 pd.read_parquet(OUT / "avg_corr.parquet").avg_corr_63d)
    correlation_chart(pd.read_parquet(OUT / "corr.parquet"), pd.read_parquet(OUT / "clusters.parquet"))
    tableau_extracts()
    print(f"charts in {IMAGES.name}/, Tableau files in {TABLEAU.name}/")
