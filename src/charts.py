"""Static charts for the README (docs/images/). Tableau holds the interactive versions."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import IMAGES, PROCESSED  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b9b8b2"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def growth_chart(daily: pd.DataFrame):
    show = ["Buy and hold (Nifty ETF)", "Equal weight", "Momentum 12-1, top 10", "SMA 50/200"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in daily.strategy.unique():
        if name in show:
            continue
        g = daily[daily.strategy == name]
        ax.plot(g.date, (1 + g.ret).cumprod(), color=MUTED, lw=1, zorder=1)
    for color, name in zip(SERIES, show):
        g = daily[daily.strategy == name]
        w = (1 + g.ret).cumprod()
        ax.plot(g.date, w, color=color, lw=2, label=name, zorder=3)
        ax.annotate(f"{name}  {w.iloc[-1]:.1f}x", (g.date.iloc[-1], w.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v:g}"))
    ax.set_title("Growth of ₹1 since Jan 2017, after costs (grey = other portfolios)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(right=daily.date.max() + pd.Timedelta(days=900))  # room for end labels
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2017, daily.date.max().year + 1, 3)]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(IMAGES / "growth.png", dpi=160)
    plt.close(fig)


def sharpe_chart(summ: pd.DataFrame):
    s = summ.sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(s))
    is_bh = s.strategy.eq("Buy and hold (Nifty ETF)").values
    colors = np.where(is_bh, SERIES[1], SERIES[0])
    ax.hlines(y, s.sharpe_lo, s.sharpe_hi, color=colors, lw=2, alpha=0.45)
    ax.scatter(s.sharpe, y, color=colors, s=60, zorder=3, edgecolor=SURFACE, linewidth=2)
    bh = s.loc[is_bh, "sharpe"].iloc[0]
    ax.axvline(bh, color=SERIES[1], lw=1, ls="--")
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(y, s.strategy)
    for yi, v in zip(y, s.sharpe):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_2)
    ax.set_xlabel("Sharpe ratio (dot) with 95% bootstrap range (bar), Jan 2017 to Oct 2026")
    ax.set_title("Most of the gaps sit inside the noise")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMAGES / "sharpe_ranges.png", dpi=160)
    plt.close(fig)


def regime_chart(reg: pd.DataFrame, close: pd.Series, avg_corr: pd.Series):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, height_ratios=[3, 2])
    stressed = reg.regime.eq("Stressed")
    spans = (stressed != stressed.shift()).cumsum()[stressed]
    for _, g in spans.groupby(spans):
        for ax in (a1, a2):
            ax.axvspan(g.index.min(), g.index.max(), color=SERIES[1], alpha=0.18, lw=0)
    a1.plot(close.index, close.values, color=SERIES[0], lw=1.6)
    a1.set_title("Nifty 50 with stressed regimes shaded (Markov switching model)")
    a1.set_ylabel("Index level")
    a2.plot(avg_corr.index, avg_corr.values, color=SERIES[0], lw=1.6)
    a2.set_ylabel("Avg correlation\n(63 days)")
    a2.set_title("Average correlation between stocks: diversification fades in stress", fontsize=11)
    a2.xaxis.set_major_locator(mdates.YearLocator(2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(IMAGES / "regimes.png", dpi=160)
    plt.close(fig)


def corr_chart(corr: pd.DataFrame, clusters: pd.DataFrame):
    order = clusters.sort_values("dendro_order").ticker.tolist()
    m = corr.pivot(index="ticker_a", columns="ticker_b", values="rho").loc[order, order]
    labels = [t.removesuffix(".NS") for t in order]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(9, 8.4))
    im = ax.imshow(m.values, cmap=cmap, vmin=0, vmax=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.grid(False)
    cl = clusters.set_index("ticker").loc[order, "cluster"].values
    edges = np.flatnonzero(np.diff(cl)) + 0.5
    for e in edges:
        ax.axhline(e, color=SURFACE, lw=1.5)
        ax.axvline(e, color=SURFACE, lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Correlation of daily returns, last 3 years")
    cb.outline.set_visible(False)
    ax.set_title("Nifty 50 stocks fall into 10 clusters (Ward linkage on correlation distance)")
    fig.tight_layout()
    fig.savefig(IMAGES / "correlation.png", dpi=160)
    plt.close(fig)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    import data
    daily = pd.read_parquet(PROCESSED / "daily_all.parquet")
    growth_chart(daily)
    sharpe_chart(pd.read_parquet(PROCESSED / "summary.parquet"))
    reg = pd.read_parquet(PROCESSED / "regimes.parquet")
    regime_chart(reg, data.index_close(), pd.read_parquet(PROCESSED / "avg_corr.parquet").avg_corr_63d)
    corr_chart(pd.read_parquet(PROCESSED / "corr.parquet"), pd.read_parquet(PROCESSED / "clusters.parquet"))
    print("charts written to", IMAGES)


if __name__ == "__main__":
    main()
