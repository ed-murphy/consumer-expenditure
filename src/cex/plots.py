"""Charts of the results, saved as PNGs next to the CSV tables (the CSVs hold the exact values)."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from cex import stubs
from cex.clean import INCOME_QUINTILES
from cex.model import EngelResults, model_sample
from cex.survey import Z_95

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Categorical slots, always assigned in this order (validated for color-vision deficiency as adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

LABELS = stubs.LABELS | {"consumer_units": "Consumer units", "income_before_tax": "Income before taxes"}


def label(name: str) -> str:
    return LABELS.get(name) or stubs.label(name)


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 150,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 9.5,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
}

DOLLARS = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k" if abs(v) >= 1000 else f"${v:,.0f}")


def _figure(width: float, height: float, title: str) -> Figure:
    fig = Figure(figsize=(width, height), layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    return fig


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def _dot_and_whisker(ax, y, estimate, low, high) -> None:
    ax.hlines(y, low, high, color=SERIES[0], linewidth=1.5)
    ax.plot(estimate, y, "o", color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1)
    ax.grid(axis="y", visible=False)


def plot_elasticities(results: EngelResults, path: Path, year: int) -> Path:
    """Expenditure and income elasticities with 95% confidence intervals, one row per category."""
    table = results.elasticities
    rows = table.drop(index="total", errors="ignore").sort_values("expenditure_elasticity")
    y = np.arange(len(rows))
    with mpl.rc_context(STYLE):
        title = f"How spending on each category scales with household resources, {year}"
        fig = _figure(9.5, 0.32 * len(rows) + 1.6, title)
        spending, income = fig.subplots(1, 2, sharey=True)

        _dot_and_whisker(spending, y, rows["expenditure_elasticity"], rows["ci_low"], rows["ci_high"])
        spending.axvline(1, color=INK_SECONDARY, linewidth=1)
        spending.set_title("Elasticity with respect to total spending")
        spending.set_xlabel("below 1: necessity   ·   above 1: luxury")

        _dot_and_whisker(income, y, rows["income_elasticity"], rows["income_ci_low"], rows["income_ci_high"])
        if "total" in table.index:
            benchmark = table.loc["total", "income_elasticity"]
            income.axvline(benchmark, color=INK_SECONDARY, linewidth=1)
            income.annotate(
                f"total spending ({benchmark:.2f})",
                (benchmark, 1),
                xycoords=("data", "axes fraction"),
                xytext=(4, -2),
                textcoords="offset points",
                va="top",
                color=INK_SECONDARY,
            )
        income.set_title("Elasticity with respect to current income")
        income.set_xlabel("compare with total spending, not with 1")

        spending.set_yticks(y, [label(c) for c in rows.index])
        spending.tick_params(axis="y", length=0)
        income.tick_params(axis="y", length=0)
        return _save(fig, path)


def plot_detail_elasticities(results: EngelResults, path: Path, year: int) -> Path:
    """Expenditure elasticities of the detail categories with 95% confidence intervals."""
    rows = results.elasticities.sort_values("expenditure_elasticity")
    y = np.arange(len(rows))
    with mpl.rc_context(STYLE):
        fig = _figure(7.5, 0.25 * len(rows) + 1.3, f"Necessities and luxuries in more detail, {year}")
        ax = fig.subplots()
        _dot_and_whisker(ax, y, rows["expenditure_elasticity"], rows["ci_low"], rows["ci_high"])
        ax.axvline(1, color=INK_SECONDARY, linewidth=1)
        # A few small, lumpy categories have very wide intervals; keep the scale readable and let them run off.
        low, high = rows["expenditure_elasticity"].min(), rows["expenditure_elasticity"].max()
        ax.set_xlim(max(rows["ci_low"].min(), low - 0.75) - 0.05, min(rows["ci_high"].max(), high + 0.75) + 0.05)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title("Elasticity with respect to total spending (bars: 95% confidence intervals)")
        ax.set_xlabel("below 1: necessity   ·   above 1: luxury")
        ax.set_yticks(y, [label(c) for c in rows.index])
        ax.tick_params(axis="y", length=0)
        return _save(fig, path)


SHARE_PANELS = [
    "groceries",
    "food_away",
    "housing",
    "transportation",
    "healthcare",
    "entertainment",
    "insurance_pensions",
    "cash_contributions",
]


def plot_quintile_shares(groups: pd.DataFrame, path: Path, year: int, categories: list[str] = SHARE_PANELS) -> Path:
    """Budget shares across income quintiles, one small panel per category."""
    quintiles = groups[groups["grouping"] == "income_quintile"]
    x = np.arange(len(INCOME_QUINTILES))
    with mpl.rc_context(STYLE):
        fig = _figure(10, 5, f"Share of total spending by income quintile, {year} (bars: 95% confidence intervals)")
        axes = fig.subplots(2, (len(categories) + 1) // 2)
        for ax, category in zip(axes.flat, categories):
            rows = quintiles[quintiles["item"] == category].set_index("group").loc[INCOME_QUINTILES]
            ax.bar(x, rows["share"], width=0.72, color=SERIES[0])
            ci = Z_95 * rows["share_se"]
            ax.errorbar(x, rows["share"], yerr=ci, fmt="none", ecolor=INK_SECONDARY, elinewidth=1)
            ax.set_title(label(category))
            ax.set_xticks(x, ["Low", "2", "3", "4", "High"])
            ax.tick_params(axis="x", length=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(axis="x", visible=False)
        for ax in axes.flat[len(categories) :]:
            ax.set_visible(False)
        fig.supxlabel("Income quintile", color=INK_SECONDARY, fontsize=9)
        return _save(fig, path)


ENGEL_PANELS = ["groceries", "housing", "entertainment", "insurance_pensions"]


def plot_engel_curves(
    households: pd.DataFrame, results: EngelResults, path: Path, year: int, categories: list[str] = ENGEL_PANELS
) -> Path:
    """Binned Engel curves: average budget share within 20 equal-population bins of total spending."""
    sample = model_sample(households)
    weight = sample["weight"].to_numpy()
    total = sample["total_annual"].to_numpy()
    order = np.argsort(total)
    cumulative = np.empty_like(weight)
    cumulative[order] = np.cumsum(weight[order]) / weight.sum()
    bins = np.minimum((cumulative * 20).astype(int), 19)
    by_bin = pd.DataFrame({"bin": bins, "w": weight, "wx": weight * total})
    grouped = by_bin.groupby("bin")
    mean_total = grouped["wx"].sum() / grouped["w"].sum()

    with mpl.rc_context(STYLE):
        fig = _figure(9, 6, f"Engel curves, {year}: budget share by household total spending (unadjusted)")
        axes = fig.subplots(2, (len(categories) + 1) // 2, sharex=True)
        for ax, category in zip(axes.flat, categories):
            share = sample[f"{category}_annual"].to_numpy() / total
            mean_share = pd.Series(weight * share).groupby(bins).sum() / grouped["w"].sum()
            ax.plot(mean_total, mean_share, "o", color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1)
            ax.set_xscale("log")
            ax.set_xticks([10_000, 20_000, 50_000, 100_000, 200_000])
            ax.xaxis.set_major_formatter(DOLLARS)
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=None))
            row = results.elasticities.loc[category]
            ax.set_title(f"{label(category)}: elasticity {row['expenditure_elasticity']:.2f} ({row['classification']})")
        for ax in axes.flat[len(categories) :]:
            ax.set_visible(False)
        fig.supxlabel("Total annual spending (log scale)", color=INK_SECONDARY, fontsize=9)
        return _save(fig, path)


TREND_SHARES = ["housing", "transportation", "insurance_pensions", "groceries", "healthcare", "food_away"]


def plot_trends(trends: pd.DataFrame, path: Path) -> Path:
    """Real income and spending per household, and budget shares, across years."""
    years = sorted(trends["year"].unique())
    base_year = years[-1]
    with mpl.rc_context(STYLE):
        fig = _figure(10, 4, f"Consumer units, {years[0]}–{years[-1]}")
        dollars, shares = fig.subplots(1, 2)

        for slot, item in enumerate(["income_before_tax", "total"]):
            rows = trends[trends["item"] == item].sort_values("year")
            dollars.plot(rows["year"], rows["real_mean"], marker="o", color=SERIES[slot], label=LABELS[item])
            dollars.annotate(
                LABELS[item],
                (rows["year"].iloc[-1], rows["real_mean"].iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
            )
        dollars.set_title(f"Average per consumer unit, in {base_year} dollars")
        dollars.yaxis.set_major_formatter(DOLLARS)
        dollars.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)

        for slot, item in enumerate(TREND_SHARES):
            rows = trends[trends["item"] == item].sort_values("year")
            shares.plot(rows["year"], rows["share"], marker="o", color=SERIES[slot], label=LABELS[item])
        shares.set_title("Share of total spending")
        shares.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        shares.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

        for ax in (dollars, shares):
            ax.set_xticks(years, [str(y) for y in years])
            ax.grid(axis="x", visible=False)
            ax.margins(x=0.08)
        return _save(fig, path)


def plot_curvature(table: pd.DataFrame, path: Path, year: int) -> Path:
    """Elasticities for small and for large budgets from the quadratic budget share model, one row per category."""
    percentiles = sorted(int(col.removeprefix("spending_p")) for col in table if col.startswith("spending_p"))
    low, high = percentiles[0], percentiles[-1]
    middle = f"elasticity_p{percentiles[len(percentiles) // 2]}"
    rows = table.sort_values(middle)
    y = np.arange(len(rows))
    ends = [rows[f"elasticity_p{low}"], rows[f"elasticity_p{high}"]]
    with mpl.rc_context(STYLE):
        fig = _figure(
            7.5, 0.33 * len(rows) + 1.9, f"Does a category's elasticity depend on the size of the budget? {year}"
        )
        ax = fig.subplots()
        ax.hlines(y, ends[0], ends[1], color=BASELINE, linewidth=1.5)
        for slot, (p, values) in enumerate(zip((low, high), ends)):
            spending = rows[f"spending_p{p}"].iloc[0]
            ax.plot(
                values,
                y,
                "o",
                color=SERIES[slot],
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1,
                label=f"{p}th percentile of total spending (${spending / 1000:,.0f}k a year)",
                clip_on=False,
            )
        ax.axvline(1, color=INK_SECONDARY, linewidth=1)
        ax.set_xlim(max(np.nanmin(ends) - 0.15, -1.5), min(np.nanmax(ends) + 0.15, 3.5))
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title("Elasticity with respect to total spending, for a household with a small or a large budget")
        ax.set_xlabel("below 1: necessity   ·   above 1: luxury")
        ax.set_yticks(y, [label(c) for c in rows.index])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.11), ncol=1)
        return _save(fig, path)


ELASTICITY_TREND_PANELS = [
    "groceries",
    "housing",
    "healthcare",
    "food_away",
    "transportation",
    "entertainment",
    "cash_contributions",
    "insurance_pensions",
]


def plot_elasticity_trends(
    by_year: dict[int, pd.DataFrame], path: Path, categories: list[str] = ELASTICITY_TREND_PANELS
) -> Path:
    """Expenditure elasticities year by year with 95% confidence intervals, one small panel per category."""
    years = sorted(by_year)
    with mpl.rc_context(STYLE):
        title = f"Expenditure elasticities, {years[0]}–{years[-1]} (bars: 95% confidence intervals)"
        fig = _figure(10, 4.6, title)
        axes = fig.subplots(2, (len(categories) + 1) // 2, sharex=True)
        for ax, category in zip(axes.flat, categories):
            rows = pd.DataFrame([by_year[year].loc[category] for year in years], index=years)
            estimate = rows["expenditure_elasticity"].astype(float)
            ax.vlines(
                years, rows["ci_low"].astype(float), rows["ci_high"].astype(float), color=SERIES[0], linewidth=1.5
            )
            ax.plot(years, estimate, "o", color=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1)
            # The same height of axis in every panel, so that the bars can be compared across panels.
            center = (rows["ci_low"].astype(float).min() + rows["ci_high"].astype(float).max()) / 2
            ax.set_ylim(center - 0.45, center + 0.45)
            if center - 0.45 < 1 < center + 0.45:
                ax.axhline(1, color=INK_SECONDARY, linewidth=1)
            ax.set_title(label(category))
            ax.set_xticks(years, [f"'{year % 100:02d}" for year in years])
            ax.grid(axis="x", visible=False)
            ax.margins(x=0.1)
        for ax in axes.flat[len(categories) :]:
            ax.set_visible(False)
        return _save(fig, path)


GROUPING_LABELS = {
    "income_quintile": "Income quintile",
    "age_group": "Age of reference person",
    "tenure": "Housing tenure",
    "education": "Education of reference person",
}
GROUP_LABELS = {
    "lowest": "Lowest",
    "second": "Second",
    "third": "Third",
    "fourth": "Fourth",
    "highest": "Highest",
    "under_25": "Under 25",
    "25_34": "25–34",
    "35_44": "35–44",
    "45_54": "45–54",
    "55_64": "55–64",
    "65_74": "65–74",
    "75_plus": "75 and over",
    "owner": "Owner",
    "renter": "Renter",
    "other": "Other (no cash rent, student housing)",
    "less_than_hs": "Less than high school",
    "high_school": "High school",
    "some_college": "Some college",
    "bachelors": "Bachelor's degree",
    "graduate": "Graduate degree",
}


def plot_inflation_gaps(cumulative: pd.DataFrame, path: Path, official: float | None = None) -> Path:
    """Each group's cumulative inflation relative to all consumer units, in percentage points."""
    everyone = cumulative[cumulative["grouping"] == "all"].iloc[0]
    first, last = int(everyone["from_year"]), int(everyone["to_year"])
    groups = cumulative[cumulative["grouping"] != "all"]
    groupings = list(dict.fromkeys(groups["grouping"]))
    sizes = [int((groups["grouping"] == grouping).sum()) for grouping in groupings]
    with mpl.rc_context(STYLE):
        title = f"Whose prices rose fastest? Inflation {first}–{last} by household group"
        fig = _figure(7.5, 0.27 * sum(sizes) + 0.55 * len(sizes) + 1.5, title)
        axes = fig.subplots(len(groupings), 1, sharex=True, height_ratios=[size + 1.2 for size in sizes])
        for ax, grouping in zip(np.atleast_1d(axes), groupings):
            rows = groups[groups["grouping"] == grouping].iloc[::-1]
            y = np.arange(len(rows))
            gap, half = 100 * rows["gap"], 100 * Z_95 * rows["gap_se"]
            _dot_and_whisker(ax, y, gap, gap - half, gap + half)
            ax.axvline(0, color=INK_SECONDARY, linewidth=1)
            ax.set_yticks(y, [GROUP_LABELS.get(group, group) for group in rows["group"]])
            ax.set_ylim(-0.7, len(rows) - 0.3)
            ax.tick_params(axis="y", length=0)
            ax.set_title(GROUPING_LABELS.get(grouping, grouping))
        note = f"Percentage points above or below all consumer units ({100 * everyone['inflation']:.1f}%"
        note += f"; published CPI-U {100 * official:.1f}%)" if official is not None else ")"
        np.atleast_1d(axes)[-1].set_xlabel(note + "\nBars: 95% confidence intervals")
        return _save(fig, path)
