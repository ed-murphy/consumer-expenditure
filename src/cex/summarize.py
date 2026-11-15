"""BLS-style weighted annual means and budget shares, overall and by group.

This mirrors how BLS builds its published "average annual expenditures" tables
from the Interview data, which makes it a good check that the data loaded correctly.
Published means will be close but not identical: BLS tables take some categories
(notably food and apparel) from the Diary survey instead.

Budget shares are ratios of means (category mean / total spending mean), the
way BLS reports "percent of total". Standard errors use the replicate weights
and, because income appears in the tables (and defines income quintiles), are
combined across BLS's five income imputations (see cex.survey).
"""

import numpy as np
import pandas as pd

from cex.clean import CATEGORIES, income_imputations
from cex.cpi import constant_dollar_factor
from cex.survey import WEIGHT_COLUMNS, estimate_and_se

NON_SPENDING_ITEMS = ["consumer_units", "income_before_tax"]

# Household characteristics BLS also tabulates means by.
GROUPINGS = ["income_quintile", "age_group", "region", "education", "tenure"]


def _means_and_shares(df: pd.DataFrame, categories: list[str]) -> np.ndarray:
    """Means and budget shares for every weight set, as an array of shape (1 + replicates, 2, items)."""
    weights = df[WEIGHT_COLUMNS].to_numpy()
    # Each interview represents its share of the year: months_in_scope / 12.
    population_weights = weights * (df["months_in_scope"].to_numpy() / 12)[:, None]
    population = population_weights.sum(axis=0)
    income = df["income_before_tax"].to_numpy() @ population_weights / population
    spending = df[[f"{c}_in_year" for c in categories]].to_numpy()
    means = spending.T @ weights / population  # (categories, weight sets)
    shares = means / (df["total_in_year"].to_numpy() @ weights / population)
    no_share = np.full((2, len(population)), np.nan)
    return np.stack([np.vstack([population, income, means]), np.vstack([no_share, shares])], axis=0).transpose(2, 0, 1)


def _table(estimate: np.ndarray, se: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({"mean": estimate[0], "mean_se": se[0], "share": estimate[1], "share_se": se[1]}, index=index)


def annual_means(df: pd.DataFrame, categories: list[str] = CATEGORIES) -> pd.DataFrame:
    """Average annual income and spending per consumer unit, budget shares, and standard errors."""
    draws = np.stack([_means_and_shares(d, categories) for d in income_imputations(df)])
    return _table(*estimate_and_se(draws), pd.Index([*NON_SPENDING_ITEMS, *categories], name="item"))


def means_by(df: pd.DataFrame, by: str, categories: list[str] = CATEGORIES) -> pd.DataFrame:
    """`annual_means` for each level of the column `by`, indexed by (level, item)."""
    values = df[by]
    if isinstance(values.dtype, pd.CategoricalDtype):
        levels = [level for level in values.cat.categories if (values == level).any()]
    else:
        levels = sorted(values.dropna().unique())
    draws = np.stack(
        [
            np.concatenate([_means_and_shares(d[d[by] == level], categories) for level in levels], axis=2)
            for d in income_imputations(df)
        ]
    )
    index = pd.MultiIndex.from_product([levels, [*NON_SPENDING_ITEMS, *categories]], names=["group", "item"])
    return _table(*estimate_and_se(draws), index)


def group_means(df: pd.DataFrame, groupings: list[str] = GROUPINGS, categories: list[str] = CATEGORIES) -> pd.DataFrame:
    """`means_by` for several groupings, stacked into one long table."""
    tables = [means_by(df, by, categories).reset_index().assign(grouping=by) for by in groupings]
    out = pd.concat(tables, ignore_index=True)
    return out[["grouping", "group", "item", "mean", "mean_se", "share", "share_se"]]


def trends(means_by_year: dict[int, pd.DataFrame], base_year: int | None = None) -> pd.DataFrame:
    """Stack `annual_means` tables across years, adding dollar values in constant `base_year` dollars.

    `base_year` defaults to the latest year. Values are deflated with the CPI-U.
    """
    base_year = max(means_by_year) if base_year is None else base_year
    frames = []
    for year in sorted(means_by_year):
        table = means_by_year[year].reset_index()
        factor = np.where(table["item"] == "consumer_units", 1.0, constant_dollar_factor(year, base_year))
        table["real_mean"] = table["mean"] * factor
        table["real_mean_se"] = table["mean_se"] * factor
        table["year"] = year
        frames.append(table)
    out = pd.concat(frames, ignore_index=True)
    return out[["year", "item", "mean", "mean_se", "real_mean", "real_mean_se", "share", "share_se"]]
