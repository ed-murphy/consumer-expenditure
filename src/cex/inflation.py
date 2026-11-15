"""Inflation as different kinds of households experienced it.

The CPI's weights come from this survey: the index averages price changes
across categories using the average household's spending. A group whose budget
differs from the average, say with more rent and gasoline and less recreation,
faces a different average price change. This module measures that for groups
of consumer units by repricing each group's own basket:

    inflation of group g from year t to t+1 = sum over categories of
        (g's share of its basket spent on the category in year t) x (the category's price change)

That is a Laspeyres index: last year's basket at this year's prices. The price
change of a category is the same national CPI series for every group (see
cex.prices), so differences between groups come only from what they buy, not
from where they shop or which variety they choose.

Two things make the basket differ from "total spending":

* It leaves out cash contributions, life insurance, and pension and Social
  Security contributions, which have no price index (the CPI leaves them out too).
* Homeowners' shelter is their home's rental value, as in the CPI, not what
  they pay out (mortgage interest, property tax, upkeep). The interview asks
  owners what their home would rent for; that amount, priced with the CPI for
  owners' equivalent rent, replaces the out-of-pocket costs. This matters most
  for older owners with no mortgage, whose out-of-pocket costs are low.

Standard errors reflect sampling error in the baskets (replicate weights, and
income imputations for the income groups); the price indexes are taken as given.
"""

import numpy as np
import pandas as pd

from cex.clean import income_imputations
from cex.prices import ALL_ITEMS, CPI_SERIES, annual_average, category_price_change
from cex.survey import WEIGHT_COLUMNS, estimate_and_se

# The categories in the basket: every detail category that has a price index.
PRICED = list(CPI_SERIES)
OWNER_SHELTER = "owned_dwellings"

GROUPINGS = ["income_quintile", "age_group", "tenure", "education"]
ALL = "all"


def basket_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Each interview's in-year spending on the priced categories, with owners' shelter at its rental value."""
    basket = df[[f"{category}_in_year" for category in PRICED]].copy()
    basket.columns = PRICED
    basket[OWNER_SHELTER] = df["rent_equivalent"].fillna(0.0) * df["months_in_scope"]
    return basket


def _draws(df: pd.DataFrame, change: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Inflation of each group under each weight set: an array of shape (weight sets, groups)."""
    basket = basket_spending(df).to_numpy()
    weights = df[WEIGHT_COLUMNS].to_numpy()
    out = []
    for mask in masks:
        totals = weights[mask].T @ basket[mask]  # (weight sets, categories)
        shares = totals / totals.sum(axis=1, keepdims=True)
        out.append(shares @ (change - 1))
    return np.column_stack(out)


def group_inflation(df: pd.DataFrame, price_change: pd.Series, groupings: list[str] = GROUPINGS) -> pd.DataFrame:
    """Inflation of each group's basket, and its gap to that of all consumer units.

    `df` is a year's interview table and `price_change` the ratio of each
    category's price index in the later year to the earlier one
    (prices.category_price_change). One row per group, the first for all
    consumer units: `inflation`, `gap` (the group's inflation minus that of
    all consumer units), and their standard errors, as fractions.
    """
    change = price_change[PRICED].to_numpy()
    rows = []
    for grouping in [None, *groupings]:
        levels = [] if grouping is None else [lv for lv in df[grouping].cat.categories if (df[grouping] == lv).any()]
        # Only the income groups depend on which income imputation is used.
        versions = list(income_imputations(df)) if grouping == "income_quintile" else [df]
        draws = []
        for version in versions:
            masks = [np.ones(len(version), dtype=bool)]
            masks += [(version[grouping] == level).to_numpy() for level in levels]
            draws.append(_draws(version, change, masks))
        draws = np.stack(draws)  # (imputations, weight sets, all consumer units + groups)
        inflation, inflation_se = estimate_and_se(draws)
        gap, gap_se = estimate_and_se(draws - draws[:, :, :1])
        for i, level in enumerate([ALL, *levels]):
            if (grouping is None) == (i == 0):
                rows.append(
                    {
                        "grouping": grouping or ALL,
                        "group": level,
                        "inflation": inflation[i],
                        "inflation_se": inflation_se[i],
                        "gap": gap[i],
                        "gap_se": gap_se[i],
                    }
                )
    return pd.DataFrame(rows)


def gap_contributions(df: pd.DataFrame, price_change: pd.Series, by: str) -> pd.DataFrame:
    """Why each group's inflation differs from the average: one row per group and category.

    `share` is the category's share of the group's basket, `share_all` its
    share of everyone's, and `contribution` the part of the group's gap that
    comes from the category: (share - share_all) x (price change relative to
    average inflation). The contributions of a group add up to its gap.
    """
    basket = basket_spending(df).to_numpy()
    weight = df["weight"].to_numpy()
    change = price_change[PRICED].to_numpy() - 1

    def shares(mask: np.ndarray) -> np.ndarray:
        totals = weight[mask] @ basket[mask]
        return totals / totals.sum()

    overall = shares(np.ones(len(df), dtype=bool))
    average = overall @ change
    tables = []
    for level in df[by].cat.categories:
        mask = (df[by] == level).to_numpy()
        if mask.any():
            share = shares(mask)
            tables.append(
                pd.DataFrame(
                    {
                        "grouping": by,
                        "group": level,
                        "category": PRICED,
                        "share": share,
                        "share_all": overall,
                        "price_change": change,
                        "contribution": (share - overall) * (change - average),
                    }
                )
            )
    return pd.concat(tables, ignore_index=True)


def chain(links: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Compound year-to-year inflation into one cumulative figure per group.

    `links` maps a base year to `group_inflation` for that year's basket and the
    price change to the next year. The result covers the first base year to the
    year after the last one. Its standard errors treat the years as independent
    samples (adjacent years share a few interviews) and use a first-order
    approximation.
    """
    years = sorted(links)
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(f"Chaining needs consecutive years, got {years}")
    keys = ["grouping", "group"]
    index = links[years[0]].set_index(keys).index
    log_index = pd.Series(0.0, index=index)
    log_index_all = 0.0
    variance = pd.Series(0.0, index=index)
    gap_variance = pd.Series(0.0, index=index)
    for year in years:
        link = links[year].set_index(keys).reindex(index)
        everyone = link.loc[(ALL, ALL), "inflation"]
        log_index += np.log1p(link["inflation"])
        log_index_all += np.log1p(everyone)
        variance += (link["inflation_se"] / (1 + link["inflation"])) ** 2
        gap_variance += (link["gap_se"] / (1 + everyone)) ** 2
    level = np.exp(log_index)
    out = pd.DataFrame(
        {
            "from_year": years[0],
            "to_year": years[-1] + 1,
            "inflation": level - 1,
            "inflation_se": level * np.sqrt(variance),
            "gap": level - np.exp(log_index_all),
            "gap_se": level * np.sqrt(gap_variance),
        }
    )
    return out.reset_index()


def official_cpi_change(prices: pd.DataFrame, start_year: int, end_year: int) -> float:
    """Change in the published all-items CPI-U between the annual averages of two years, as a fraction."""
    annual = annual_average(prices)[ALL_ITEMS]
    return annual[end_year] / annual[start_year] - 1


def year_link(df: pd.DataFrame, prices: pd.DataFrame, year: int) -> pd.DataFrame:
    """`group_inflation` from `year` to the next, using `year`'s baskets."""
    return group_inflation(df, category_price_change(prices, year, year + 1))
