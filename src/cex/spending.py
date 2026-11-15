"""Build each interview's spending by category from the item-level records.

The FMLI file ships ready-made totals for the major categories (TOTEXPPQ,
HOUSPQ, ...), and for one year they are the quick way in. Across years they are
not reliable:

* Food at home changes meaning in Apr-Jun 2024, from food bought at grocery
  stores to everything bought there, a jump of about a quarter.
* Education is nearly empty from Apr-Jun 2021 through Jan-Mar 2023.
* They leave out items that BLS's published categories include (laundry and
  dry cleaning, jewelry, some household goods), about 1% of spending.

So this package adds up the item-level records itself, using BLS's category
tree for each year (cex.stubs). `compare_with_fmli` reports how the result
lines up with the ready-made variables, quarter by quarter.

Two measures are built for every category, as in the rest of the package:

    <name>_annual   spending over the interview's 3 recall months, times 4
    <name>_in_year  the part of it that falls in the calendar year (for annual means)
"""

import numpy as np
import pandas as pd

from cex import stubs
from cex.survey import WEIGHT_COLUMNS, estimate_and_se


def interview_spending(records: pd.DataFrame, tree: pd.DataFrame, newids: pd.Series, year: int) -> pd.DataFrame:
    """Spending per interview for the total, the major categories, and the detail categories.

    `records` are item-level records (load.load_ucc_records), `tree` the year's
    category tree (stubs.expenditure_tree), `newids` the interviews to cover.
    Interviews with no record in a category get 0. The result is indexed by NEWID.
    """
    items = tree[tree["is_item"]].set_index("ucc")
    records = records[records["ucc"].isin(items.index)]
    unknown = set(records["newid"]) - set(newids)
    if unknown:
        raise ValueError(
            f"{len(unknown):,} interviews have spending records but no FMLI row, e.g. {sorted(unknown)[:3]}"
        )
    files = records.groupby("ucc")["file"].nunique()
    if (files > 1).any():
        raise ValueError(f"UCCs found in both MTBI and ITBI (would be counted twice): {sorted(files.index[files > 1])}")

    newids = pd.Index(newids, name="newid")
    cost = records["cost"] * records["ucc"].map(items["factor"])
    in_year = records["ref_year"] == year
    columns = {}
    for categories in (stubs.major_categories(tree), stubs.detail_categories(tree)):
        category = records["ucc"].map(categories)
        whole = cost.groupby([records["newid"], category]).sum().unstack(fill_value=0.0)
        part = cost[in_year].groupby([records.loc[in_year, "newid"], category[in_year]]).sum().unstack(fill_value=0.0)
        for name in dict.fromkeys(categories):
            for suffix, table, scale in (("annual", whole, 4.0), ("in_year", part, 1.0)):
                values = table[name] if name in table else pd.Series(0.0, index=newids)
                columns[f"{name}_{suffix}"] = scale * values.reindex(newids, fill_value=0.0)
    out = pd.DataFrame(columns, index=newids)
    for suffix in ("in_year", "annual"):
        out.insert(0, f"total_{suffix}", out[[f"{name}_{suffix}" for name in stubs.MAJOR]].sum(axis=1))
    return out


def compare_with_fmli(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuilt spending next to BLS's ready-made FMLI variables, by quarterly file and category.

    Per 3-month recall period: the mean of each, their ratio, and the share of
    interviews where the two agree to the dollar. Uses the `fmli_<name>` columns
    that clean_fmli keeps for this purpose.
    """
    tables = []
    for name in ["total", *stubs.MAJOR]:
        rebuilt, fmli = df[f"{name}_annual"] / 4, df[f"fmli_{name}"]
        both = pd.DataFrame({"fmli": fmli, "rebuilt": rebuilt, "share_same": (rebuilt - fmli).abs() <= 1})
        table = both.groupby(df["source_file"], sort=False).mean()
        table.insert(0, "category", name)
        tables.append(table.reset_index())
    out = pd.concat(tables, ignore_index=True)
    out["ratio"] = out["rebuilt"] / out["fmli"]
    return out[["source_file", "category", "fmli", "rebuilt", "ratio", "share_same"]]


def check_coverage(comparison: pd.DataFrame, tolerance: float = 0.10, min_dollars: float = 25.0) -> None:
    """Raise if rebuilt spending falls well short of the FMLI variable in any category and quarter.

    That is the sign of spending recorded under a UCC that the stub file does
    not list, as happened with groceries. (Rebuilt spending above FMLI's is
    expected: see the module docstring.)
    """
    shortfall = comparison["fmli"] - comparison["rebuilt"]
    bad = comparison[(shortfall > tolerance * comparison["fmli"].abs()) & (shortfall > min_dollars)]
    if len(bad):
        raise ValueError(
            "Spending rebuilt from the item-level records is well below BLS's summary variable for:\n"
            f"{bad.to_string(index=False)}\n"
            "A UCC is probably missing from the year's stub file; see cex.stubs."
        )


def tree_means(records: pd.DataFrame, tree: pd.DataFrame, interviews: pd.DataFrame, year: int) -> pd.DataFrame:
    """Average annual spending per consumer unit for every row of the category tree, items included.

    The Interview-only counterpart of BLS's detailed published table. Columns:
    `level` (1 for a major category), `title`, `ucc` (blank for groups), `mean`
    and `share` of total spending, and their standard errors.
    """
    items = tree[tree["is_item"]].reset_index(drop=True)
    position = pd.Series(np.arange(len(items)), index=items["ucc"])
    records = records[records["ucc"].isin(position.index) & (records["ref_year"] == year)]
    interviews = interviews.set_index("newid")
    weights = interviews[WEIGHT_COLUMNS].to_numpy()
    row = interviews.index.get_indexer(records["newid"])
    ucc = position[records["ucc"]].to_numpy()
    cost = records["cost"].to_numpy() * items["factor"].to_numpy()[ucc]
    # Weighted total for each UCC under each of the 45 weights.
    by_ucc = np.column_stack(
        [np.bincount(ucc, weights=cost * weights[row, j], minlength=len(items)) for j in range(weights.shape[1])]
    )
    population = (weights * (interviews["months_in_scope"].to_numpy() / 12)[:, None]).sum(axis=0)

    # A row's total is the sum over the UCCs at or below it in the tree.
    paths = list(tree["path"])
    index = {path: i for i, path in enumerate(paths)}
    totals = np.zeros((len(tree), weights.shape[1]))
    for path, sums in zip(items["path"], by_ucc):
        for depth in range(1, len(path) + 1):
            totals[index[path[:depth]]] += sums
    # Draws of shape (1 imputation, weight sets, tree rows, [mean, share]).
    draws = np.stack([totals / population, totals / by_ucc.sum(axis=0)], axis=-1).transpose(1, 0, 2)[None]
    estimate, se = estimate_and_se(draws)
    return pd.DataFrame(
        {
            "level": [len(path) for path in paths],
            "title": tree["title"].to_numpy(),
            "ucc": tree["ucc"].fillna("").to_numpy(),
            "mean": estimate[:, 0],
            "mean_se": se[:, 0],
            "share": estimate[:, 1],
            "share_se": se[:, 1],
        }
    )
