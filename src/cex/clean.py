"""Turn raw FMLI records into a tidy interview table with readable names and labels.

`clean_fmli` gives one row per interview with identifiers, timing, weights,
income, and household traits. Spending by category is built separately from
the item-level records (cex.spending) and joined on with `add_spending`.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from cex import stubs
from cex.load import INCOME_IMPUTATIONS, INCOME_RANKS, REPLICATE_WEIGHTS
from cex.survey import N_IMPUTATIONS, REPLICATE_WEIGHT_COLUMNS, WEIGHT_COLUMNS

# The 15 major components of total spending. They add up to it exactly.
COMPONENTS = stubs.MAJOR
CATEGORIES = ["total", *COMPONENTS]

# BLS's ready-made FMLI summary variable for each category, kept as `fmli_<name>`
# for cross-checking only (see cex.spending). FDHOME is not quite "groceries":
# through Jan-Mar 2024 it is food only.
FMLI_SUMMARY_CODES = {
    "TOTEXP": "total",
    "FDHOME": "groceries",
    "FDAWAY": "food_away",
    "ALCBEV": "alcohol",
    "HOUS": "housing",
    "APPAR": "apparel",
    "TRANS": "transportation",
    "HEALTH": "healthcare",
    "ENTERT": "entertainment",
    "PERSCA": "personal_care",
    "READ": "reading",
    "EDUCA": "education",
    "TOBACC": "tobacco",
    "MISC": "miscellaneous",
    "CASHCO": "cash_contributions",
    "PERINS": "insurance_pensions",
}
assert list(FMLI_SUMMARY_CODES.values()) == CATEGORIES

REGIONS = {1: "northeast", 2: "midwest", 3: "south", 4: "west"}

EDUCATION = {
    0: "less_than_hs",
    10: "less_than_hs",
    11: "less_than_hs",
    12: "high_school",
    13: "some_college",
    14: "some_college",
    15: "bachelors",
    16: "graduate",
    17: "graduate",
}
EDUCATION_LEVELS = ["less_than_hs", "high_school", "some_college", "bachelors", "graduate"]

TENURE = {1: "owner", 2: "owner", 3: "owner", 4: "renter", 5: "other", 6: "other"}

AGE_BINS = [0, 25, 35, 45, 55, 65, 75, np.inf]
AGE_LABELS = ["under_25", "25_34", "35_44", "45_54", "55_64", "65_74", "75_plus"]

INCOME_QUINTILES = ["lowest", "second", "third", "fourth", "highest"]

# Characteristics of the household (as opposed to its spending, income, or weights).
HOUSEHOLD_TRAITS = [
    "age_ref",
    "age_group",
    "female_ref",
    "education",
    "family_size",
    "children",
    "seniors",
    "earners",
    "vehicles",
    "region",
    "urban",
    "tenure",
]


def months_in_scope(interview_year: pd.Series, interview_month: pd.Series, year: int) -> pd.Series:
    """How many of the 3 recall months fall inside calendar `year`.

    An interview covers the 3 months before the interview month. Interviews in
    Jan-Mar of `year` reach back into the prior year; interviews in Jan-Mar of
    `year + 1` reach back into `year`.
    """
    scope = pd.Series(3, index=interview_year.index)
    first = (interview_year == year) & (interview_month <= 3)
    last = interview_year == year + 1
    scope[first] = interview_month[first] - 1
    scope[last] = 4 - interview_month[last]
    return scope


def income_quintile(rank: pd.Series) -> pd.Series:
    """Income quintile from a 0-1 weighted income rank (BLS's INC_RNK variables)."""
    return pd.cut(rank, [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], labels=INCOME_QUINTILES)


def clean_fmli(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}

    cols["newid"] = raw["NEWID"].astype("int64")
    # NEWID = CU id followed by a 1-digit interview number (the CU is interviewed up to 4 times).
    cols["cuid"] = cols["newid"] // 10
    cols["interview"] = cols["newid"] % 10
    cols["source_file"] = raw["SOURCE_FILE"]
    cols["interview_year"] = raw["QINTRVYR"].astype(int)
    cols["interview_month"] = raw["QINTRVMO"].astype(int)
    cols["months_in_scope"] = months_in_scope(cols["interview_year"], cols["interview_month"], year)

    cols["weight"] = raw["FINLWT21"]
    # A blank replicate weight means the CU is excluded from that replicate.
    for raw_col, col in zip(REPLICATE_WEIGHTS, REPLICATE_WEIGHT_COLUMNS):
        cols[col] = raw[raw_col].fillna(0.0)

    # Annual income before taxes over the past 12 months. BLS imputes it for
    # nonresponse five times; FINCBTXM is the mean of the five. Estimates that use
    # income should loop over the imputations (see income_imputations).
    # (Recent releases no longer include an after-tax income variable.)
    cols["income_before_tax"] = raw["FINCBTXM"]
    cols["income_rank"] = raw["INC_RNKM"]
    for m, (income, rank) in enumerate(zip(INCOME_IMPUTATIONS, INCOME_RANKS), start=1):
        cols[f"income_before_tax_{m}"] = raw[income]
        cols[f"income_rank_{m}"] = raw[rank]
    cols["income_quintile"] = income_quintile(raw["INC_RNKM"])

    cols["age_ref"] = raw["AGE_REF"]
    cols["age_group"] = pd.cut(raw["AGE_REF"], AGE_BINS, labels=AGE_LABELS, right=False)
    cols["female_ref"] = (raw["SEX_REF"] == 2).astype(int)
    cols["education"] = pd.Categorical(raw["EDUC_REF"].map(EDUCATION), categories=EDUCATION_LEVELS)
    cols["family_size"] = raw["FAM_SIZE"]
    cols["children"] = raw["PERSLT18"]
    cols["seniors"] = raw["PERSOT64"]
    cols["earners"] = raw["NO_EARNR"]
    cols["vehicles"] = raw["VEHQ"]  # owned vehicles
    cols["region"] = pd.Categorical(raw["REGION"].map(REGIONS), categories=list(REGIONS.values()))
    cols["urban"] = (raw["BLS_URBN"] == 1).astype(int)
    cols["tenure"] = pd.Categorical(raw["CUTENURE"].map(TENURE), categories=["owner", "renter", "other"])
    # What owners think their home would rent for per month (blank for everyone else). Not spending:
    # it stands in for owners' shelter costs when a price index is built (see cex.inflation).
    cols["rent_equivalent"] = raw["RENTEQVX"]

    # BLS's summary variables over the 3-month recall period: previous quarter + current quarter.
    for code, name in FMLI_SUMMARY_CODES.items():
        cols[f"fmli_{name}"] = raw[f"{code}PQ"].fillna(0.0) + raw[f"{code}CQ"].fillna(0.0)

    return pd.DataFrame(cols, index=raw.index)


def add_spending(interviews: pd.DataFrame, spending: pd.DataFrame) -> pd.DataFrame:
    """Join spending by category (spending.interview_spending, indexed by NEWID) onto the interview table."""
    return interviews.join(spending, on="newid", validate="one_to_one")


def spending_columns(df: pd.DataFrame, suffix: str = "annual") -> list[str]:
    """Names of the categories that have a `<name>_<suffix>` column in `df`."""
    return [col.removesuffix(f"_{suffix}") for col in df if col.endswith(f"_{suffix}")]


def has_imputations(df: pd.DataFrame) -> bool:
    return all(f"income_before_tax_{m}" in df for m in range(1, N_IMPUTATIONS + 1))


def use_imputation(df: pd.DataFrame, m: int) -> pd.DataFrame:
    """`df` with income, and its rank and quintile when present, taken from BLS imputation `m` (1-5)."""
    out = df.copy()
    out["income_before_tax"] = df[f"income_before_tax_{m}"]
    if f"income_rank_{m}" in df:
        out["income_rank"] = df[f"income_rank_{m}"]
        out["income_quintile"] = income_quintile(df[f"income_rank_{m}"])
    return out


def income_imputations(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield `df` once per income imputation, or just `df` if it has no imputation columns."""
    if not has_imputations(df):
        yield df
        return
    for m in range(1, N_IMPUTATIONS + 1):
        yield use_imputation(df, m)


def household_year(df: pd.DataFrame) -> pd.DataFrame:
    """One row per consumer unit, combining its interviews.

    A single quarter is a noisy measure of a household's spending: a car or a
    sofa bought that quarter shows up as a huge outlay, and a quarter without
    such a purchase as zero. Averaging a CU's annualized spending across its
    (up to four) interviews smooths much of that out.

    Each interview's weight stands for a quarter of a population-year, so a
    CU's weights (full-sample and replicates) are the sums of its interview
    weights divided by 4. (A year's data hold five quarterly files, so these sum
    to about 1.25 times the population; only relative weights matter for the
    models.) Income is averaged across interviews; household traits come from
    the latest interview.
    """
    df = df.sort_values(["cuid", "interview_year", "interview_month"])
    by_cu = df.groupby("cuid", sort=True)
    averaged = [c for c in df if c.endswith("_annual") or c.startswith("income_before_tax")]
    latest = df.drop_duplicates("cuid", keep="last").set_index("cuid")[HOUSEHOLD_TRAITS]
    return pd.concat(
        [
            by_cu[averaged].mean(),
            by_cu[WEIGHT_COLUMNS].sum() / 4,
            by_cu.size().rename("interviews"),
            latest,
        ],
        axis=1,
    )
