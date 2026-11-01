"""Turn raw FMLI records into a tidy analysis table with readable names and labels."""

import numpy as np
import pandas as pd

from cex.load import EXPENDITURE_CODES, REPLICATE_WEIGHTS

CATEGORY_NAMES = {
    "TOTEXP": "total",
    "FDHOME": "food_home",
    "FDAWAY": "food_away",
    "HOUS": "housing",
    "TRANS": "transportation",
    "HEALTH": "healthcare",
    "ENTERT": "entertainment",
    "APPAR": "apparel",
}
# "food" is derived as food_home + food_away.
CATEGORIES = ["total", "food", "food_home", "food_away", "housing", "transportation", "healthcare", "entertainment", "apparel"]

REGIONS = {1: "northeast", 2: "midwest", 3: "south", 4: "west"}

EDUCATION = {
    0: "less_than_hs", 10: "less_than_hs", 11: "less_than_hs",
    12: "high_school",
    13: "some_college", 14: "some_college",
    15: "bachelors",
    16: "graduate", 17: "graduate",
}
EDUCATION_LEVELS = ["less_than_hs", "high_school", "some_college", "bachelors", "graduate"]

TENURE = {1: "owner", 2: "owner", 3: "owner", 4: "renter", 5: "other", 6: "other"}

AGE_BINS = [0, 25, 35, 45, 55, 65, 75, np.inf]
AGE_LABELS = ["under_25", "25_34", "35_44", "45_54", "55_64", "65_74", "75_plus"]


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


def clean_fmli(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    out["newid"] = raw["NEWID"].astype("int64")
    # NEWID = CU id followed by a 1-digit interview number (the CU is interviewed up to 4 times).
    out["cuid"] = out["newid"] // 10
    out["interview"] = out["newid"] % 10
    out["source_file"] = raw["SOURCE_FILE"]
    out["interview_year"] = raw["QINTRVYR"].astype(int)
    out["interview_month"] = raw["QINTRVMO"].astype(int)
    out["months_in_scope"] = months_in_scope(out["interview_year"], out["interview_month"], year)

    out["weight"] = raw["FINLWT21"]
    # A blank replicate weight means the CU is excluded from that replicate.
    for i, col in enumerate(REPLICATE_WEIGHTS, start=1):
        out[f"rep_weight_{i:02d}"] = raw[col].fillna(0.0)

    first = (out["interview_year"] == year) & (out["interview_month"] <= 3)
    last = out["interview_year"] == year + 1
    for code, name in CATEGORY_NAMES.items():
        pq = raw[f"{code}PQ"].fillna(0.0)
        cq = raw[f"{code}CQ"].fillna(0.0)
        # Spending over the full 3-month recall period, scaled to an annual rate.
        out[f"{name}_annual"] = 4 * (pq + cq)
        # Only the part of the recall period inside `year`, used for BLS-style annual means.
        out[f"{name}_in_year"] = np.where(first, cq, np.where(last, pq, pq + cq))
    for suffix in ("annual", "in_year"):
        out[f"food_{suffix}"] = out[f"food_home_{suffix}"] + out[f"food_away_{suffix}"]

    # Annual income before taxes over the past 12 months, with BLS imputation for nonresponse.
    # (Recent releases no longer include an after-tax income variable.)
    out["income_before_tax"] = raw["FINCBTXM"]

    out["age_ref"] = raw["AGE_REF"]
    out["age_group"] = pd.cut(raw["AGE_REF"], AGE_BINS, labels=AGE_LABELS, right=False)
    out["female_ref"] = (raw["SEX_REF"] == 2).astype(int)
    out["education"] = pd.Categorical(raw["EDUC_REF"].map(EDUCATION), categories=EDUCATION_LEVELS)
    out["family_size"] = raw["FAM_SIZE"]
    out["children"] = raw["PERSLT18"]
    out["seniors"] = raw["PERSOT64"]
    out["region"] = pd.Categorical(raw["REGION"].map(REGIONS), categories=list(REGIONS.values()))
    out["urban"] = (raw["BLS_URBN"] == 1).astype(int)
    out["tenure"] = pd.Categorical(raw["CUTENURE"].map(TENURE), categories=["owner", "renter", "other"])

    return out
