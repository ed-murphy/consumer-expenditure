"""Synthetic stand-in for a BLS intrvwYY.zip so the pipeline can be tested without real data."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from cex.load import COLUMNS, EXPENDITURE_CODES, REPLICATE_WEIGHTS

YEAR = 2024
TRUE_FOOD_ELASTICITY = 0.3
# (file stem, interview year, interview months)
FILES = [
    ("fmli241x", 2024, [1, 2, 3]),
    ("fmli242", 2024, [4, 5, 6]),
    ("fmli243", 2024, [7, 8, 9]),
    ("fmli244", 2024, [10, 11, 12]),
    ("fmli251", 2025, [1, 2, 3]),
]


def make_fmli(rng: np.random.Generator, interview_year: int, months: list[int], n: int, start_id: int) -> pd.DataFrame:
    df = pd.DataFrame({col: 0.0 for col in COLUMNS}, index=range(n))
    df["NEWID"] = (start_id + np.arange(n)) * 10 + rng.integers(1, 5, n)
    df["QINTRVYR"] = interview_year
    df["QINTRVMO"] = rng.choice(months, n)
    df["FINLWT21"] = rng.uniform(5_000, 15_000, n)
    for col in REPLICATE_WEIGHTS:
        df[col] = df["FINLWT21"] * rng.uniform(0.8, 1.2, n)

    income = np.exp(rng.normal(11, 0.8, n))
    df["FINCBTXM"] = income
    df["AGE_REF"] = rng.integers(18, 90, n)
    df["SEX_REF"] = rng.integers(1, 3, n)
    df["EDUC_REF"] = rng.choice([10, 11, 12, 13, 14, 15, 16, 17], n)
    df["FAM_SIZE"] = rng.integers(1, 7, n)
    df["PERSLT18"] = rng.integers(0, 3, n)
    df["PERSOT64"] = rng.integers(0, 2, n)
    df["REGION"] = rng.integers(1, 5, n)
    df["BLS_URBN"] = rng.integers(1, 3, n)
    df["CUTENURE"] = rng.integers(1, 7, n)

    # Quarterly spending; food at home follows a known income elasticity.
    quarterly = {code: np.exp(rng.normal(7, 0.5, n)) for code in EXPENDITURE_CODES}
    quarterly["FDHOME"] = np.exp(
        2.5 + TRUE_FOOD_ELASTICITY * np.log(income) + 0.4 * np.log(df["FAM_SIZE"]) + rng.normal(0, 0.3, n)
    )
    for code, total in quarterly.items():
        cq_share = (df["QINTRVMO"] - 1) % 3 / 3  # months of the recall period in the interview's quarter
        df[f"{code}CQ"] = total * cq_share
        df[f"{code}PQ"] = total * (1 - cq_share)
    return df


@pytest.fixture(scope="session")
def synthetic_zip(tmp_path_factory) -> "Path":
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("raw") / f"intrvw{YEAR % 100}.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for i, (stem, year, months) in enumerate(FILES):
            df = make_fmli(rng, year, months, n=2_000, start_id=100_000 * (i + 1))
            if stem != "fmli241x":
                # Like the real 2024 release: from 2024 Q2 on, food at home moves to GROCER*
                # and the old FDHOME* columns remain but are all zero.
                df["GROCERPQ"], df["GROCERCQ"] = df["FDHOMEPQ"], df["FDHOMECQ"]
                df["FDHOMEPQ"], df["FDHOMECQ"] = 0.0, 0.0
            if stem == "fmli242":
                df.columns = df.columns.str.lower()  # some releases use lowercase headers
                df.loc[:5, "region"] = np.nan  # written as "." below, like the real files
            zf.writestr(f"intrvw{YEAR % 100}/{stem}.csv", df.to_csv(index=False, na_rep="."))
    return path
