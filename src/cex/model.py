"""Engel curves: how spending on a category responds to household income.

For each category we fit a weighted log-log regression

    log(annual spending) = a + b * log(income) + c * log(family size) + controls

on households with positive spending in that category. The coefficient b is the
income elasticity: b < 1 marks a necessity (budget share falls as income rises),
b > 1 a luxury. Point estimates use the CE final weight (FINLWT21); standard
errors come from re-estimating with each of the 44 replicate weights, which
accounts for the survey's complex sample design.
"""

import numpy as np
import pandas as pd

from cex.summarize import N_REPLICATES, replicate_se

CONTROLS = ["age_group", "education", "region", "tenure", "urban", "female_ref"]

DEFAULT_CATEGORIES = ["total", "food_home", "food_away", "housing", "transportation", "healthcare", "entertainment", "apparel"]


def model_sample(df: pd.DataFrame, category: str, min_income: float = 1_000) -> pd.DataFrame:
    """Households usable for the `category` regression.

    Very low or negative incomes (business losses, mostly) are dropped because
    their log is undefined or an extreme outlier.
    """
    spend = f"{category}_annual"
    needed = ["income_before_tax", "family_size", spend, *CONTROLS]
    sample = df.dropna(subset=needed)
    return sample[(sample["income_before_tax"] >= min_income) & (sample[spend] > 0)]


def design_matrix(sample: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=sample.index)
    X["const"] = 1.0
    X["log_income"] = np.log(sample["income_before_tax"])
    X["log_family_size"] = np.log(sample["family_size"])
    # One dummy per category level, dropping the first level as the baseline.
    dummies = pd.get_dummies(sample[["age_group", "education", "region", "tenure"]], drop_first=True, dtype=float)
    X = X.join(dummies)
    X["urban"] = sample["urban"].astype(float)
    X["female_ref"] = sample["female_ref"].astype(float)
    # Drop dummies for levels that never occur in this sample.
    return X.loc[:, X.abs().sum() > 0]


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sqrt_w = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
    return beta


def fit_engel(df: pd.DataFrame, category: str, min_income: float = 1_000) -> dict:
    sample = model_sample(df, category, min_income)
    X = design_matrix(sample)
    y = np.log(sample[f"{category}_annual"]).to_numpy()
    Xa = X.to_numpy()

    beta = wls(Xa, y, sample["weight"].to_numpy())
    replicates = np.column_stack(
        [wls(Xa, y, sample[f"rep_weight_{i:02d}"].to_numpy()) for i in range(1, N_REPLICATES + 1)]
    )
    se = np.array([replicate_se(b, reps) for b, reps in zip(beta, replicates)])

    w = sample["weight"].to_numpy()
    resid = y - Xa @ beta
    y_bar = np.average(y, weights=w)
    r2 = 1 - np.sum(w * resid**2) / np.sum(w * (y - y_bar) ** 2)

    coefficients = pd.DataFrame({"coef": beta, "se": se, "t": beta / se}, index=X.columns)
    return {"category": category, "coefficients": coefficients, "n": len(sample), "r2": r2}


def elasticity_table(df: pd.DataFrame, categories: list[str] = DEFAULT_CATEGORIES) -> pd.DataFrame:
    rows = []
    for cat in categories:
        fit = fit_engel(df, cat)
        b = fit["coefficients"].loc["log_income"]
        rows.append(
            {
                "category": cat,
                "income_elasticity": b["coef"],
                "se": b["se"],
                "ci_low": b["coef"] - 1.96 * b["se"],
                "ci_high": b["coef"] + 1.96 * b["se"],
                "n": fit["n"],
                "share_positive": fit["n"] / len(model_sample(df, "total")),
                "r2": fit["r2"],
            }
        )
    return pd.DataFrame(rows).set_index("category")
