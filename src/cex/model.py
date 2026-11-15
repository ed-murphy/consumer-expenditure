"""Engel curves: how spending on each category varies with a household's resources.

Both models are fit on household-year data (each consumer unit's interviews
averaged; see clean.household_year), on one common sample, with the same
controls: log family size, age, education, region, housing tenure, urban, sex
of the reference person, and the numbers of children and seniors.

Expenditure elasticities (the main result)
------------------------------------------
A Working-Leser budget share model,

    share_c = a_c + b_c * log(total spending) + controls,

estimated by two-stage least squares with log income as the instrument for log
total spending. The elasticity of spending on c with respect to total spending,
at the mean budget share, is 1 + b_c / mean share_c. Below 1 marks a necessity
(its share of the budget falls as households spend more), above 1 a luxury.
The shares add up to 1, so the share-weighted elasticities add up to exactly 1,
which is what makes 1 the dividing line. Households that spent nothing on c
stay in the sample with a share of 0.

Why instrument: measured total spending shares its noise with its components
(a car bought this quarter raises both transportation and the total), which
biases ordinary least squares upward for lumpy categories. Income moves
long-run spending but not that noise.

Income elasticities
-------------------
Poisson pseudo-maximum likelihood (PPML),

    E[spending_c] = exp(a_c + e_c * log(income) + controls),

where e_c is the elasticity of spending with respect to current income. Unlike
a log-log regression it keeps households that spent nothing on c. Spending
responds much less to one year's income than to total spending (households
smooth temporary income swings, and reported income is noisy), so these are
small across the board and 1 is NOT the necessity/luxury line: compare each one
with the income elasticity of total spending instead.

Curvature
---------
`fit_quadratic` adds the square of log total spending to the budget share
model, letting a category be a luxury for small budgets and a necessity for
large ones (or the reverse).

Uncertainty
-----------
Point estimates use the CE final weight. Standard errors come from re-estimating
with each of the 44 replicate weights, combined across BLS's five income
imputations with Rubin's rules (see cex.survey).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cex.clean import COMPONENTS, has_imputations
from cex.survey import N_IMPUTATIONS, WEIGHT_COLUMNS, Z_95, estimate_and_se

CATEGORICAL_CONTROLS = ["age_group", "education", "region", "tenure"]
NUMERIC_CONTROLS = ["urban", "female_ref", "children", "seniors"]
CONTROLS = [*CATEGORICAL_CONTROLS, *NUMERIC_CONTROLS]

DEFAULT_CATEGORIES = ["total", *COMPONENTS]

MIN_INCOME = 1_000


@dataclass
class EngelResults:
    elasticities: pd.DataFrame  # one row per category
    coefficients: pd.DataFrame  # budget share model, indexed by (category, term)
    first_stage: pd.Series  # log income in the regression of log total spending on income and controls
    n: int


def income_columns(df: pd.DataFrame) -> list[str]:
    if has_imputations(df):
        return [f"income_before_tax_{m}" for m in range(1, N_IMPUTATIONS + 1)]
    return ["income_before_tax"]


def model_sample(households: pd.DataFrame, min_income: float = MIN_INCOME) -> pd.DataFrame:
    """Households usable for the models.

    Requires complete controls, positive total spending, and income of at least
    `min_income` in every imputation: very low or negative incomes (business
    losses, mostly) make log income undefined or an extreme outlier. The same
    households are used for every category and imputation, which keeps the
    estimates comparable and makes the expenditure elasticities add up.
    """
    incomes = income_columns(households)
    sample = households.dropna(subset=["family_size", "total_annual", *CONTROLS, *incomes])
    keep = (sample[incomes] >= min_income).all(axis=1) & (sample["total_annual"] > 0)
    return sample[keep]


def controls_matrix(sample: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=sample.index)
    X["const"] = 1.0
    X["log_family_size"] = np.log(sample["family_size"])
    # One dummy per category level, dropping the first level present as the baseline.
    categorical = sample[CATEGORICAL_CONTROLS].apply(lambda s: s.cat.remove_unused_categories())
    X = X.join(pd.get_dummies(categorical, drop_first=True, dtype=float))
    for col in NUMERIC_CONTROLS:
        X[col] = sample[col].astype(float)
    # Drop controls that never vary from zero in this sample.
    return X.loc[:, X.abs().sum() > 0]


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares. `y` may be a vector or a matrix with one outcome per column."""
    sqrt_w = np.sqrt(w)
    yw = y * (sqrt_w[:, None] if y.ndim == 2 else sqrt_w)
    beta, *_ = np.linalg.lstsq(X * sqrt_w[:, None], yw, rcond=None)
    return beta


def ppml(X: np.ndarray, y: np.ndarray, w: np.ndarray, start: np.ndarray | None = None, tol: float = 1e-9) -> np.ndarray:
    """Weighted Poisson pseudo-maximum likelihood, by iteratively reweighted least squares.

    Fits E[y | X] = exp(X @ beta) for y >= 0, zeros included. The first column of
    `X` must be the constant. `start` warm-starts the iterations.
    """
    if start is None:
        beta = np.zeros(X.shape[1])
        beta[0] = np.log(np.average(y, weights=w))
    else:
        beta = start.copy()
    for _ in range(100):
        eta = X @ beta
        mu = np.exp(eta)
        Xw = X * (w * mu)[:, None]
        new = np.linalg.solve(X.T @ Xw, Xw.T @ (eta + (y - mu) / mu))
        if np.max(np.abs(new - beta)) < tol:
            return new
        beta = new
    raise RuntimeError("PPML did not converge in 100 iterations")


def _fit_imputation(
    sample: pd.DataFrame, X: np.ndarray, income: str, categories: list[str], starts: list, income_model: bool
) -> dict:
    """Both models for one income imputation, for the full-sample weight and every replicate weight.

    Each array in the result has the weight sets on its first axis. `starts`
    holds the latest full-sample PPML estimates, used as starting values.
    Without `income_model` the PPML fits are skipped and the income
    elasticities are missing.
    """
    log_income = np.log(sample[income].to_numpy())
    total = sample["total_annual"].to_numpy()
    spending = sample[[f"{c}_annual" for c in categories]].to_numpy()
    shares = spending / total[:, None]
    # PPML needs y >= 0. A handful of healthcare values are negative (net reimbursements).
    spending = np.clip(spending, 0, None)

    Z = np.column_stack([log_income, X])  # first stage: instrument + controls
    X_income = np.column_stack([X[:, :1], log_income, X[:, 1:]])  # PPML: const, log income, controls

    draws = {"first_stage": [], "coefficients": [], "mean_share": [], "expenditure": [], "income": []}
    for j, weight in enumerate(WEIGHT_COLUMNS):
        w = sample[weight].to_numpy()
        first_stage = wls(Z, np.log(total), w)
        coefficients = wls(np.column_stack([Z @ first_stage, X]), shares, w)
        mean_share = w @ shares / w.sum()
        income_elasticity = [np.nan] * len(categories)
        for c in range(len(categories) if income_model else 0):
            beta = ppml(X_income, spending[:, c], w, start=starts[c])
            if j == 0:
                starts[c] = beta  # replicate fits start from the full-sample estimate
            income_elasticity[c] = beta[1]
        draws["first_stage"].append(first_stage[0])
        draws["coefficients"].append(coefficients)
        draws["mean_share"].append(mean_share)
        draws["expenditure"].append(1 + coefficients[0] / mean_share)
        draws["income"].append(income_elasticity)
    return {key: np.array(value) for key, value in draws.items()}


def classify(ci_low: float, ci_high: float) -> str:
    """Label an expenditure elasticity by where its 95% confidence interval lies."""
    if ci_high < 0:
        return "inferior"
    if ci_high < 1:
        return "necessity"
    if ci_low > 1:
        return "luxury"
    return "unclear"


def fit_engel(
    households: pd.DataFrame,
    categories: list[str] = DEFAULT_CATEGORIES,
    min_income: float = MIN_INCOME,
    income_model: bool = True,
) -> EngelResults:
    """Fit both Engel models for each category on household-year data.

    `categories` should add up to total spending (the 15 major categories, or
    the detail categories) for the expenditure elasticities to average to 1.
    `income_model=False` skips the income elasticities, which take most of the time.
    """
    sample = model_sample(households, min_income)
    X = controls_matrix(sample)
    starts = [None] * len(categories)
    fits = [
        _fit_imputation(sample, X.to_numpy(), income, categories, starts, income_model)
        for income in income_columns(sample)
    ]
    draws = {key: np.stack([fit[key] for fit in fits]) for key in fits[0]}
    estimates = {key: estimate_and_se(value) for key, value in draws.items()}

    table = pd.DataFrame(index=pd.Index(categories, name="category"))
    table["budget_share"], table["budget_share_se"] = estimates["mean_share"]
    table["expenditure_elasticity"], table["expenditure_elasticity_se"] = estimates["expenditure"]
    if "total" in categories:  # 1 by construction; avoid reporting rounding noise
        table.loc["total", ["expenditure_elasticity", "expenditure_elasticity_se"]] = [1.0, 0.0]
    table["ci_low"] = table["expenditure_elasticity"] - Z_95 * table["expenditure_elasticity_se"]
    table["ci_high"] = table["expenditure_elasticity"] + Z_95 * table["expenditure_elasticity_se"]
    table["classification"] = [
        "" if cat == "total" else classify(lo, hi) for cat, lo, hi in zip(categories, table["ci_low"], table["ci_high"])
    ]
    table["income_elasticity"], table["income_elasticity_se"] = estimates["income"]
    table["income_ci_low"] = table["income_elasticity"] - Z_95 * table["income_elasticity_se"]
    table["income_ci_high"] = table["income_elasticity"] + Z_95 * table["income_elasticity_se"]
    positive = sample[[f"{c}_annual" for c in categories]].to_numpy() > 0
    table["share_positive"] = sample["weight"].to_numpy() @ positive / sample["weight"].sum()
    table["n"] = len(sample)

    coef, se = estimates["coefficients"]
    terms = ["log_total_spending", *X.columns]
    coefficients = pd.DataFrame(
        {"coef": coef.ravel(order="F"), "se": se.ravel(order="F")},
        index=pd.MultiIndex.from_product([categories, terms], names=["category", "term"]),
    )
    coefficients["t"] = coefficients["coef"] / coefficients["se"]

    fs, fs_se = estimates["first_stage"]
    first_stage = pd.Series({"coef": fs, "se": fs_se, "t": fs / fs_se, "F": (fs / fs_se) ** 2})
    return EngelResults(table, coefficients, first_stage, len(sample))


SPENDING_PERCENTILES = (10, 50, 90)
MIN_SHARE_RATIO = 0.25


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The value below which a share `q` of the total weight lies."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cumulative, q)])


def fit_quadratic(
    households: pd.DataFrame,
    categories: list[str] = COMPONENTS,
    min_income: float = MIN_INCOME,
    percentiles: tuple[int, ...] = SPENDING_PERCENTILES,
) -> pd.DataFrame:
    """Budget share model with curvature: does a category's elasticity change with the size of the budget?

    Adds the square of log total spending to the budget share model,

        share_c = a_c + b_c * z + c_c * z**2 + controls,    z = log(total spending) - its mean,

    so the elasticity, 1 + (b_c + 2 * c_c * z) / share_c(z), can differ between
    small and large budgets. (With `c_c` = 0 this is the model in `fit_engel`.)
    Both z and z**2 are instrumented, with log income and its square. The
    square is instrumented directly: squaring the fitted value of z from a
    first stage would not be valid.

    One row per category: `slope` (b_c) and `curvature` (c_c) with standard
    errors, and the predicted budget share and elasticity at the given
    percentiles of total spending for a household with average characteristics.

    Nothing keeps this model's predicted shares above zero, and an elasticity
    divides by the share. Where the predicted share falls below
    MIN_SHARE_RATIO of the category's average share the elasticity is left
    missing rather than reported as a huge, meaningless number.
    """
    sample = model_sample(households, min_income)
    X = controls_matrix(sample).to_numpy()
    total = sample["total_annual"].to_numpy()
    shares = sample[[f"{c}_annual" for c in categories]].to_numpy() / total[:, None]
    full_weight = sample["weight"].to_numpy()
    z = np.log(total) - np.average(np.log(total), weights=full_weight)
    levels = [weighted_quantile(total, full_weight, p / 100) for p in percentiles]
    z_at = np.log(levels) - np.average(np.log(total), weights=full_weight)
    endogenous = np.column_stack([z, z**2])

    draws = []
    for income in income_columns(sample):
        log_income = np.log(sample[income].to_numpy())
        log_income = log_income - np.average(log_income, weights=full_weight)
        Z = np.column_stack([log_income, log_income**2, X])  # instruments + controls
        by_weight = []
        for weight in WEIGHT_COLUMNS:
            w = sample[weight].to_numpy()
            fitted = Z @ wls(Z, endogenous, w)
            slope, curvature = wls(np.column_stack([fitted, X]), shares, w)[:2]
            mean_share, mean_z, mean_z2 = (w @ v / w.sum() for v in (shares, z, z**2))
            columns = [mean_share, slope, curvature]
            for at in z_at:
                share_at = mean_share + slope * (at - mean_z) + curvature * (at**2 - mean_z2)
                columns += [share_at, 1 + (slope + 2 * curvature * at) / share_at]
            by_weight.append(np.column_stack(columns))
        draws.append(by_weight)
    estimate, se = estimate_and_se(np.array(draws))  # (categories, quantities)

    names = ["budget_share", "slope", "curvature"]
    names += [f"{quantity}_p{p}" for p in percentiles for quantity in ("share", "elasticity")]
    table = pd.DataFrame(index=pd.Index(categories, name="category"))
    for i, name in enumerate(names):
        table[name], table[f"{name}_se"] = estimate[:, i], se[:, i]
    table.insert(6, "curvature_t", table["curvature"] / table["curvature_se"])
    for p, level in zip(percentiles, levels):
        too_small = table[f"share_p{p}"] < MIN_SHARE_RATIO * table["budget_share"]
        table.loc[too_small, [f"elasticity_p{p}", f"elasticity_p{p}_se"]] = np.nan
        table[f"spending_p{p}"] = level
    table["n"] = len(sample)
    return table
