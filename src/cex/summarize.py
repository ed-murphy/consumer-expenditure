"""BLS-style weighted annual means, with replicate-weight standard errors.

This mirrors how BLS builds its published "average annual expenditures" tables
from the Interview data, which makes it a good check that the data loaded correctly.
Published Interview means will be close but not identical: BLS tables integrate
Diary data for some categories (notably food) and use more detailed rules.
"""

import numpy as np
import pandas as pd

from cex.clean import CATEGORIES

N_REPLICATES = 44


def replicate_se(estimate: float, replicates: np.ndarray) -> float:
    """CE variance formula for its 44 balanced repeated replication weights."""
    return float(np.sqrt(np.sum((replicates - estimate) ** 2) / N_REPLICATES))


def _annual_means(df: pd.DataFrame, weight: pd.Series) -> pd.Series:
    # Each interview represents its share of the year: months_in_scope / 12.
    population_weight = weight * df["months_in_scope"] / 12
    population = population_weight.sum()
    means = {"consumer_units": population}
    means["income_before_tax"] = (population_weight * df["income_before_tax"]).sum() / population
    for cat in CATEGORIES:
        means[cat] = (weight * df[f"{cat}_in_year"]).sum() / population
    return pd.Series(means)


def annual_means(df: pd.DataFrame) -> pd.DataFrame:
    estimate = _annual_means(df, df["weight"])
    replicates = np.column_stack(
        [_annual_means(df, df[f"rep_weight_{i:02d}"]) for i in range(1, N_REPLICATES + 1)]
    )
    se = [replicate_se(est, reps) for est, reps in zip(estimate, replicates)]
    return pd.DataFrame({"mean": estimate, "se": se})
