"""Variance estimation for the CE Interview survey.

Two sources of uncertainty are combined:

* Sampling error, from the 44 balanced repeated replication (BRR) weights BLS
  publishes. An estimate is recomputed with each replicate weight and the
  variance is sum((replicate - full) ** 2) / 44.
* Income imputation error. BLS fills in missing income five times
  (FINCBTX1-FINCBTX5). Any estimate that uses income is computed once per
  imputation and combined with Rubin's rules: the point estimate is the mean
  across imputations and the variance is the mean replicate variance plus
  (1 + 1/5) times the variance between imputations.

Estimators in this package compute a "draws" array of shape
(n_imputations, 1 + N_REPLICATES, ...): for each imputation, the estimate with
the full-sample weight followed by the 44 replicate estimates.
`estimate_and_se` reduces that array to a point estimate and standard error.
"""

import numpy as np

N_REPLICATES = 44
N_IMPUTATIONS = 5

REPLICATE_WEIGHT_COLUMNS = [f"rep_weight_{i:02d}" for i in range(1, N_REPLICATES + 1)]
# Full-sample weight first, then the replicates: the layout of axis 1 of a draws array.
WEIGHT_COLUMNS = ["weight", *REPLICATE_WEIGHT_COLUMNS]

# Two-sided 95% critical value. Normal rather than t: with Rubin's rules the
# degrees of freedom vary by estimate and are large for nearly all of them.
Z_95 = 1.96


def replicate_se(estimate: float, replicates: np.ndarray) -> float:
    """CE variance formula for its 44 balanced repeated replication weights."""
    return float(np.sqrt(np.sum((np.asarray(replicates) - estimate) ** 2) / N_REPLICATES))


def estimate_and_se(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point estimates and standard errors from a (imputations, 1 + replicates, ...) array."""
    draws = np.asarray(draws, dtype=float)
    full = draws[:, 0]
    within = np.sum((draws[:, 1:] - full[:, None]) ** 2, axis=1) / N_REPLICATES
    n_imputations = draws.shape[0]
    estimate = full.mean(axis=0)
    variance = within.mean(axis=0)
    if n_imputations > 1:
        variance = variance + (1 + 1 / n_imputations) * full.var(axis=0, ddof=1)
    return estimate, np.sqrt(variance)
