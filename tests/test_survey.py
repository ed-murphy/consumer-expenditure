import numpy as np
import pytest

from cex.survey import N_REPLICATES, estimate_and_se, replicate_se


def test_replicate_se_formula():
    replicates = np.full(N_REPLICATES, 2.0)
    replicates[0] = 4.0
    assert replicate_se(2.0, replicates) == pytest.approx(np.sqrt(4 / 44))


def test_estimate_and_se_with_one_imputation():
    draws = np.zeros((1, 1 + N_REPLICATES, 2))
    draws[0, :, 0] = 5.0  # no sampling variation
    draws[0, 1:, 1] = 1.0  # every replicate off by 1
    estimate, se = estimate_and_se(draws)
    np.testing.assert_allclose(estimate, [5.0, 0.0])
    np.testing.assert_allclose(se, [0.0, 1.0])


def test_rubins_rules():
    # Five imputations with no sampling error: the variance is (1 + 1/5) times the between-imputation variance.
    full = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    draws = np.repeat(full[:, None], 1 + N_REPLICATES, axis=1)
    estimate, se = estimate_and_se(draws)
    assert estimate == pytest.approx(3.0)
    assert se == pytest.approx(np.sqrt(1.2 * 2.5))
