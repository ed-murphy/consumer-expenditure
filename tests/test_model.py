import numpy as np
import pytest

from cex import model
from cex.clean import COMPONENTS, household_year
from cex.model import MIN_INCOME, classify, fit_engel, fit_quadratic, model_sample, ppml, weighted_quantile

from conftest import APPAREL_INCOME_ELASTICITY, GROCERY_SHARE_SLOPE


@pytest.fixture(scope="module")
def households(synthetic_year):
    return household_year(synthetic_year.interviews)


@pytest.fixture(scope="module")
def results(households):
    return fit_engel(households)


def test_recovers_budget_share_slope(results):
    row = results.coefficients.loc[("groceries", "log_total_spending")]
    assert abs(row["coef"] - GROCERY_SHARE_SLOPE) < 3 * row["se"]
    assert row["se"] < 0.002  # and the estimate is tight enough for that to mean something
    # A grocery share that falls as spending rises makes groceries a necessity.
    assert results.elasticities.loc["groceries", "classification"] == "necessity"


def test_expenditure_elasticities_add_up_to_one(results):
    components = results.elasticities.drop(index="total")
    assert components["budget_share"].sum() == pytest.approx(1)
    assert (components["budget_share"] * components["expenditure_elasticity"]).sum() == pytest.approx(1, abs=1e-9)


def test_recovers_income_elasticity_despite_zero_spending(results):
    apparel = results.elasticities.loc["apparel"]
    assert apparel["share_positive"] < 0.9  # households with no apparel purchases stay in
    assert abs(apparel["income_elasticity"] - APPAREL_INCOME_ELASTICITY) < 3 * apparel["income_elasticity_se"]
    assert apparel["income_elasticity_se"] < 0.1


def test_first_stage_is_strong(results):
    assert results.first_stage["coef"] > 0
    assert results.first_stage["F"] > 100


def test_detail_categories_without_the_income_model(synthetic_year, households, results):
    detail = fit_engel(households, categories=synthetic_year.detail, income_model=False)
    table = detail.elasticities
    assert list(table.index) == synthetic_year.detail
    # They partition total spending too, so their elasticities also average to 1.
    assert (table["budget_share"] * table["expenditure_elasticity"]).sum() == pytest.approx(1, abs=1e-9)
    assert table["income_elasticity"].isna().all()
    # A category that is the same at both levels gets the same answer.
    for column in ["budget_share", "expenditure_elasticity", "expenditure_elasticity_se"]:
        assert table.loc["groceries", column] == pytest.approx(results.elasticities.loc["groceries", column])
    # Synthetic rent barely rises with income and lodging on trips rises fast.
    assert (
        table.loc["rented_dwellings", "expenditure_elasticity"] < table.loc["other_lodging", "expenditure_elasticity"]
    )


@pytest.fixture(scope="module")
def curvature(households):
    return fit_quadratic(households)


def test_quadratic_model_finds_no_curvature_in_a_linear_share(curvature, results):
    groceries = curvature.loc["groceries"]
    # The synthetic grocery share is exactly linear in log total spending.
    assert abs(groceries["curvature_t"]) < 3
    assert abs(groceries["slope"] - GROCERY_SHARE_SLOPE) < 3 * groceries["slope_se"]
    assert groceries["slope_se"] < 0.003
    # So the elasticity at the median is close to the linear model's, and it falls as the budget grows
    # only because the share itself falls.
    assert groceries["elasticity_p50"] == pytest.approx(
        results.elasticities.loc["groceries", "expenditure_elasticity"], abs=0.05
    )
    assert groceries["share_p10"] > groceries["share_p50"] > groceries["share_p90"] > 0


def test_quadratic_model_adds_up(curvature):
    assert list(curvature.index) == COMPONENTS
    # Shares sum to 1 at every budget, so the slopes and the curvatures each sum to 0.
    assert curvature["budget_share"].sum() == pytest.approx(1)
    assert curvature["slope"].sum() == pytest.approx(0, abs=1e-9)
    assert curvature["curvature"].sum() == pytest.approx(0, abs=1e-9)
    for p in (10, 50, 90):
        assert curvature[f"share_p{p}"].sum() == pytest.approx(1)
        weighted = (curvature[f"share_p{p}"] * curvature[f"elasticity_p{p}"]).sum()
        assert weighted == pytest.approx(1, abs=1e-4)  # not exact: each is an average over income imputations
    assert curvature["spending_p10"].iloc[0] < curvature["spending_p50"].iloc[0] < curvature["spending_p90"].iloc[0]


def test_quadratic_model_leaves_out_elasticities_where_the_share_vanishes(households, monkeypatch):
    monkeypatch.setattr(model, "MIN_SHARE_RATIO", 1.0)  # keep only where the predicted share is above average
    table = fit_quadratic(households, categories=["groceries", "insurance_pensions"])
    # Groceries take a larger share of small budgets; insurance and pensions of large ones.
    assert table.loc["groceries", ["elasticity_p10", "elasticity_p90"]].notna().tolist() == [True, False]
    assert table.loc["insurance_pensions", ["elasticity_p10", "elasticity_p90"]].notna().tolist() == [False, True]
    assert table[["share_p10", "share_p90"]].notna().all().all()


def test_weighted_quantile():
    values, weights = np.array([3.0, 1.0, 2.0, 4.0]), np.array([1.0, 1.0, 6.0, 2.0])
    assert weighted_quantile(values, weights, 0.5) == 2.0
    assert weighted_quantile(values, weights, 0.05) == 1.0
    assert weighted_quantile(values, weights, 0.95) == 4.0


def test_model_sample_uses_every_income_imputation(households):
    sample = model_sample(households)
    incomes = sample[[f"income_before_tax_{m}" for m in range(1, 6)]]
    assert (incomes >= MIN_INCOME).all().all()
    assert (sample["total_annual"] > 0).all()


def test_ppml_recovers_an_exact_fit():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(500), rng.normal(size=500), rng.integers(0, 2, 500)])
    beta = np.array([2.0, 0.7, -0.3])
    np.testing.assert_allclose(ppml(X, np.exp(X @ beta), rng.uniform(1, 2, 500)), beta, atol=1e-8)


def test_ppml_keeps_zeros():
    # y is zero half the time; the mean still follows exp(1 + 0.5 x).
    rng = np.random.default_rng(2)
    x = rng.normal(size=20_000)
    y = np.exp(1 + 0.5 * x) * 2 * (rng.random(20_000) < 0.5)
    beta = ppml(np.column_stack([np.ones_like(x), x]), y, np.ones_like(x))
    assert beta[1] == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize(
    ("low", "high", "label"),
    [(-0.9, -0.1, "inferior"), (0.2, 0.4, "necessity"), (1.1, 1.5, "luxury"), (0.9, 1.1, "unclear")],
)
def test_classify(low, high, label):
    assert classify(low, high) == label
