import numpy as np
import pytest

from cex.clean import COMPONENTS, INCOME_QUINTILES, use_imputation
from cex.summarize import GROUPINGS, annual_means, group_means, means_by, trends

IMPUTATION_COLUMNS = [f"income_before_tax_{m}" for m in range(1, 6)] + [f"income_rank_{m}" for m in range(1, 6)]


@pytest.fixture(scope="module")
def clean(synthetic_year):
    return synthetic_year.interviews


@pytest.fixture(scope="module")
def means(clean):
    return annual_means(clean)


def test_annual_means_population(clean, means):
    expected = (clean["weight"] * clean["months_in_scope"] / 12).sum()
    assert means.loc["consumer_units", "mean"] == pytest.approx(expected)
    assert (means["mean_se"] > 0).all()


def test_budget_shares(means):
    assert means.loc["total", "share"] == pytest.approx(1)
    assert means.loc[COMPONENTS, "share"].sum() == pytest.approx(1)
    assert (means.loc[COMPONENTS, "share_se"] > 0).all()


def test_any_categories_can_be_tabulated(synthetic_year, clean, means):
    detail = annual_means(clean, categories=synthetic_year.detail)
    assert list(detail.index) == ["consumer_units", "income_before_tax", *synthetic_year.detail]
    # Shares are of total spending, whether or not the total is in the table.
    assert detail.loc[synthetic_year.detail, "share"].sum() == pytest.approx(1)
    assert detail.loc["groceries"].equals(means.loc["groceries"])
    parts = ["owned_dwellings", "rented_dwellings", "other_lodging", "utilities", "household_operations", "furnishings"]
    assert detail.loc[parts, "mean"].sum() == pytest.approx(means.loc["housing", "mean"])


def test_income_combines_imputations_with_rubins_rules(clean, means):
    per_imputation = [
        annual_means(use_imputation(clean, m).drop(columns=IMPUTATION_COLUMNS)).loc["income_before_tax"]
        for m in range(1, 6)
    ]
    estimates = np.array([row["mean"] for row in per_imputation])
    variances = np.array([row["mean_se"] ** 2 for row in per_imputation])
    combined = means.loc["income_before_tax"]
    assert combined["mean"] == pytest.approx(estimates.mean())
    assert combined["mean_se"] == pytest.approx(np.sqrt(variances.mean() + 1.2 * estimates.var(ddof=1)))
    # Spending doesn't involve income, so its standard error is the plain replicate one.
    single = annual_means(clean.drop(columns=IMPUTATION_COLUMNS))
    assert means.loc["total", "mean_se"] == pytest.approx(single.loc["total", "mean_se"])


def test_income_quintiles_split_population_evenly(clean):
    table = means_by(clean, "income_quintile")
    population = table.xs("consumer_units", level="item")["mean"]
    assert list(population.index) == INCOME_QUINTILES
    np.testing.assert_allclose(population / population.sum(), 0.2, atol=0.02)
    # Synthetic spending rises with income.
    assert table.xs("total", level="item")["mean"].is_monotonic_increasing


def test_group_means(clean, means):
    table = group_means(clean)
    assert set(table["grouping"]) == set(GROUPINGS)
    # Groups without missing values partition the population.
    for grouping in ["income_quintile", "age_group", "tenure", "education"]:
        rows = table[(table["grouping"] == grouping) & (table["item"] == "consumer_units")]
        assert rows["mean"].sum() == pytest.approx(means.loc["consumer_units", "mean"])
    gasoline = group_means(clean, groupings=["tenure"], categories=["gasoline"])
    assert set(gasoline["item"]) == {"consumer_units", "income_before_tax", "gasoline"}


def test_trends_deflate_to_the_latest_year(means):
    table = trends({2023: means, 2024: means}).set_index(["year", "item"])
    assert table.loc[(2023, "total"), "real_mean"] == pytest.approx(means.loc["total", "mean"] * 313.689 / 304.702)
    assert table.loc[(2024, "total"), "real_mean"] == pytest.approx(means.loc["total", "mean"])
    assert table.loc[(2023, "consumer_units"), "real_mean"] == table.loc[(2023, "consumer_units"), "mean"]
    assert table.loc[(2023, "groceries"), "share"] == table.loc[(2024, "groceries"), "share"]
