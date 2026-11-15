import numpy as np
import pandas as pd
import pytest

from cex.clean import (
    CATEGORIES,
    COMPONENTS,
    clean_fmli,
    household_year,
    income_imputations,
    income_quintile,
    months_in_scope,
    spending_columns,
)
from cex.load import load_fmli

from conftest import YEAR


@pytest.fixture(scope="module")
def interviews(synthetic_raw):
    return clean_fmli(load_fmli(YEAR, synthetic_raw), YEAR)


@pytest.fixture(scope="module")
def clean(synthetic_year):
    return synthetic_year.interviews


def test_months_in_scope():
    years = pd.Series([2024, 2024, 2024, 2024, 2024, 2025, 2025, 2025])
    months = pd.Series([1, 2, 3, 4, 12, 1, 2, 3])
    assert months_in_scope(years, months, 2024).tolist() == [0, 1, 2, 3, 3, 3, 2, 1]


def test_interview_table_has_no_spending_of_its_own(interviews):
    assert interviews["newid"].is_unique
    assert not spending_columns(interviews)
    # BLS's summary variables are kept, over the whole recall period, for cross-checking only.
    assert [c.removeprefix("fmli_") for c in interviews if c.startswith("fmli_")] == CATEGORIES
    np.testing.assert_allclose(interviews[[f"fmli_{c}" for c in COMPONENTS]].sum(axis=1), interviews["fmli_total"])


def test_spending_is_joined_by_interview(synthetic_year, clean):
    assert len(COMPONENTS) == 15
    assert set(spending_columns(clean)) == {"total", *COMPONENTS, *synthetic_year.detail}
    assert set(spending_columns(clean, "in_year")) == set(spending_columns(clean))
    assert len(clean) == clean["newid"].nunique()


def test_blank_replicate_weight_means_excluded(clean):
    replicate_weights = clean.filter(like="rep_weight_")
    assert replicate_weights.shape[1] == 44
    assert replicate_weights.notna().all().all()
    assert (clean["rep_weight_05"] == 0).any()


def test_income_quintile_boundaries():
    quintiles = income_quintile(pd.Series([0.01, 0.2, 0.2001, 0.5, 0.8, 0.95, 1.0, np.nan]))
    assert quintiles.iloc[:-1].tolist() == ["lowest", "lowest", "second", "third", "fourth", "highest", "highest"]
    assert pd.isna(quintiles.iloc[-1])


def test_income_imputations(clean):
    versions = list(income_imputations(clean))
    assert len(versions) == 5
    third = versions[2]
    np.testing.assert_array_equal(third["income_before_tax"], clean["income_before_tax_3"])
    assert third["income_quintile"].equals(income_quintile(clean["income_rank_3"]))
    # FINCBTXM, the default income, is the mean of the five imputations.
    np.testing.assert_allclose(np.mean([v["income_before_tax"] for v in versions], axis=0), clean["income_before_tax"])
    # Without imputation columns there is just the one version.
    single = clean.drop(columns=[f"income_before_tax_{m}" for m in range(1, 6)])
    assert len(list(income_imputations(single))) == 1


def test_household_year(synthetic_year, clean):
    households = household_year(clean)
    assert households.index.is_unique
    assert len(households) == clean["cuid"].nunique()
    assert households["interviews"].sum() == len(clean)
    # Each interview weight is a quarter of a population-year.
    assert households["weight"].sum() == pytest.approx(clean["weight"].sum() / 4)
    assert households["rep_weight_07"].sum() == pytest.approx(clean["rep_weight_07"].sum() / 4)
    # Every category is carried over, detail included, and none of the cross-check or in-year columns.
    assert set(spending_columns(households)) == {"total", *COMPONENTS, *synthetic_year.detail}
    assert not [c for c in households if c.startswith("fmli_") or c.endswith("_in_year")]

    cuid = households.index[households["interviews"] == 4][0]
    interviews = clean[clean["cuid"] == cuid].sort_values(["interview_year", "interview_month"])
    assert households.loc[cuid, "groceries_annual"] == pytest.approx(interviews["groceries_annual"].mean())
    assert households.loc[cuid, "gasoline_annual"] == pytest.approx(interviews["gasoline_annual"].mean())
    assert households.loc[cuid, "income_before_tax_2"] == pytest.approx(interviews["income_before_tax_2"].mean())
    assert households.loc[cuid, "age_ref"] == interviews["age_ref"].iloc[-1]
    assert households.loc[cuid, "vehicles"] == interviews["vehicles"].iloc[-1]
    assert isinstance(households["education"].dtype, pd.CategoricalDtype)
