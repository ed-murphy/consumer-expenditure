import numpy as np
import pandas as pd
import pytest

from cex import spending, stubs
from cex.clean import COMPONENTS
from cex.summarize import annual_means

from conftest import FOOD_SHARE_OF_GROCERIES, GROCERY_UCC, ITEMS, RENTAL_VALUE_UCC, TRIP_GROCERY_UCC, YEAR


@pytest.fixture(scope="module")
def df(synthetic_year):
    return synthetic_year.interviews


def test_total_is_every_listed_ucc_times_its_factor(synthetic_year, df):
    records = synthetic_year.records
    factor = {item.ucc: item.factor for item in ITEMS} | {GROCERY_UCC: 1, TRIP_GROCERY_UCC: 1}
    assert set(records["ucc"]) == set(factor)  # the rental value of the home, also in MTBI, is not spending
    assert RENTAL_VALUE_UCC not in factor
    expected = (records["cost"] * records["ucc"].map(factor)).groupby(records["newid"]).sum()
    np.testing.assert_allclose(df.set_index("newid")["total_annual"] / 4, expected.reindex(df["newid"]))


def test_categories_add_up(synthetic_year, df):
    np.testing.assert_allclose(df[[f"{c}_annual" for c in COMPONENTS]].sum(axis=1), df["total_annual"])
    np.testing.assert_allclose(df[[f"{c}_annual" for c in synthetic_year.detail]].sum(axis=1), df["total_annual"])
    np.testing.assert_allclose(df[[f"{c}_in_year" for c in synthetic_year.detail]].sum(axis=1), df["total_in_year"])
    shelter = ["owned_dwellings", "rented_dwellings", "other_lodging", "utilities", "household_operations"]
    np.testing.assert_allclose(df[[f"{c}_annual" for c in [*shelter, "furnishings"]]].sum(axis=1), df["housing_annual"])


def test_in_year_spending_is_the_part_of_the_recall_period_in_the_year(df):
    np.testing.assert_allclose(df["total_in_year"], df["total_annual"] / 4 * df["months_in_scope"] / 3)
    assert (df["months_in_scope"] < 3).any()


def test_interviews_without_a_purchase_get_zero(df):
    assert (df["education_annual"] == 0).mean() > 0.5
    assert df[[f"{c}_annual" for c in COMPONENTS]].notna().all().all()


def test_comparison_with_fmli_summary_variables(synthetic_year):
    check = synthetic_year.fmli_check.set_index(["category", "source_file"])
    # Most categories match BLS's summary variable to the dollar.
    for category in ["food_away", "housing", "transportation", "insurance_pensions"]:
        assert check.loc[category, "share_same"].min() == 1.0
    # Groceries: the summary variable is food only (80%) until it switches to all purchases in Apr-Jun 2024.
    groceries = check.loc["groceries", "ratio"]
    before = (1 - FOOD_SHARE_OF_GROCERIES) * 0.98  # what the food-only variable leaves out, as a share of groceries
    assert groceries["fmli241x"] == pytest.approx(1 / (1 - before), rel=1e-6)
    assert groceries[["fmli242", "fmli243", "fmli244", "fmli251"]].tolist() == pytest.approx([1.0] * 4)
    # Apparel: laundry is in the published category but not in the summary variable.
    assert (check.loc["apparel", "ratio"] > 1.05).all()
    # Miscellaneous: finance charges carry a factor of 4 that the summary variable does not apply.
    assert check.loc[("miscellaneous", "fmli251"), "ratio"] > 1.05
    spending.check_coverage(synthetic_year.fmli_check)


def test_coverage_check_catches_a_ucc_missing_from_the_tree(synthetic_year):
    tree = synthetic_year.tree[synthetic_year.tree["ucc"] != "260211"]  # electricity
    df = synthetic_year.interviews
    rebuilt = spending.interview_spending(synthetic_year.records, tree, df["newid"], YEAR)
    both = df[[c for c in df if c.startswith("fmli_") or c == "source_file"]].join(rebuilt, on=df["newid"])
    with pytest.raises(ValueError, match="housing"):
        spending.check_coverage(spending.compare_with_fmli(both))


def test_records_for_unknown_interviews_are_an_error(synthetic_year):
    newids = synthetic_year.interviews["newid"].iloc[1:]
    with pytest.raises(ValueError, match="no FMLI row"):
        spending.interview_spending(synthetic_year.records, synthetic_year.tree, newids, YEAR)


def test_ucc_in_both_files_is_an_error(synthetic_year):
    rent = synthetic_year.records[synthetic_year.records["ucc"] == "210110"].head(1).assign(file="itbi")
    records = pd.concat([synthetic_year.records, rent])
    with pytest.raises(ValueError, match="210110"):
        spending.interview_spending(records, synthetic_year.tree, synthetic_year.interviews["newid"], YEAR)


def test_tree_means_match_annual_means(synthetic_year, df):
    table = spending.tree_means(synthetic_year.records, synthetic_year.tree, df, YEAR)
    means = annual_means(df)
    assert len(table) == len(synthetic_year.tree)
    majors = table[table["level"] == 1]
    assert majors["mean"].sum() == pytest.approx(means.loc["total", "mean"])
    assert majors["share"].sum() == pytest.approx(1)
    by_title = table.set_index("title")
    for title, name in [("Food at home", "groceries"), ("Housing", "housing"), ("Reading", "reading")]:
        assert by_title.loc[title, "mean"] == pytest.approx(means.loc[name, "mean"])
        assert by_title.loc[title, "mean_se"] == pytest.approx(means.loc[name, "mean_se"])
        assert by_title.loc[title, "share_se"] == pytest.approx(means.loc[name, "share_se"])
    # Items carry their UCC and are counted with their factor.
    finance = table[table["ucc"] == "005420"].iloc[0]
    in_year = synthetic_year.records.query("ucc == '005420' and ref_year == @YEAR")
    weights = df.set_index("newid")["weight"]
    population = (df["weight"] * df["months_in_scope"] / 12).sum()
    assert finance["mean"] == pytest.approx(
        4 * (in_year["cost"] * weights[in_year["newid"]].to_numpy()).sum() / population
    )
    assert set(stubs.MAJOR) <= set(means.index)
