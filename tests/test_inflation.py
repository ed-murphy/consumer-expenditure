import numpy as np
import pandas as pd
import pytest

from cex import inflation
from cex.prices import ALL_ITEMS, CPI_SERIES, category_price_change, load_prices

from conftest import ANNUAL_INFLATION, BASE_INFLATION, YEAR, write_synthetic_prices


@pytest.fixture(scope="module")
def df(synthetic_year):
    return synthetic_year.interviews


@pytest.fixture(scope="module")
def prices(tmp_path_factory):
    return load_prices(write_synthetic_prices(tmp_path_factory.mktemp("prices"), YEAR, YEAR + 2).parent)


@pytest.fixture(scope="module")
def change(prices):
    return category_price_change(prices, YEAR, YEAR + 1)


@pytest.fixture(scope="module")
def table(df, change):
    return inflation.group_inflation(df, change).set_index(["grouping", "group"])


def test_basket_prices_owners_shelter_at_rental_value(df):
    basket = inflation.basket_spending(df)
    assert list(basket.columns) == list(CPI_SERIES) and "cash_contributions" not in basket
    owners = (df["tenure"] == "owner").to_numpy()
    expected = df["rent_equivalent"] * df["months_in_scope"]
    np.testing.assert_allclose(basket.loc[owners, "owned_dwellings"], expected[owners])
    assert (basket.loc[~owners, "owned_dwellings"] == 0).all()  # blank for everyone else
    np.testing.assert_allclose(basket["groceries"], df["groceries_in_year"])


def test_same_price_change_everywhere_means_same_inflation_for_everyone(df, change):
    flat = pd.Series(1.05, index=change.index)
    table = inflation.group_inflation(df, flat)
    np.testing.assert_allclose(table["inflation"], 0.05)
    np.testing.assert_allclose(table[["inflation_se", "gap", "gap_se"]], 0, atol=1e-12)


def test_inflation_is_the_basket_weighted_price_change(df, change, table):
    basket = inflation.basket_spending(df).mul(df["weight"], axis=0).sum()
    expected = (basket / basket.sum()) @ (change - 1)
    everyone = table.loc[("all", "all")]
    assert everyone["inflation"] == pytest.approx(expected)
    assert everyone["gap"] == 0 and everyone["gap_se"] == 0
    assert 0 < everyone["inflation_se"] < 0.002
    # Rent rises 8% and gasoline falls, everything else rises 3%.
    rent, gasoline = basket[["rented_dwellings", "gasoline"]] / basket.sum()
    assert expected == pytest.approx(BASE_INFLATION + rent * 0.05 - gasoline * 0.05)


def test_renters_face_higher_inflation_when_rent_rises_fastest(table):
    assert list(table.index.get_level_values("grouping").unique()) == ["all", *inflation.GROUPINGS]
    renters, owners = table.loc[("tenure", "renter")], table.loc[("tenure", "owner")]
    assert renters["gap"] > 3 * renters["gap_se"] > 0
    assert owners["gap"] < 0
    assert renters["gap"] == pytest.approx(renters["inflation"] - table.loc[("all", "all"), "inflation"])
    # The income groups carry imputation error on top of sampling error.
    assert (table.loc["income_quintile", "inflation_se"] > 0).all()


def test_contributions_add_up_to_the_gap(df, change, table):
    parts = inflation.gap_contributions(df, change, "tenure")
    assert set(parts["category"]) == set(CPI_SERIES)
    gaps = parts.groupby("group", observed=True)["contribution"].sum()
    for group in ["owner", "renter", "other"]:
        assert gaps[group] == pytest.approx(table.loc[("tenure", group), "gap"])
    assert parts.groupby("group", observed=True)["share"].sum().tolist() == pytest.approx([1, 1, 1])
    renters = parts[parts["group"] == "renter"].set_index("category")["contribution"]
    assert renters.idxmax() == "rented_dwellings"


def test_chain_compounds_the_years(df, prices):
    links = {
        year: inflation.group_inflation(df, category_price_change(prices, year, year + 1)) for year in (YEAR, YEAR + 1)
    }
    cumulative = inflation.chain(links).set_index(["grouping", "group"])
    first, second = (links[year].set_index(["grouping", "group"]) for year in (YEAR, YEAR + 1))
    assert (cumulative["from_year"] == YEAR).all() and (cumulative["to_year"] == YEAR + 2).all()
    np.testing.assert_allclose(cumulative["inflation"], (1 + first["inflation"]) * (1 + second["inflation"]) - 1)
    everyone = cumulative.loc[("all", "all")]
    np.testing.assert_allclose(cumulative["gap"], cumulative["inflation"] - everyone["inflation"])
    assert everyone["gap_se"] == 0
    # Two years of similar sampling error: about sqrt(2) times one year's.
    ratio = cumulative.loc[("tenure", "renter"), "gap_se"] / first.loc[("tenure", "renter"), "gap_se"]
    assert ratio == pytest.approx(np.sqrt(2), rel=0.15)
    with pytest.raises(ValueError, match="consecutive"):
        inflation.chain({YEAR: links[YEAR], YEAR + 2: links[YEAR + 1]})


def test_official_cpi_change_and_year_link(df, prices, table):
    assert inflation.official_cpi_change(prices, YEAR, YEAR + 2) == pytest.approx(
        (1 + ANNUAL_INFLATION[ALL_ITEMS]) ** 2 - 1
    )
    link = inflation.year_link(df, prices, YEAR).set_index(["grouping", "group"])
    pd.testing.assert_frame_equal(link, table)
    with pytest.raises(KeyError, match=str(YEAR + 3)):
        inflation.year_link(df, prices, YEAR + 2)
