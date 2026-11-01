import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from cex.clean import clean_fmli, months_in_scope
from cex.load import load_fmli
from cex.model import elasticity_table, fit_engel
from cex.summarize import annual_means

from conftest import TRUE_FOOD_ELASTICITY, YEAR


@pytest.fixture(scope="module")
def clean(synthetic_zip):
    return clean_fmli(load_fmli(synthetic_zip), YEAR)


def test_load_stacks_quarterly_files_in_order(synthetic_zip):
    raw = load_fmli(synthetic_zip)
    assert list(raw["SOURCE_FILE"].unique()) == ["fmli241x", "fmli242", "fmli243", "fmli244", "fmli251"]
    assert len(raw) == 10_000
    assert raw["REGION"].isna().sum() == 6
    # GROCER* columns in later quarters are read as FDHOME*.
    assert (raw["FDHOMEPQ"] > 0).all() and "GROCERPQ" not in raw


def test_load_rejects_file_missing_a_column(synthetic_zip, tmp_path):
    bad = tmp_path / "intrvw24.zip"
    with zipfile.ZipFile(synthetic_zip) as src, zipfile.ZipFile(bad, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name.endswith("fmli243.csv"):
                df = pd.read_csv(io.BytesIO(data)).drop(columns=["HOUSPQ"])
                data = df.to_csv(index=False).encode()
            dst.writestr(name, data)
    with pytest.raises(KeyError, match="fmli243.*HOUSPQ"):
        load_fmli(bad)


def test_months_in_scope():
    years = pd.Series([2024, 2024, 2024, 2024, 2024, 2025, 2025, 2025])
    months = pd.Series([1, 2, 3, 4, 12, 1, 2, 3])
    assert months_in_scope(years, months, 2024).tolist() == [0, 1, 2, 3, 3, 3, 2, 1]


def test_in_year_spending_never_exceeds_quarter(clean):
    assert (clean["total_in_year"] <= clean["total_annual"] / 4 + 1e-6).all()
    middle = clean["source_file"].isin(["fmli242", "fmli243", "fmli244"])
    np.testing.assert_allclose(clean.loc[middle, "total_in_year"], clean.loc[middle, "total_annual"] / 4)


def test_annual_means_population(clean):
    means = annual_means(clean)
    expected = (clean["weight"] * clean["months_in_scope"] / 12).sum()
    assert means.loc["consumer_units", "mean"] == pytest.approx(expected)
    assert (means["se"] > 0).all()


def test_engel_recovers_known_elasticity(clean):
    fit = fit_engel(clean, "food_home")
    b = fit["coefficients"].loc["log_income"]
    assert abs(b["coef"] - TRUE_FOOD_ELASTICITY) < 3 * b["se"] + 0.02
    assert fit["n"] == 10_000 - 6  # rows with missing region are dropped


def test_elasticity_table(clean):
    table = elasticity_table(clean, ["total", "food_home"])
    assert list(table.index) == ["total", "food_home"]
    assert table["ci_low"].lt(table["ci_high"]).all()
