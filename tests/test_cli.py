import argparse

import pandas as pd
import pytest

from cex.cli import default_root, main, parse_years

from conftest import write_synthetic_prices, write_synthetic_raw


def test_parse_years():
    assert parse_years("2024") == [2024]
    assert parse_years("2020-2022") == [2020, 2021, 2022]
    assert parse_years("2019, 2022-2023") == [2019, 2022, 2023]
    for bad in ["2024-2020", "abc", "2020-"]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_years(bad)


def test_run_multiple_years_end_to_end(tmp_path):
    data = tmp_path / "data"
    # 2023's release starts at Apr-Jun, so this also exercises reading Jan-Mar from the 2022 zip.
    write_synthetic_raw(data / "raw", years=[2023, 2024], seed=10, n_cu=1_500)
    write_synthetic_prices(data / "raw", 2023, 2025)
    out = tmp_path / "outputs"

    main(["run", "--years", "2023-2024", "--data-dir", str(data), "--output-dir", str(out)])

    tables = [
        "annual_means",
        "group_means",
        "detail_means",
        "elasticities",
        "engel_coefficients",
        "detail_elasticities",
        "engel_curvature",
        "inflation",
        "fmli_check",
    ]
    figures = ["elasticities", "detail_elasticities", "engel_curvature", "budget_shares_by_quintile", "engel_curves"]
    for year in (2023, 2024):
        assert (data / "processed" / f"interviews_{year}.parquet").exists()
        for table in tables:
            assert (out / str(year) / f"{table}.csv").exists(), table
        for figure in figures:
            assert (out / str(year) / f"{figure}.png").exists(), figure
    both = out / "2023-2024"
    trends = pd.read_csv(both / "trends.csv")
    assert sorted(trends["year"].unique()) == [2023, 2024]
    elasticities = pd.read_csv(both / "elasticities.csv")
    assert sorted(elasticities["year"].unique()) == [2023, 2024] and len(elasticities) == 2 * 16
    cumulative = pd.read_csv(both / "inflation.csv")
    assert (cumulative["from_year"] == 2023).all() and (cumulative["to_year"] == 2025).all()
    renters = cumulative[cumulative["group"] == "renter"].iloc[0]
    assert renters["gap"] > 0  # synthetic rent rises fastest
    assert pd.read_csv(both / "inflation_contributions.csv")["grouping"].eq("income_quintile").all()
    for figure in ["trends", "elasticities", "inflation"]:
        assert (both / f"{figure}.png").exists(), figure
    detail = pd.read_csv(out / "2024" / "detail_elasticities.csv", index_col="category")
    assert len(detail) == 31 and "income_elasticity" not in detail


def test_run_without_price_data_skips_inflation(tmp_path, capsys):
    data = tmp_path / "data"
    write_synthetic_raw(data / "raw", n_cu=600)
    out = tmp_path / "outputs"
    main(["run", "--years", "2024", "--data-dir", str(data), "--output-dir", str(out), "--no-plots"])
    assert "Skipping inflation by group" in capsys.readouterr().out
    assert (out / "2024" / "elasticities.csv").exists()
    assert not (out / "2024" / "inflation.csv").exists()
    assert not list(out.rglob("*.png"))

    # With prices for 2024 but not 2025, there is still nothing to reprice the 2024 basket with.
    write_synthetic_prices(data / "raw", 2023, 2024)
    main(["run", "--years", "2024", "--data-dir", str(data), "--output-dir", str(out), "--no-plots"])
    assert "No inflation by group for 2024-2025" in capsys.readouterr().out


def test_download_fetches_definitions_prices_and_the_previous_release(tmp_path, monkeypatch):
    calls = []
    raw = write_synthetic_raw(tmp_path / "raw", years=[2023], n_cu=50)
    monkeypatch.setattr("cex.cli.download_stubs", lambda raw_dir, contact, overwrite: calls.append("stubs"))
    monkeypatch.setattr(
        "cex.cli.download_prices", lambda raw_dir, start, end, overwrite: calls.append(("prices", start))
    )
    monkeypatch.setattr("cex.cli.download_interview", lambda year, *args: calls.append(year))
    main(["download", "--years", "2023", "--contact", "someone@example.com", "--data-dir", str(raw.parent)])
    # The 2023 release starts at Apr-Jun, so 2022's is fetched as well.
    assert calls == ["stubs", ("prices", 2023), 2023, 2022]


def test_run_checks_every_input_before_starting(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "intrvw23.zip").touch()
    with pytest.raises(SystemExit, match="intrvw22.zip"):
        main(["run", "--years", "2022-2023", "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])


def test_run_needs_cpi_for_trends(tmp_path):
    (tmp_path / "raw").mkdir()
    for yy in ("02", "03"):
        (tmp_path / "raw" / f"intrvw{yy}.zip").touch()
    with pytest.raises(SystemExit, match="No CPI-U annual average for 2002"):
        main(["run", "--years", "2002-2003", "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])


def test_download_requires_contact(monkeypatch):
    monkeypatch.delenv("CEX_CONTACT_EMAIL", raising=False)
    with pytest.raises(SystemExit):
        main(["download", "--years", "2024"])


def test_default_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CEX_HOME", str(tmp_path))
    assert default_root() == tmp_path
    monkeypatch.delenv("CEX_HOME")
    assert (default_root() / "pyproject.toml").exists()  # running from the source checkout
