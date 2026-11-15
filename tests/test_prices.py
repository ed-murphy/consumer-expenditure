import io
import json

import pandas as pd
import pytest

from cex import prices

from conftest import ANNUAL_INFLATION, BASE_INFLATION, write_synthetic_prices


def fake_api(monkeypatch, missing=()):
    """Stand in for the BLS API: every requested series gets 12 months per year, plus the rows BLS adds."""
    requests = []

    def urlopen(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        series = []
        for number, series_id in enumerate(s for s in body["seriesid"] if s not in missing):
            data = [{"year": body["endyear"], "period": "M13", "value": "999"}]  # an annual average: not a month
            for year in range(int(body["startyear"]), int(body["endyear"]) + 1):
                for month in range(1, 13):
                    value = "-" if (year, month) == (2025, 10) else str(100 + number + month / 100)
                    data.append({"year": str(year), "period": f"M{month:02d}", "value": value})
            series.append({"seriesID": series_id, "data": data})
        return io.StringIO(json.dumps({"status": "REQUEST_SUCCEEDED", "message": [], "Results": {"series": series}}))

    monkeypatch.setattr(prices.urllib.request, "urlopen", urlopen)
    return requests


def test_every_priced_category_has_a_series():
    assert prices.ALL_ITEMS in prices.SERIES and set(prices.CPI_SERIES.values()) <= set(prices.SERIES)
    # Transfers and savings have no price.
    for unpriced in ["cash_contributions", "life_insurance", "pensions_social_security"]:
        assert unpriced not in prices.CPI_SERIES


def test_fetch_splits_requests_and_keeps_only_months(monkeypatch):
    requests = fake_api(monkeypatch)
    table = prices.fetch_prices(2024, 2025)
    assert all(len(body["seriesid"]) <= prices.API_MAX_SERIES for body in requests)
    assert sorted(s for body in requests for s in body["seriesid"]) == prices.SERIES
    assert list(table.columns) == ["series", "year", "month", "value"]
    assert set(table["series"]) == set(prices.SERIES) and table["value"].max() < 999
    one = table[table["series"] == prices.ALL_ITEMS]
    assert len(one[one["year"] == 2024]) == 12
    assert 10 not in set(one.loc[one["year"] == 2025, "month"])  # BLS's "-" for a month without data


def test_fetch_covers_long_spans_in_several_requests(monkeypatch):
    requests = fake_api(monkeypatch)
    table = prices.fetch_prices(2010, 2024, series=[prices.ALL_ITEMS])
    assert [(body["startyear"], body["endyear"]) for body in requests] == [("2010", "2019"), ("2020", "2024")]
    assert sorted(table["year"].unique()) == list(range(2010, 2025))


def test_fetch_reports_failures(monkeypatch):
    fake_api(monkeypatch, missing={prices.ALL_ITEMS})
    with pytest.raises(RuntimeError, match=prices.ALL_ITEMS):
        prices.fetch_prices(2024, 2024)
    monkeypatch.setattr(
        prices.urllib.request, "urlopen", lambda request, timeout: io.StringIO('{"status": "REQUEST_NOT_PROCESSED"}')
    )
    with pytest.raises(RuntimeError, match="REQUEST_NOT_PROCESSED"):
        prices.fetch_prices(2024, 2024)


def test_download_skips_what_is_already_there(monkeypatch, tmp_path, capsys):
    requests = fake_api(monkeypatch)
    prices.download_prices(tmp_path, 2023, 2024)
    assert len(requests) > 0 and prices.prices_path(tmp_path).exists()
    done = len(requests)
    prices.download_prices(tmp_path, 2023, 2024)
    assert len(requests) == done and "Already downloaded" in capsys.readouterr().out
    prices.download_prices(tmp_path, 2022, 2024)  # a wider span is fetched
    assert len(requests) > done


def test_load_prices_names_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="cex download"):
        prices.load_prices(tmp_path)


def test_annual_average_needs_most_of_the_year():
    table = pd.DataFrame(
        [{"series": "A", "year": 2024, "month": m, "value": float(m)} for m in range(1, 13)]
        + [{"series": "A", "year": 2025, "month": m, "value": 10.0} for m in range(1, 12)]  # one month missing
        + [{"series": "A", "year": 2026, "month": m, "value": 10.0} for m in range(1, 11)]  # a year in progress
    )
    annual = prices.annual_average(table)["A"]
    assert annual[2024] == pytest.approx(6.5) and annual[2025] == 10.0
    assert pd.isna(annual[2026])


def test_category_price_change(tmp_path):
    table = prices.load_prices(write_synthetic_prices(tmp_path, 2023, 2025).parent)
    change = prices.category_price_change(table, 2023, 2025)
    assert list(change.index) == list(prices.CPI_SERIES)
    assert change["groceries"] == pytest.approx((1 + BASE_INFLATION) ** 2)
    assert change["rented_dwellings"] == pytest.approx(
        (1 + ANNUAL_INFLATION[prices.CPI_SERIES["rented_dwellings"]]) ** 2
    )
    assert change["household_operations"] == change["furnishings"]  # priced with the same index
    with pytest.raises(KeyError, match="2026"):
        prices.category_price_change(table, 2023, 2026)
