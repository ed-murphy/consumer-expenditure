"""Price series from BLS, to put next to the spending data.

Monthly consumer price indexes (CPI-U, U.S. city average, not seasonally
adjusted) for the item group that best matches each detail spending category
(CPI_SERIES), plus the all-items index. An index says how prices changed over
time; its level means nothing by itself.

`fetch_prices` gets them from BLS's public API, which needs no registration or
contact details for a few requests a day, and `cex download` saves them as
`data/raw/prices.csv`. Everything else here reads that file.
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

API_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
API_MAX_SERIES = 25  # per request, without a registration key
API_MAX_YEARS = 10

ALL_ITEMS = "CUUR0000SA0"

# CPI-U item group for each detail spending category (see stubs.detail_names).
# These are the closest matches, not exact ones: the CPI groups items its own way.
# No price index applies to cash contributions, life insurance, or pension and
# Social Security contributions, which the CPI leaves out as well.
CPI_SERIES = {
    "groceries": "CUUR0000SAF11",  # food at home
    "food_away": "CUUR0000SEFV",
    "alcohol_home": "CUUR0000SEFW",
    "alcohol_away": "CUUR0000SEFX",
    "owned_dwellings": "CUUR0000SEHC",  # owners' equivalent rent of residences
    "rented_dwellings": "CUUR0000SEHA",  # rent of primary residence
    "other_lodging": "CUUR0000SEHB",  # lodging away from home
    "utilities": "CUUR0000SAH2",  # fuels and utilities
    "household_operations": "CUUR0000SAH3",  # household furnishings and operations
    "furnishings": "CUUR0000SAH3",
    "apparel": "CUUR0000SAA",
    "vehicle_purchases": "CUUR0000SETA",  # new and used motor vehicles
    "gasoline": "CUUR0000SETB",  # motor fuel
    "other_vehicle": "CUUR0000SAS4",  # transportation services: insurance, repair, fees (and fares)
    "public_transportation": "CUUR0000SETG",
    # Not the CPI's own health insurance index: that one prices only insurers' margins and swings
    # wildly, while households pay whole premiums, most of which buy medical care.
    "health_insurance": "CUUR0000SAM",  # medical care
    "medical_services": "CUUR0000SAM2",
    "prescription_drugs": "CUUR0000SEMF01",
    "medical_supplies": "CUUR0000SAM1",  # medical care commodities
    "fees_admissions": "CUUR0000SERF",  # other recreation services
    "audio_visual": "CUUR0000SERA",  # video and audio
    "pets_toys_hobbies": "CUUR0000SERB",  # pets, pet products and services
    "other_entertainment": "CUUR0000SERC",  # sporting goods
    "personal_care": "CUUR0000SAG1",
    "reading": "CUUR0000SERG",  # recreational reading materials
    "education": "CUUR0000SEEB",  # tuition, other school fees, and childcare
    "tobacco": "CUUR0000SEGA",
    "miscellaneous": "CUUR0000SEGD",  # miscellaneous personal services: legal, funeral, financial
}

SERIES = sorted({ALL_ITEMS, *CPI_SERIES.values()})


def prices_path(raw_dir: Path) -> Path:
    return Path(raw_dir) / "prices.csv"


def _request(series: list[str], start_year: int, end_year: int) -> list[dict]:
    body = json.dumps({"seriesid": series, "startyear": str(start_year), "endyear": str(end_year)}).encode()
    request = urllib.request.Request(API_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.load(response)
    if reply.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API request failed: {reply.get('status')}: {reply.get('message')}")
    return reply["Results"]["series"]


def fetch_prices(start_year: int, end_year: int, series: list[str] = SERIES) -> pd.DataFrame:
    """Monthly values of `series` from the BLS API: columns series, year, month, value."""
    rows = []
    for first_year in range(start_year, end_year + 1, API_MAX_YEARS):
        last_year = min(first_year + API_MAX_YEARS - 1, end_year)
        for i in range(0, len(series), API_MAX_SERIES):
            for result in _request(series[i : i + API_MAX_SERIES], first_year, last_year):
                for point in result["data"]:
                    # M01-M12 are months (M13 would be an annual average). BLS reports "-" for a month
                    # with no data, such as October 2025.
                    if point["period"][0] == "M" and point["period"] != "M13" and point["value"] != "-":
                        rows.append(
                            {
                                "series": result["seriesID"],
                                "year": int(point["year"]),
                                "month": int(point["period"][1:]),
                                "value": float(point["value"]),
                            }
                        )
    table = pd.DataFrame(rows).sort_values(["series", "year", "month"], ignore_index=True)
    missing = sorted(set(series) - set(table["series"]))
    if missing:
        raise RuntimeError(f"The BLS API returned no data for: {missing}")
    return table


def download_prices(raw_dir: Path, start_year: int, end_year: int, overwrite: bool = False) -> Path:
    """Save the series this package uses, for `start_year` through `end_year`, as prices.csv in `raw_dir`."""
    dest = prices_path(raw_dir)
    if dest.exists() and not overwrite:
        have = pd.read_csv(dest)
        if have["year"].min() <= start_year and have["year"].max() >= end_year and set(SERIES) <= set(have["series"]):
            print(f"Already downloaded: {dest}")
            return dest
    print(f"Downloading {len(SERIES)} price series for {start_year}-{end_year} from {API_URL} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetch_prices(start_year, end_year).to_csv(dest, index=False)
    return dest


def load_prices(raw_dir: Path) -> pd.DataFrame:
    path = prices_path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `cex download` (it fetches the price series too).")
    return pd.read_csv(path)


def annual_average(prices: pd.DataFrame, min_months: int = 11) -> pd.DataFrame:
    """Average of the monthly values by year: one row per year, one column per series.

    A series-year with fewer than `min_months` months of data is left missing,
    so that a year still in progress is not mistaken for a full one.
    """
    by_year = prices.groupby(["year", "series"])["value"].agg(["mean", "count"])
    return by_year["mean"].where(by_year["count"] >= min_months).unstack("series")


def category_price_change(prices: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    """Ratio of each category's average price index in `end_year` to that in `start_year` (index: category)."""
    annual = annual_average(prices)
    for year in (start_year, end_year):
        if year not in annual.index or annual.loc[year, list(set(CPI_SERIES.values()))].isna().any():
            raise KeyError(f"prices.csv has no full year of price indexes for {year}; run `cex download --overwrite`.")
    ratio = annual.loc[end_year] / annual.loc[start_year]
    return pd.Series({category: ratio[series] for category, series in CPI_SERIES.items()}, name="price_change")
