"""Synthetic stand-ins for BLS's files so the pipeline can be tested without real data.

`write_synthetic_raw` writes `intrvwYY.zip` (quarterly FMLI, MTBI, and ITBI
files) and `stubs.zip` (the category tree) into a folder, laid out like the
real ones, quirks included:

* The 2024 release holds all five quarterly files, the first with an "x". Other
  years' releases start at Apr-Jun and their Jan-Mar files sit in the previous
  year's zip.
* Until Apr-Jun 2024 the FMLI food-at-home variable is FDHOME, built from the
  food-only grocery UCC. After that it is GROCER, built from all grocery
  purchases, and FDHOME is all zeros. The stub file lists only the food-only UCC.
* FMLI's summary variables leave out an item the stub file includes (laundry).
* Social Security deductions are in ITBI, as are credit card finance charges,
  which are asked only in a CU's fourth interview and carry a factor of 4.

The data follow relationships with known answers that the models should recover:

* The budget share of groceries is linear in log total spending, with slope
  GROCERY_SHARE_SLOPE. Total spending includes groceries, so it is endogenous;
  income is a valid instrument.
* Mean apparel spending has a constant income elasticity,
  APPAREL_INCOME_ELASTICITY, and most quarters have no apparel purchase.
"""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cex.load import COLUMNS, EXPENDITURE_CODES, REPLICATE_WEIGHTS
from cex.prices import ALL_ITEMS, CPI_SERIES, SERIES

YEAR = 2024
GROCERY_SHARE_SLOPE = -0.04
GROCERY_SHARE_INTERCEPT = 0.15 - GROCERY_SHARE_SLOPE * 11.2  # a 15% share at about $73k of annual spending
APPAREL_INCOME_ELASTICITY = 1.3
FOOD_SHARE_OF_GROCERIES = 0.8
TRIP_SHARE_OF_GROCERIES = 0.02  # groceries bought on trips, a separate UCC

# Releases whose zip holds all five quarterly files (the first with an "x"), as BLS's 2019 and 2024 do.
FIVE_FILE_RELEASES = {2019, 2024}


@dataclass
class Item:
    """One UCC other than the grocery ones, and how much the synthetic households spend on it."""

    ucc: str
    title: str
    path: tuple[str, ...]  # the groups above it in the stub file, below "Average annual expenditures"
    fmli: str | None  # the FMLI summary variable that includes it, if any
    base: float  # mean spending per 3-month recall period at log income 11
    elasticity: float  # with respect to income
    incidence: float  # chance of any spending in a quarter
    file: str = "mtbi"
    factor: int = 1


HOUSING, TRANSPORT, HEALTH, FUN = "Housing", "Transportation", "Healthcare", "Entertainment"
INSURANCE = "Personal insurance and pensions"
ITEMS = [
    Item("790410", "Meals at restaurants", ("Food", "Food away from home"), "FDAWAY", 1_100, 0.6, 0.9),
    Item("790330", "Beer, wine, other alcohol", ("Alcoholic beverages", "At home"), "ALCBEV", 100, 0.6, 0.5),
    Item("790420", "Alcohol at restaurants", ("Alcoholic beverages", "Away from home"), "ALCBEV", 50, 0.8, 0.3),
    Item(
        "220311",
        "Mortgage interest",
        (HOUSING, "Shelter", "Owned dwellings", "Mortgage interest and charges"),
        "HOUS",
        1_800,
        0.6,
        0.5,
    ),
    Item("210110", "Rent", (HOUSING, "Shelter", "Rented dwellings"), "HOUS", 1_500, 0.3, 0.4),
    Item("210210", "Lodging on trips", (HOUSING, "Shelter", "Other lodging"), "HOUS", 200, 1.2, 0.2),
    Item("260211", "Electricity", (HOUSING, "Utilities, fuels, and public services"), "HOUS", 900, 0.3, 1.0),
    Item("340211", "Babysitting", (HOUSING, "Household operations"), "HOUS", 400, 0.7, 0.3),
    Item("290110", "Furniture", (HOUSING, "Household furnishings and equipment"), "HOUS", 500, 0.9, 0.3),
    Item(
        "720110", "Adult clothing", ("Apparel and services", "Clothing"), "APPAR", 350, APPAREL_INCOME_ELASTICITY, 0.4
    ),
    Item(
        "440210",
        "Laundry and dry cleaning",
        ("Apparel and services", "Other apparel products and services"),
        None,  # in BLS's published category, but not in its FMLI summary variable
        50,
        APPAREL_INCOME_ELASTICITY,
        0.3,
    ),
    Item("450110", "New cars", (TRANSPORT, "Vehicle purchases (net outlay)"), "TRANS", 1_200, 1.0, 0.05),
    Item("470111", "Gasoline", (TRANSPORT, "Gasoline and other fuels"), "TRANS", 600, 0.4, 0.9),
    Item("500110", "Vehicle insurance", (TRANSPORT, "Other vehicle expenses"), "TRANS", 900, 0.6, 0.9),
    Item("530110", "Airline fares", (TRANSPORT, "Public and other transportation"), "TRANS", 300, 1.2, 0.2),
    Item("580111", "Health insurance premiums", (HEALTH, "Health insurance"), "HEALTH", 1_000, 0.4, 0.8),
    Item("560110", "Physician's services", (HEALTH, "Medical services"), "HEALTH", 300, 0.6, 0.4),
    Item("540000", "Prescription drugs", (HEALTH,), "HEALTH", 100, 0.3, 0.5),
    Item("550110", "Eyeglasses and contact lenses", (HEALTH, "Medical supplies"), "HEALTH", 100, 0.5, 0.2),
    Item("620111", "Club memberships", (FUN, "Fees and admissions"), "ENTERT", 250, 1.0, 0.5),
    Item("270310", "Cable television", (FUN, "Audio and visual equipment and services"), "ENTERT", 250, 0.4, 0.8),
    Item("610320", "Pet food", (FUN, "Pets, toys, hobbies, and playground equipment"), "ENTERT", 200, 0.7, 0.4),
    Item(
        "600210",
        "Boats",
        (FUN, "Other entertainment supplies, equipment, and services"),
        "ENTERT",
        100,
        1.5,
        0.05,
    ),
    Item("650310", "Personal care services", ("Personal care products and services",), "PERSCA", 250, 0.6, 0.7),
    Item("590230", "Books", ("Reading",), "READ", 30, 0.6, 0.3),
    Item("670110", "College tuition", ("Education",), "EDUCA", 400, 0.6, 0.15),
    Item("630110", "Cigarettes", ("Tobacco products and smoking supplies",), "TOBACC", 90, 0.0, 0.15),
    Item("680110", "Legal fees", ("Miscellaneous",), "MISC", 250, 0.6, 0.3),
    # Asked in the fourth interview only (see make_quarter), about the past 12 months.
    Item("005420", "Finance charges for credit cards", ("Miscellaneous",), "MISC", 60, 0.3, 0.6, "itbi", factor=4),
    Item("800821", "Contributions to charities", ("Cash contributions",), "CASHCO", 550, 0.9, 0.5),
    Item("700110", "Life insurance", (INSURANCE, "Life and other personal insurance"), "PERINS", 200, 0.8, 0.5),
    Item(
        "800940",
        "Deductions for Social Security",
        (INSURANCE, "Retirement, pensions, and Social Security"),
        "PERINS",
        2_200,
        1.0,
        0.85,
        "itbi",
    ),
]
FOURTH_INTERVIEW_ONLY = "005420"
RENT_UCC, MORTGAGE_UCC = "210110", "220311"
assert {"TOTEXP", "FDHOME"} | {item.fmli for item in ITEMS if item.fmli} == set(EXPENDITURE_CODES)

GROCERY_UCC, FOOD_ONLY_UCC, TRIP_GROCERY_UCC = "790210", "790240", "190904"
# In MTBI but not spending: must be ignored.
RENTAL_VALUE_UCC, WAGES_UCC = "910050", "900000"


def stub_text() -> str:
    """A stub file in BLS's fixed-width layout: sections before and after the spending tree, like the real one."""

    def line(level: int, title: str, code: str, kind: str, factor: int = 1, section: str = "EXPEND") -> str:
        text = "  " * (level - 1) + title
        first, rest = text[:60], text[60:]
        out = f"1  {level}  {first:<60}  {code:<8}  {kind}  {factor}  {section}"
        if rest:  # a long title continues on a type-2 line
            out += "\n2  " + " " * 3 + rest
        return out

    lines = [
        line(1, "Interview stub parameter file", "HEADINTR", "H", section="CUCHARS"),
        line(1, "Consumer unit characteristics:", "TITLECU", "T", section="CUCHARS"),
        line(2, "Income before taxes", "INCBFTAX", "G", section="CUCHARS"),
        line(3, "Income before taxes", "980000", "I", section="CUCHARS"),
        line(1, "Average annual expenditures", "TOTALEXP", "G"),
        line(2, "Food", "FOODTOTL", "G"),
        line(3, "Food at home", "FOODHOME", "G", section="FOOD"),
        line(4, "Food and non alcoholic beverages", FOOD_ONLY_UCC, "I", section="FOOD"),
        line(
            4,
            "Food prepared by consumer unit on out of town trips, a title long enough to wrap",
            TRIP_GROCERY_UCC,
            "I",
            section="FOOD",
        ),
    ]
    written: set[tuple[str, ...]] = {("Food",), ("Food", "Food at home")}
    for item in ITEMS:
        for depth in range(1, len(item.path) + 1):
            group = item.path[:depth]
            if group not in written:
                written.add(group)
                lines.append(line(depth + 1, group[-1], f"G{len(written):05d}", "G"))
        lines.append(line(len(item.path) + 2, item.title, item.ucc, "I", item.factor))
    lines += [
        line(1, "Sources of pretax income:", "TITLESI", "T", section="INCOME"),
        line(2, "Wages and salaries", WAGES_UCC, "I", section="INCOME"),
        line(1, "Addenda:", "TITLEADD", "T", section="ADDENDA"),
        line(2, "Rental equivalence of owned home", RENTAL_VALUE_UCC, "I", section="ADDENDA"),
    ]
    return "\r\n".join(lines) + "\r\n"


def quarterly_stems(year: int) -> list[tuple[str, int, int]]:
    """(file suffix, interview year, interview quarter) for the five quarterly files covering a year."""
    yy, next_yy = year % 100, (year + 1) % 100
    first = f"{yy:02d}1x" if year in FIVE_FILE_RELEASES else f"{yy:02d}1"
    return [
        (first, year, 1),
        (f"{yy:02d}2", year, 2),
        (f"{yy:02d}3", year, 3),
        (f"{yy:02d}4", year, 4),
        (f"{next_yy:02d}1", year + 1, 1),
    ]


def weighted_rank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    rank = np.empty_like(weights)
    rank[order] = np.cumsum(weights[order]) / weights.sum()
    return rank


def make_consumer_units(rng: np.random.Generator, n: int, year: int) -> pd.DataFrame:
    cu = pd.DataFrame({"cuid": 100_000 * (year % 100) + np.arange(n)})
    cu["interviews"] = rng.integers(1, 5, n)
    cu["first_file"] = rng.integers(0, 6 - cu["interviews"])  # stay within the year's 5 files
    cu["month_in_quarter"] = rng.integers(0, 3, n)
    cu["income"] = np.exp(rng.normal(11, 0.7, n))
    cu["weight"] = rng.uniform(5_000, 15_000, n)
    # Replicate weights like BLS's: each replicate keeps a random half of the CUs at about double weight.
    in_replicate = rng.integers(0, 2, (n, len(REPLICATE_WEIGHTS)))
    cu = cu.join(pd.DataFrame(2.0 * in_replicate, columns=[f"factor_{col}" for col in REPLICATE_WEIGHTS]))
    cu["grocery_taste"] = rng.normal(0, 0.015, n)
    for item in ITEMS:
        cu[f"taste_{item.ucc}"] = rng.normal(0, 0.3, n)
    cu["AGE_REF"] = rng.integers(18, 90, n)
    cu["SEX_REF"] = rng.integers(1, 3, n)
    cu["EDUC_REF"] = rng.choice([10, 11, 12, 13, 14, 15, 16, 17], n)
    cu["FAM_SIZE"] = rng.integers(1, 7, n)
    cu["PERSLT18"] = rng.integers(0, 3, n)
    cu["PERSOT64"] = rng.integers(0, 2, n)
    cu["NO_EARNR"] = rng.integers(0, 3, n)
    cu["VEHQ"] = rng.integers(0, 4, n)
    cu["REGION"] = rng.integers(1, 5, n)
    cu["BLS_URBN"] = rng.integers(1, 3, n)
    cu["CUTENURE"] = rng.integers(1, 7, n)
    return cu


TRAITS = [
    "AGE_REF",
    "SEX_REF",
    "EDUC_REF",
    "FAM_SIZE",
    "PERSLT18",
    "PERSOT64",
    "NO_EARNR",
    "VEHQ",
    "REGION",
    "BLS_URBN",
    "CUTENURE",
]


def make_quarter(rng: np.random.Generator, cu: pd.DataFrame, file_index: int, interview_year: int, quarter: int):
    """The FMLI, MTBI, and ITBI tables for one quarterly file."""
    active = cu[(cu["first_file"] <= file_index) & (file_index < cu["first_file"] + cu["interviews"])]
    n = len(active)
    interview_number = (file_index - active["first_file"] + 1).to_numpy()
    fmli = pd.DataFrame({col: 0.0 for col in COLUMNS}, index=range(n))
    fmli["NEWID"] = active["cuid"].to_numpy() * 10 + interview_number
    fmli["QINTRVYR"] = interview_year
    fmli["QINTRVMO"] = (3 * (quarter - 1) + 1 + active["month_in_quarter"]).to_numpy()
    fmli["FINLWT21"] = active["weight"].to_numpy()
    for col in REPLICATE_WEIGHTS:
        fmli[col] = fmli["FINLWT21"] * active[f"factor_{col}"].to_numpy()
    fmli["WTREP05"] = fmli["WTREP05"].replace(0.0, np.nan)  # excluded from a replicate can also be a blank
    for col in TRAITS:
        fmli[col] = active[col].to_numpy()
    # Owners (CUTENURE 1-3) report what their home would rent for per month; it is blank for the rest.
    owner = active["CUTENURE"].to_numpy() <= 3
    fmli["RENTEQVX"] = np.where(owner, np.round(0.012 * active["income"].to_numpy() + 600), np.nan)

    income = active["income"].to_numpy()
    imputations = income[:, None] * np.exp(rng.normal(0, 0.05, (n, 5)))
    for m in range(5):
        fmli[f"FINCBTX{m + 1}"] = imputations[:, m]
        fmli[f"INC_RNK{m + 1}"] = weighted_rank(imputations[:, m], fmli["FINLWT21"].to_numpy())
    fmli["FINCBTXM"] = imputations.mean(axis=1)
    fmli["INC_RNKM"] = weighted_rank(fmli["FINCBTXM"].to_numpy(), fmli["FINLWT21"].to_numpy())

    # Recorded cost of each UCC over the 3-month recall period.
    cost = {}
    for item in ITEMS:
        mean = item.base * np.exp(item.elasticity * (np.log(income) - 11) + active[f"taste_{item.ucc}"].to_numpy())
        bought = rng.random(n) < item.incidence
        noise = np.exp(rng.normal(-0.045, 0.3, n))  # mean 1
        cost[item.ucc] = mean * bought / item.incidence * noise
    cost[FOURTH_INTERVIEW_ONLY] = cost[FOURTH_INTERVIEW_ONLY] * (interview_number == 4)
    cost[RENT_UCC] = cost[RENT_UCC] * (active["CUTENURE"].to_numpy() == 4)  # only renters pay rent
    cost[MORTGAGE_UCC] = cost[MORTGAGE_UCC] * owner
    # What counts toward published spending: the cost times the stub file's factor.
    other = sum(cost[item.ucc] * item.factor for item in ITEMS)
    # Groceries: share = a + b * log(annual total) + taste, where the total includes groceries.
    taste = active["grocery_taste"].to_numpy()
    share = np.full(n, 0.1)
    for _ in range(50):  # fixed point; converges fast because the slope is small
        share = GROCERY_SHARE_INTERCEPT + GROCERY_SHARE_SLOPE * np.log(4 * other / (1 - share)) + taste
    groceries = share / (1 - share) * other
    cost[TRIP_GROCERY_UCC] = TRIP_SHARE_OF_GROCERIES * groceries
    cost[GROCERY_UCC] = groceries - cost[TRIP_GROCERY_UCC]
    grocer_era = (interview_year, quarter) >= (2024, 2)
    if not grocer_era:
        cost[FOOD_ONLY_UCC] = FOOD_SHARE_OF_GROCERIES * cost[GROCERY_UCC]

    # FMLI summary variables, split between the previous and the current calendar quarter.
    summary = {code: np.zeros(n) for code in EXPENDITURE_CODES}
    for item in ITEMS:
        if item.fmli:
            summary[item.fmli] += cost[item.ucc]
    summary["FDHOME"] = cost[GROCERY_UCC if grocer_era else FOOD_ONLY_UCC] + cost[TRIP_GROCERY_UCC]
    summary["TOTEXP"] = sum(summary[code] for code in EXPENDITURE_CODES if code != "TOTEXP")
    current_quarter = (fmli["QINTRVMO"] - 1) % 3 / 3  # share of the recall months in the interview's quarter
    for code, total in summary.items():
        fmli[f"{code}CQ"] = total * current_quarter
        fmli[f"{code}PQ"] = total * (1 - current_quarter)
    fmli = fmli.copy()  # consolidate after many column assignments
    if grocer_era:
        # Like the real files: food at home moves to GROCER* and the old FDHOME* columns remain, all zero.
        fmli["GROCERPQ"], fmli["GROCERCQ"] = fmli["FDHOMEPQ"], fmli["FDHOMECQ"]
        fmli["FDHOMEPQ"], fmli["FDHOMECQ"] = 0.0, 0.0

    # Item-level records: a third of the cost in each of the 3 months before the interview month.
    cost[RENTAL_VALUE_UCC] = np.full(n, 4_500.0)
    cost[WAGES_UCC] = income / 4
    file_of = {item.ucc: item.file for item in ITEMS} | {WAGES_UCC: "itbi"}
    records = {"mtbi": [], "itbi": []}
    months = 12 * fmli["QINTRVYR"].to_numpy() + fmli["QINTRVMO"].to_numpy() - 1  # months since year 0
    for ucc, total in cost.items():
        spent = total != 0
        for back in (1, 2, 3):
            month = months[spent] - back
            records[file_of.get(ucc, "mtbi")].append(
                pd.DataFrame(
                    {
                        "NEWID": fmli["NEWID"].to_numpy()[spent],
                        "REF_MO": month % 12 + 1,
                        "REF_YR": month // 12,
                        "UCC": ucc,
                        "COST": total[spent] / 3,
                    }
                )
            )
    mtbi = pd.concat(records["mtbi"], ignore_index=True)
    mtbi.insert(1, "SEQNO", 0)
    mtbi["PUBFLAG"] = 2
    itbi = pd.concat(records["itbi"], ignore_index=True)
    itbi = itbi.rename(columns={"REF_MO": "REFMO", "REF_YR": "REFYR", "COST": "VALUE"})
    itbi["VALUE_"] = ""
    return fmli, mtbi, itbi


def _zip_name(year: int) -> str:
    return f"intrvw{year % 100:02d}.zip"


def write_synthetic_raw(raw_dir: Path, years: list[int] = (YEAR,), seed: int = 0, n_cu: int = 4_000) -> Path:
    """Write stubs.zip and the Interview releases needed for `years` into `raw_dir`; return `raw_dir`."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_dir / "stubs.zip", "w") as zf:
        for year in years:
            zf.writestr(f"stubs/CE-HG-Inter-{year}.txt", stub_text().encode("latin-1"))
            zf.writestr(f"stubs/CE-HG-Integ-{year}.txt", "not read")

    for year in sorted(years):
        rng = np.random.default_rng(seed + year)
        cu = make_consumer_units(rng, n_cu, year)
        folder = f"intrvw{year % 100:02d}"
        for i, (stem, interview_year, quarter) in enumerate(quarterly_stems(year)):
            # A release that starts at Apr-Jun: its Jan-Mar files are the last ones of the previous release.
            in_previous = i == 0 and year not in FIVE_FILE_RELEASES
            target = raw_dir / _zip_name(year - 1 if in_previous else year)
            if in_previous and target.exists():
                with zipfile.ZipFile(target) as zf:
                    if any(name.endswith(f"fmli{stem}.csv") for name in zf.namelist()):
                        continue  # written as the last quarter of the previous year
            fmli, mtbi, itbi = make_quarter(rng, cu, i, interview_year, quarter)
            if i == 1:
                fmli.columns = fmli.columns.str.lower()  # some releases use lowercase headers
                fmli.loc[:5, "region"] = np.nan  # written as "." below, like the real files
            with zipfile.ZipFile(target, "a") as zf:
                zf.writestr(f"{folder}/fmli{stem}.csv", fmli.to_csv(index=False, na_rep="."))
                zf.writestr(f"{folder}/mtbi{stem}.csv", mtbi.to_csv(index=False))
                zf.writestr(f"{folder}/itbi{stem}.csv", itbi.to_csv(index=False))
    return raw_dir


# Synthetic price indexes: every series starts at 100 in January of the first year and grows at a
# constant annual rate. Rent rises fastest, which is what makes renters' inflation the highest.
BASE_INFLATION = 0.03
ANNUAL_INFLATION = {CPI_SERIES["rented_dwellings"]: 0.08, CPI_SERIES["gasoline"]: -0.02, ALL_ITEMS: 0.035}


def write_synthetic_prices(raw_dir: Path, first_year: int = YEAR, last_year: int = YEAR + 1) -> Path:
    """Write prices.csv with twelve months of every series for each year; return its path."""
    rows = []
    for series in SERIES:
        rate = ANNUAL_INFLATION.get(series, BASE_INFLATION)
        for year in range(first_year, last_year + 1):
            for month in range(1, 13):
                value = 100 * (1 + rate) ** (year - first_year + (month - 1) / 12)
                rows.append({"series": series, "year": year, "month": month, "value": value})
    path = Path(raw_dir) / "prices.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def rewrite_member(zip_path: Path, suffix: str, change) -> None:
    """Replace the csv in `zip_path` whose name ends with `suffix` by `change(its DataFrame)`."""
    with zipfile.ZipFile(zip_path) as zf:
        contents = {name: zf.read(name) for name in zf.namelist()}
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in contents.items():
            if name.endswith(suffix):
                data = change(pd.read_csv(io.BytesIO(data), dtype=str)).to_csv(index=False).encode()
            zf.writestr(name, data)


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory) -> Path:
    """A raw-data folder with the synthetic 2024 release and stub file."""
    return write_synthetic_raw(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def synthetic_year(synthetic_raw):
    from cex.dataset import load_year

    return load_year(YEAR, synthetic_raw)
