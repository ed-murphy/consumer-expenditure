"""Compare estimates from the real data with BLS's published tables, for every year that is downloaded.

Needs the files in data/raw/ (`cex download --years 2019-2024`); years without them are skipped.

Published figures: Table A of the annual "Consumer Expenditures" news releases,
https://www.bls.gov/news.release/archives/cesan_09092021.htm (2019, 2020),
https://www.bls.gov/news.release/archives/cesan_09252024.htm (2021-2023), and
https://www.bls.gov/news.release/archives/cesan_12192025.htm (2024).

BLS's tables combine the Interview and Diary surveys, while this pipeline uses
the Interview alone, so the two agree closely only where BLS draws on the
Interview. Cash contributions and personal insurance and pensions come entirely
from it and match to the dollar in most years, which checks the weights, the
calendar-year scoping, the choice of quarterly files, and the item-level
rebuild all at once. Total spending is looser because food, apparel, and some
household items are published from the Diary. The tolerances were set a little
wider than the gaps observed when this test was written.
"""

from pathlib import Path

import pytest

from cex.dataset import load_year
from cex.load import quarterly_files
from cex.stubs import read_stub
from cex.summarize import annual_means

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# year: published consumer units (thousands), income before taxes, total, transportation,
# cash contributions, personal insurance and pensions.
PUBLISHED = {
    2019: (132_242, 82_852, 63_036, 10_742, 1_995, 7_165),
    2020: (131_234, 84_352, 61_334, 9_826, 2_283, 7_246),
    2021: (133_595, 87_432, 66_928, 10_961, 2_415, 7_873),
    2022: (134_090, 94_003, 72_967, 12_295, 2_755, 8_742),
    2023: (134_556, 101_805, 77_280, 13_174, 2_378, 9_556),
    2024: (135_760, 104_207, 78_535, 13_318, 2_292, 9_797),
}


def _available(year: int) -> bool:
    try:
        for kind in ("fmli", "mtbi", "itbi"):
            quarterly_files(year, RAW, kind)
        read_stub(year, RAW)
    except FileNotFoundError:
        return False
    return True


@pytest.mark.realdata
@pytest.mark.parametrize("year", PUBLISHED)
def test_matches_published_table(year):
    if not _available(year):
        pytest.skip(f"{year} files not downloaded to {RAW}")
    consumer_units, income, total, transportation, cash, insurance = PUBLISHED[year]
    means = annual_means(load_year(year, RAW).interviews)["mean"]
    assert means["consumer_units"] / 1000 == pytest.approx(consumer_units, rel=0.001)
    assert means["income_before_tax"] == pytest.approx(income, rel=0.01)
    assert means["total"] == pytest.approx(total, rel=0.04)
    assert means["transportation"] == pytest.approx(transportation, rel=0.025)
    assert means["insurance_pensions"] == pytest.approx(insurance, rel=0.005)
    # The 2024 public file is about 1% below the published figure; earlier years match to the dollar.
    assert means["cash_contributions"] == pytest.approx(cash, rel=0.015 if year == 2024 else 0.001)
