"""Assemble one calendar year of Interview data: households from FMLI, spending from the item-level records."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cex import spending, stubs
from cex.clean import add_spending, clean_fmli
from cex.load import load_fmli, load_ucc_records


@dataclass
class YearData:
    year: int
    # One row per interview: identifiers, timing, weights, income, household traits,
    # and <category>_annual / <category>_in_year spending for every major and detail category.
    interviews: pd.DataFrame
    records: pd.DataFrame  # the item-level spending records behind it (load.load_ucc_records)
    tree: pd.DataFrame  # the year's category tree (stubs.expenditure_tree)
    detail: list[str]  # names of the detail categories
    fmli_check: pd.DataFrame  # spending.compare_with_fmli


def load_year(year: int, raw_dir: Path) -> YearData:
    """Read and combine everything for `year`. Raises if the rebuilt spending looks incomplete."""
    tree = stubs.expenditure_tree(year, raw_dir)
    interviews = clean_fmli(load_fmli(year, raw_dir), year)
    records = load_ucc_records(year, raw_dir, uccs=set(tree.loc[tree["is_item"], "ucc"]))
    interviews = add_spending(interviews, spending.interview_spending(records, tree, interviews["newid"], year))
    fmli_check = spending.compare_with_fmli(interviews)
    spending.check_coverage(fmli_check)
    return YearData(year, interviews, records, tree, stubs.detail_names(tree), fmli_check)
