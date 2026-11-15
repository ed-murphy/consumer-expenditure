"""Command-line entry point: `uv run cex download ...` and `uv run cex run ...`."""

import argparse
import datetime
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cex import inflation, plots, spending
from cex.clean import CATEGORIES, household_year
from cex.cpi import cpi
from cex.dataset import YearData, load_year
from cex.download import download_interview, download_stubs
from cex.load import needs_previous_release, quarterly_files
from cex.model import EngelResults, fit_engel, fit_quadratic
from cex.prices import category_price_change, download_prices, load_prices
from cex.stubs import read_stub
from cex.summarize import annual_means, group_means, trends


def default_root() -> Path:
    """Where data/ and outputs/ live: $CEX_HOME, else this source checkout, else the current directory."""
    if os.environ.get("CEX_HOME"):
        return Path(os.environ["CEX_HOME"])
    checkout = Path(__file__).resolve().parents[2]
    return checkout if (checkout / "pyproject.toml").exists() else Path.cwd()


def parse_years(spec: str) -> list[int]:
    """'2024' -> [2024]; '2020-2024' -> [2020, ..., 2024]; '2019,2022-2024' -> [2019, 2022, 2023, 2024]."""
    years: set[int] = set()
    try:
        for part in spec.split(","):
            start, dash, end = part.strip().partition("-")
            first, last = int(start), int(end) if dash else int(start)
            if first > last:
                raise ValueError
            years.update(range(first, last + 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid year list {spec!r}; use e.g. 2024, 2020-2024, or 2019,2022-2024"
        ) from None
    return sorted(years)


def _print(title: str, table: pd.DataFrame, decimals: int = 2) -> None:
    with pd.option_context(
        "display.float_format", f"{{:,.{decimals}f}}".format, "display.width", 160, "display.max_columns", None
    ):
        print(f"\n{title}\n")
        print(table)


@dataclass
class YearResults:
    data: YearData
    households: pd.DataFrame
    means: pd.DataFrame
    groups: pd.DataFrame  # group means for the major and the detail categories
    engel: EngelResults
    detail_engel: EngelResults
    inflation: pd.DataFrame | None  # inflation.year_link, or None without price data for the following year


def _print_fmli_check(data: YearData) -> None:
    """Where BLS's ready-made summary variables depart from spending rebuilt from the item-level records."""
    check = data.fmli_check
    ratio = check.pivot(index="category", columns="source_file", values="ratio").reindex(CATEGORIES)
    ratio = ratio[list(dict.fromkeys(check["source_file"]))]
    off = ratio[((ratio - 1).abs() > 0.02).any(axis=1)]
    _print(
        "Spending rebuilt from item-level records, relative to BLS's FMLI summary variable, by quarterly file\n"
        "(categories within 2% in every quarter are not shown; see cex.spending for why the rest differ)",
        off,
    )


def _percent(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return table.assign(**{col: 100 * table[col] for col in columns})


def run_year(
    year: int, data_dir: Path, output_dir: Path, make_plots: bool = True, prices: pd.DataFrame | None = None
) -> YearResults:
    """Load, clean, summarize, and model one year."""
    print(f"\n=== {year} ===\nLoading the quarterly FMLI, MTBI, and ITBI files ...")
    data = load_year(year, data_dir / "raw")
    df = data.interviews
    print(
        f"  {len(df):,} interviews of {df['cuid'].nunique():,} consumer units, "
        f"{len(data.records):,} item-level spending records"
    )
    processed_path = data_dir / "processed" / f"interviews_{year}.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path)
    print(f"  -> {processed_path}")

    out = output_dir / str(year)
    out.mkdir(parents=True, exist_ok=True)
    data.fmli_check.to_csv(out / "fmli_check.csv", index=False)
    _print_fmli_check(data)

    means = annual_means(df)
    means.to_csv(out / "annual_means.csv")
    shown = means.assign(share=100 * means["share"]).rename(columns={"share": "% of total"})
    _print(
        f"Average annual values per consumer unit, {year} (Interview survey only)\n"
        "Compare with https://www.bls.gov/cex/tables.htm",
        shown[["mean", "mean_se", "% of total"]],
    )
    spending.tree_means(data.records, data.tree, df, year).to_csv(out / "detail_means.csv", index=False)

    categories = list(dict.fromkeys([*CATEGORIES, *data.detail]))
    groups = group_means(df, categories=categories)
    groups.to_csv(out / "group_means.csv", index=False)
    by_quintile = groups[(groups["grouping"] == "income_quintile") & (groups["item"] == "total")]
    _print("Total spending by income quintile", by_quintile.set_index("group")[["mean", "mean_se"]], decimals=0)

    households = household_year(df)
    print(f"\nFitting Engel curves on {len(households):,} households (5 income imputations x 45 weights) ...")
    engel = fit_engel(households)
    engel.elasticities.to_csv(out / "elasticities.csv")
    engel.coefficients.to_csv(out / "engel_coefficients.csv")
    columns = {
        "budget_share": "share",
        "expenditure_elasticity": "vs_spending",
        "expenditure_elasticity_se": "se",
        "classification": "class",
        "income_elasticity": "vs_income",
        "income_elasticity_se": "income_se",
    }
    _print(
        f"Elasticities (n={engel.n:,} households). vs_spending: < 1 necessity, > 1 luxury.\n"
        "vs_income: elasticity with respect to current income; compare with the row for total, not with 1.",
        engel.elasticities[list(columns)].rename(columns=columns),
        decimals=3,
    )
    fs = engel.first_stage
    print(
        f"\nFirst stage: log total spending on log income, coef {fs['coef']:.3f} (se {fs['se']:.3f}, F {fs['F']:,.0f})"
    )

    detail_engel = fit_engel(households, categories=data.detail, income_model=False)
    detail_engel.elasticities.drop(columns=[c for c in detail_engel.elasticities if c.startswith("income_")]).to_csv(
        out / "detail_elasticities.csv"
    )

    curvature = fit_quadratic(households)
    curvature.to_csv(out / "engel_curvature.csv")
    curved = curvature[curvature["curvature_t"].abs() > 1.96]
    shown = [c for c in curvature if c.startswith("elasticity_p") and not c.endswith("_se")]
    _print(
        "Categories whose elasticity changes with the size of the budget (quadratic model, |t| > 1.96).\n"
        "Elasticity at the 10th, 50th, and 90th percentile of total spending:",
        curved[["curvature_t", *shown]],
    )

    link = None
    if prices is not None:
        try:
            link = inflation.year_link(df, prices, year)
        except KeyError as err:
            print(f"\nNo inflation by group for {year}-{year + 1}: {err.args[0]}")
        else:
            link.to_csv(out / "inflation.csv", index=False)
            _print(
                f"Inflation {year}-{year + 1} by group, in %, with each group's {year} basket at {year + 1} prices.\n"
                "gap: percentage points above or below all consumer units",
                _percent(link, ["inflation", "inflation_se", "gap", "gap_se"]).set_index(["grouping", "group"]),
            )

    if make_plots:
        plots.plot_elasticities(engel, out / "elasticities.png", year)
        plots.plot_detail_elasticities(detail_engel, out / "detail_elasticities.png", year)
        plots.plot_curvature(curvature, out / "engel_curvature.png", year)
        plots.plot_quintile_shares(groups, out / "budget_shares_by_quintile.png", year)
        plots.plot_engel_curves(households, engel, out / "engel_curves.png", year)
    return YearResults(data, households, means, groups, engel, detail_engel, link)


def run(years: list[int], data_dir: Path, output_dir: Path, make_plots: bool = True) -> None:
    # Check every input up front, so a multi-year run fails before doing any work.
    if len(years) > 1:
        try:
            for year in years:
                cpi(year)
        except KeyError as err:
            raise SystemExit(err.args[0]) from None
    try:
        for year in years:
            for kind in ("fmli", "mtbi", "itbi"):
                quarterly_files(year, data_dir / "raw", kind)
            read_stub(year, data_dir / "raw")
    except FileNotFoundError as err:
        raise SystemExit(f"{err} (Or place the file there manually.)") from None

    try:
        prices = load_prices(data_dir / "raw")
    except FileNotFoundError as err:
        prices = None
        print(f"\nSkipping inflation by group: {err}")

    results = {year: run_year(year, data_dir, output_dir, make_plots, prices) for year in years}

    if len(years) > 1:
        print(f"\n=== {years[0]}-{years[-1]} ===")
        out = output_dir / f"{years[0]}-{years[-1]}"
        out.mkdir(parents=True, exist_ok=True)
        table = trends({year: result.means for year, result in results.items()})
        table.to_csv(out / "trends.csv", index=False)
        wide = table[table["item"].isin(["income_before_tax", "total"])].pivot(
            index="year", columns="item", values="real_mean"
        )
        _print(f"Income and total spending per consumer unit, in {years[-1]} dollars", wide, decimals=0)

        elasticities = {year: result.engel.elasticities for year, result in results.items()}
        pd.concat(elasticities, names=["year"]).to_csv(out / "elasticities.csv")
        if make_plots:
            plots.plot_trends(table, out / "trends.png")
            plots.plot_elasticity_trends(elasticities, out / "elasticities.png")
        _run_inflation(results, prices, out, make_plots)

    print(f"\nResults written to {output_dir}")


def _run_inflation(results: dict[int, YearResults], prices: pd.DataFrame | None, out: Path, make_plots: bool) -> None:
    """Cumulative inflation by group over all the years, if they are consecutive and each has price data."""
    years = sorted(results)
    links = {year: results[year].inflation for year in years}
    if prices is None or any(link is None for link in links.values()) or years != list(range(years[0], years[-1] + 1)):
        return
    first, last = years[0], years[-1] + 1
    cumulative = inflation.chain(links)
    cumulative.to_csv(out / "inflation.csv", index=False)
    official = inflation.official_cpi_change(prices, first, last)
    _print(
        f"Inflation {first}-{last} by group, in %, compounding each year's basket at the next year's prices.\n"
        f"gap: percentage points above or below all consumer units. Published CPI-U: {100 * official:.1f}%",
        _percent(cumulative, ["inflation", "inflation_se", "gap", "gap_se"])
        .set_index(["grouping", "group"])
        .drop(columns=["from_year", "to_year"]),
    )
    # Why the income groups differ: their first-year baskets against the whole period's price changes.
    change = category_price_change(prices, first, last)
    inflation.gap_contributions(results[first].data.interviews, change, "income_quintile").to_csv(
        out / "inflation_contributions.csv", index=False
    )
    if make_plots:
        plots.plot_inflation_gaps(cumulative, out / "inflation.png", official)


def main(argv: list[str] | None = None) -> None:
    root = default_root()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--years",
        "--year",
        dest="years",
        type=parse_years,
        default=[2024],
        help="a year, a range (2020-2024), or a list (2019,2022-2024); default 2024",
    )
    common.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data",
        help="folder holding raw/ and processed/ (default: %(default)s; or set CEX_HOME)",
    )

    parser = argparse.ArgumentParser(prog="cex", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser(
        "download", parents=[common], help="Download Interview microdata, category definitions, and price indexes"
    )
    dl.add_argument(
        "--contact",
        default=os.environ.get("CEX_CONTACT_EMAIL"),
        help="Your email, sent in the User-Agent as BLS requires (or set CEX_CONTACT_EMAIL)",
    )
    dl.add_argument("--overwrite", action="store_true")

    rn = sub.add_parser("run", parents=[common], help="Load, clean, summarize, model, and chart each year")
    rn.add_argument("--output-dir", type=Path, default=root / "outputs", help="default: %(default)s")
    rn.add_argument("--no-plots", action="store_true", help="skip the PNG charts")

    args = parser.parse_args(argv)
    if args.command == "download":
        if not args.contact:
            parser.error("BLS requires contact info: pass --contact you@example.com or set CEX_CONTACT_EMAIL")
        raw_dir = args.data_dir / "raw"
        download_stubs(raw_dir, args.contact, args.overwrite)
        # Prices through the current year: inflation for a year's basket needs the following year's prices.
        download_prices(raw_dir, min(args.years), datetime.date.today().year, args.overwrite)
        for year in args.years:
            download_interview(year, raw_dir, args.contact, args.overwrite)
            if needs_previous_release(year, raw_dir):
                print(f"  The {year} release starts at Apr-Jun; its Jan-Mar interviews are in the {year - 1} release.")
                download_interview(year - 1, raw_dir, args.contact)
    elif args.command == "run":
        run(args.years, args.data_dir, args.output_dir, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
