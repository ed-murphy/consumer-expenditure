"""Command-line entry point: `uv run cex download ...` and `uv run cex run ...`."""

import argparse
import os
from pathlib import Path

import pandas as pd

from cex.clean import clean_fmli
from cex.download import download_interview
from cex.load import load_fmli, raw_zip_path
from cex.model import elasticity_table, fit_engel
from cex.summarize import annual_means

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run(year: int) -> None:
    zip_path = raw_zip_path(year, RAW_DIR)
    if not zip_path.exists():
        raise SystemExit(f"{zip_path} not found. Run `cex download --year {year}` or place the file there manually.")

    print(f"Loading {zip_path.name} ...")
    raw = load_fmli(zip_path)
    print(f"  {len(raw):,} interview records from {raw['SOURCE_FILE'].nunique()} quarterly files")

    df = clean_fmli(raw, year)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DIR / f"fmli_{year}.parquet"
    df.to_parquet(processed_path)
    print(f"  {df['cuid'].nunique():,} distinct consumer units -> {processed_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 160, "display.max_columns", None):
        means = annual_means(df)
        means.to_csv(OUTPUT_DIR / f"annual_means_{year}.csv")
        print(f"\nAverage annual values per consumer unit, {year} (Interview survey only)")
        print("Compare with https://www.bls.gov/cex/tables.htm\n")
        print(means)

    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 160, "display.max_columns", None):
        food = fit_engel(df, "food_home")
        food["coefficients"].to_csv(OUTPUT_DIR / f"engel_food_home_{year}.csv")
        print(f"\nEngel curve for food at home (n={food['n']:,}, weighted R^2={food['r2']:.3f})\n")
        print(food["coefficients"])

        elasticities = elasticity_table(df)
        elasticities.to_csv(OUTPUT_DIR / f"income_elasticities_{year}.csv")
        print("\nIncome elasticities by category (< 1 = necessity, > 1 = luxury)\n")
        print(elasticities)

    print(f"\nResults written to {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="cex", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download the Interview PUMD zip from BLS")
    dl.add_argument("--year", type=int, default=2024)
    dl.add_argument(
        "--contact",
        default=os.environ.get("CEX_CONTACT_EMAIL"),
        help="Your email, sent in the User-Agent as BLS requires (or set CEX_CONTACT_EMAIL)",
    )
    dl.add_argument("--overwrite", action="store_true")

    rn = sub.add_parser("run", help="Load, clean, summarize, and model one year")
    rn.add_argument("--year", type=int, default=2024)

    args = parser.parse_args()
    if args.command == "download":
        if not args.contact:
            parser.error("BLS requires contact info: pass --contact you@example.com or set CEX_CONTACT_EMAIL")
        download_interview(args.year, RAW_DIR, args.contact, args.overwrite)
    elif args.command == "run":
        run(args.year)


if __name__ == "__main__":
    main()
