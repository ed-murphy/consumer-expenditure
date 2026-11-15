"""Read the raw FMLI (consumer-unit characteristics, income, and summary spending) files.

A calendar year of spending is covered by five quarterly files, e.g. for 2024:

    fmli241x.csv  interviews in Jan-Mar 2024 (reference period partly in 2023)
    fmli242.csv   interviews in Apr-Jun 2024
    fmli243.csv   interviews in Jul-Sep 2024
    fmli244.csv   interviews in Oct-Dec 2024
    fmli251.csv   interviews in Jan-Mar 2025 (reference period partly in 2024)

Each row is one consumer unit (CU, roughly a household) in one quarterly interview.

The release `intrvwYY.zip` does not always hold all five. The 2019 and 2024
releases do (their first file carries an "x": a re-release of that quarter with
updated processing). The 2020-2023 releases hold only four, starting at Apr-Jun;
their Jan-Mar interviews are the last file of the previous release (`fmli201.csv`
in `intrvw19.zip`, and so on). `quarterly_files` finds all five wherever they are.
The other quarterly file types (MTBI, MEMI, ...) are laid out the same way.
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

# BLS's ready-made summary expenditure variables. Each comes as a pair: <CODE>PQ
# (spending in the previous calendar quarter) and <CODE>CQ (spending in the
# current calendar quarter). The 15 codes after TOTEXP are the major components
# of total spending. This package builds spending from the item-level records
# instead (see cex.spending) and reads these only to cross-check that work,
# because their definitions are not stable over time.
EXPENDITURE_CODES = [
    "TOTEXP",
    "FDHOME",
    "FDAWAY",
    "ALCBEV",
    "HOUS",
    "APPAR",
    "TRANS",
    "HEALTH",
    "ENTERT",
    "PERSCA",
    "READ",
    "EDUCA",
    "TOBACC",
    "MISC",
    "CASHCO",
    "PERINS",
]

REPLICATE_WEIGHTS = [f"WTREP{i:02d}" for i in range(1, 45)]

# Income before taxes: FINCBTX1-5 are BLS's five imputations for nonresponse and
# FINCBTXM is their mean. INC_RNK1-5 / INC_RNKM rank each CU by that income within
# its quarterly file (weighted cumulative share, 0-1).
INCOME_IMPUTATIONS = [f"FINCBTX{m}" for m in range(1, 6)]
INCOME_RANKS = [f"INC_RNK{m}" for m in range(1, 6)]

COLUMNS = [
    "NEWID",
    "QINTRVYR",
    "QINTRVMO",
    "FINLWT21",
    *REPLICATE_WEIGHTS,
    "FINCBTXM",
    *INCOME_IMPUTATIONS,
    "INC_RNKM",
    *INCOME_RANKS,
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
    "RENTEQVX",
    *[f"{code}{suffix}" for code in EXPENDITURE_CODES for suffix in ("PQ", "CQ")],
]

# Variables BLS replaced, mapped to the name used in COLUMNS. Starting in Apr-Jun
# 2024 the food-at-home summary variable FDHOME gives way to GROCER. It is not
# the same thing: FDHOME was food bought at grocery stores, GROCER is everything
# bought there, about a quarter more (see stubs._use_all_grocery_purchases).
RENAMES = {"GROCERPQ": "FDHOMEPQ", "GROCERCQ": "FDHOMECQ"}

# The two files of item-level records, and the names this package gives their columns.
# MTBI holds expenses. ITBI holds income, and also the deductions from pay for
# Social Security and pensions, which BLS counts as spending.
UCC_RECORD_COLUMNS = {
    "mtbi": {"NEWID": "newid", "REF_YR": "ref_year", "REF_MO": "ref_month", "UCC": "ucc", "COST": "cost"},
    "itbi": {"NEWID": "newid", "REFYR": "ref_year", "REFMO": "ref_month", "UCC": "ucc", "VALUE": "cost"},
}


def raw_zip_path(year: int, raw_dir: Path) -> Path:
    return Path(raw_dir) / f"intrvw{year % 100:02d}.zip"


def _quarterly_members(zip_path: Path, kind: str) -> dict[tuple[int, int], str]:
    """The `kind` files in a release, keyed by (two-digit interview year, interview quarter)."""
    pattern = re.compile(rf"(?:^|/){kind}(\d{{2}})(\d)x?\.csv$", re.IGNORECASE)
    with zipfile.ZipFile(zip_path) as zf:
        matches = ((pattern.search(name), name) for name in zf.namelist())
        return {(int(m[1]), int(m[2])): name for m, name in matches if m}


def quarterly_files(year: int, raw_dir: Path, kind: str = "fmli") -> list[tuple[Path, str]]:
    """The five quarterly `kind` files covering calendar `year`, in order, as (zip path, member name).

    Four or five come from `intrvwYY.zip`. If it has no Jan-Mar file, that
    quarter is taken from the previous year's release (see the module docstring).
    """
    release = raw_zip_path(year, raw_dir)
    if not release.exists():
        raise FileNotFoundError(f"{release} not found. Run `cex download --years {year}`.")
    yy, next_yy = year % 100, (year + 1) % 100
    members = _quarterly_members(release, kind)
    files = {key: (release, members[key]) for key in members}
    if (yy, 1) not in files:
        previous = raw_zip_path(year - 1, raw_dir)
        if not previous.exists():
            raise FileNotFoundError(
                f"{release.name} has no {kind} file for Jan-Mar {year}; BLS ships that quarter in {previous.name}, "
                f"which was not found in {previous.parent}. Run `cex download --years {year - 1}`."
            )
        earlier = _quarterly_members(previous, kind)
        if (yy, 1) in earlier:
            files[(yy, 1)] = (previous, earlier[(yy, 1)])
    needed = [(yy, 1), (yy, 2), (yy, 3), (yy, 4), (next_yy, 1)]
    missing = [f"{kind}{y:02d}{q}" for y, q in needed if (y, q) not in files]
    if missing:
        raise FileNotFoundError(f"Quarterly files not found for {year}: {', '.join(missing)} (looked in {release})")
    return [files[key] for key in needed]


def needs_previous_release(year: int, raw_dir: Path) -> bool:
    """Whether the `year` release lacks its Jan-Mar files, so that the previous release is needed too."""
    return (year % 100, 1) not in _quarterly_members(raw_zip_path(year, raw_dir), "fmli")


def read_member(zip_path: Path, member: str, **read_csv_options) -> pd.DataFrame:
    """One csv from a release, with upper-case column names and a SOURCE_FILE column."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        # PUMD csvs mark missing values with blanks or "."
        df = pd.read_csv(f, na_values=[".", " "], low_memory=False, **read_csv_options)
    df.columns = df.columns.str.upper()
    df = df.copy()  # consolidate the column blocks read_csv creates before adding one more
    df["SOURCE_FILE"] = Path(member).stem.lower()
    return df


def load_fmli(year: int, raw_dir: Path, columns: list[str] = COLUMNS) -> pd.DataFrame:
    """Stack the five quarterly FMLI files for `year`, keeping only `columns`.

    Every file must contain every column (after RENAMES): a variable that exists
    in only some quarters would otherwise silently become missing in the rest.
    """
    wanted = set(columns)
    readable = wanted | {old for old, new in RENAMES.items() if new in wanted}
    frames = []
    for zip_path, member in quarterly_files(year, raw_dir, "fmli"):
        df = read_member(zip_path, member, usecols=lambda c: c.upper() in readable)
        # Files after the rename still carry the old columns, filled with zeros; the new name wins.
        df = df.drop(columns=[new for old, new in RENAMES.items() if old in df and new in df])
        df = df.rename(columns=RENAMES)
        missing = wanted - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found in {member}: {sorted(missing)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_ucc_records(year: int, raw_dir: Path, uccs: set[str] | None = None) -> pd.DataFrame:
    """Item-level spending records for `year` from the MTBI and ITBI files, optionally only for `uccs`.

    One row per consumer unit, interview, reference month, and UCC: `newid`,
    `ref_year`, `ref_month`, `ucc` (six characters, zero-padded), `cost`, and
    `file` ("mtbi" or "itbi").
    """
    frames = []
    for kind, names in UCC_RECORD_COLUMNS.items():
        for zip_path, member in quarterly_files(year, raw_dir, kind):
            # Everything as text first: UCCs like 005420 would lose their zeros as numbers.
            df = read_member(zip_path, member, usecols=lambda c, names=names: c.upper() in names, dtype=str)
            missing = set(names) - set(df.columns)
            if missing:
                raise KeyError(f"Columns not found in {member}: {sorted(missing)}")
            df = df.rename(columns=names)[list(names.values())]
            df["ucc"] = df["ucc"].str.strip().str.zfill(6)
            if uccs is not None:
                df = df[df["ucc"].isin(uccs)]
            df = df.astype({"newid": "int64", "ref_year": "int64", "ref_month": "int64", "cost": float})
            frames.append(df.assign(file=kind))
    return pd.concat(frames, ignore_index=True)
