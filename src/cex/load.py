"""Read the raw FMLI (consumer-unit characteristics, income, and summary spending) files.

Each Interview release `intrvwYY.zip` holds five quarterly FMLI files, e.g. for 2024:

    fmli241x.csv  interviews in Jan-Mar 2024 (reference period partly in 2023)
    fmli242.csv   interviews in Apr-Jun 2024
    fmli243.csv   interviews in Jul-Sep 2024
    fmli244.csv   interviews in Oct-Dec 2024
    fmli251.csv   interviews in Jan-Mar 2025 (reference period partly in 2024)

Each row is one consumer unit (CU, roughly a household) in one quarterly interview.
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

FMLI_PATTERN = re.compile(r"fmli(\d{2})(\d)x?\.csv$", re.IGNORECASE)

# Summary expenditure variables. Each comes as a pair: <CODE>PQ (spending in the
# previous calendar quarter) and <CODE>CQ (spending in the current calendar quarter).
# Total food (FOOD) is not loaded: BLS dropped it after 2024 Q1, and it equals FDHOME + FDAWAY.
EXPENDITURE_CODES = ["TOTEXP", "FDHOME", "FDAWAY", "HOUS", "TRANS", "HEALTH", "ENTERT", "APPAR"]

REPLICATE_WEIGHTS = [f"WTREP{i:02d}" for i in range(1, 45)]

COLUMNS = [
    "NEWID",
    "QINTRVYR",
    "QINTRVMO",
    "FINLWT21",
    *REPLICATE_WEIGHTS,
    "FINCBTXM",
    "AGE_REF",
    "SEX_REF",
    "EDUC_REF",
    "FAM_SIZE",
    "PERSLT18",
    "PERSOT64",
    "REGION",
    "BLS_URBN",
    "CUTENURE",
    *[f"{code}{suffix}" for code in EXPENDITURE_CODES for suffix in ("PQ", "CQ")],
]

# Variables BLS renamed, mapped to the name used in COLUMNS. Starting in 2024 Q2,
# food at home is GROCERPQ/GROCERCQ (same UCCs as FDHOMEPQ/FDHOMECQ); see
# https://www.bls.gov/cex/pumd/errata/errata-2023.htm
RENAMES = {"GROCERPQ": "FDHOMEPQ", "GROCERCQ": "FDHOMECQ"}


def raw_zip_path(year: int, raw_dir: Path) -> Path:
    return Path(raw_dir) / f"intrvw{year % 100:02d}.zip"


def fmli_members(zf: zipfile.ZipFile) -> list[str]:
    """Return the FMLI csv members of the zip in chronological order."""
    members = [name for name in zf.namelist() if FMLI_PATTERN.search(name)]
    return sorted(members, key=lambda name: FMLI_PATTERN.search(name).groups())


def load_fmli(zip_path: Path, columns: list[str] = COLUMNS) -> pd.DataFrame:
    """Stack the quarterly FMLI files, keeping only `columns`.

    Every file must contain every column (after RENAMES): a variable that exists
    in only some quarters would otherwise silently become missing in the rest.
    """
    wanted = set(columns)
    readable = wanted | {old for old, new in RENAMES.items() if new in wanted}
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        members = fmli_members(zf)
        if not members:
            raise FileNotFoundError(f"No fmli*.csv files found in {zip_path}")
        for member in members:
            with zf.open(member) as f:
                # PUMD csvs mark missing values with blanks or "."
                df = pd.read_csv(
                    f,
                    usecols=lambda c: c.upper() in readable,
                    na_values=[".", " "],
                    low_memory=False,
                )
            df.columns = df.columns.str.upper()
            # Files after the rename still carry the old columns, filled with zeros; the new name wins.
            df = df.drop(columns=[new for old, new in RENAMES.items() if old in df and new in df])
            df = df.rename(columns=RENAMES)
            missing = wanted - set(df.columns)
            if missing:
                raise KeyError(f"Columns not found in {member}: {sorted(missing)}")
            df["SOURCE_FILE"] = Path(member).stem.lower()
            frames.append(df)

    return pd.concat(frames, ignore_index=True)
