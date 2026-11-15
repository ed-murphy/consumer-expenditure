"""Download CE Interview public-use microdata (PUMD) from BLS.

BLS rejects anonymous scripted requests. Its access policy
(https://www.bls.gov/bls/pss.htm) asks that automated requests identify the
requester with contact information in the User-Agent header, so a contact
email is required here.
"""

import shutil
import urllib.request
from pathlib import Path

from cex.load import raw_zip_path
from cex.stubs import stubs_zip_path

PUMD_URL = "https://www.bls.gov/cex/pumd/data/{folder}/intrvw{yy:02d}.zip"
# "Hierarchical groupings": for every year, the tree of spending categories and
# the UCCs (item codes) under each one. See cex.stubs.
STUBS_URL = "https://www.bls.gov/cex/pumd/stubs.zip"


def interview_url(year: int) -> str:
    # BLS moved the CSV files to a different folder starting with the 2022 release.
    folder = "csv" if year >= 2022 else "comma"
    return PUMD_URL.format(folder=folder, yy=year % 100)


def _fetch(url: str, dest: Path, contact: str, overwrite: bool) -> Path:
    if dest.exists() and not overwrite:
        print(f"Already downloaded: {dest}")
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": f"cex-research-pipeline ({contact})"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(request, timeout=300) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)
    return dest


def download_interview(year: int, raw_dir: Path, contact: str, overwrite: bool = False) -> Path:
    return _fetch(interview_url(year), raw_zip_path(year, raw_dir), contact, overwrite)


def download_stubs(raw_dir: Path, contact: str, overwrite: bool = False) -> Path:
    return _fetch(STUBS_URL, stubs_zip_path(raw_dir), contact, overwrite)
