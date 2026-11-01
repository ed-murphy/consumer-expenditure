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

PUMD_URL = "https://www.bls.gov/cex/pumd/data/{folder}/intrvw{yy:02d}.zip"


def interview_url(year: int) -> str:
    # BLS moved the CSV files to a different folder starting with the 2022 release.
    folder = "csv" if year >= 2022 else "comma"
    return PUMD_URL.format(folder=folder, yy=year % 100)


def download_interview(year: int, raw_dir: Path, contact: str, overwrite: bool = False) -> Path:
    dest = raw_zip_path(year, raw_dir)
    if dest.exists() and not overwrite:
        print(f"Already downloaded: {dest}")
        return dest

    url = interview_url(year)
    request = urllib.request.Request(url, headers={"User-Agent": f"cex-research-pipeline ({contact})"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(request, timeout=300) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)
    return dest
