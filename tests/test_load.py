import zipfile

import pytest

from cex.load import COLUMNS, load_fmli, load_ucc_records, needs_previous_release, quarterly_files

from conftest import GROCERY_UCC, WAGES_UCC, YEAR, rewrite_member, write_synthetic_raw


@pytest.fixture(scope="module")
def split_raw(tmp_path_factory):
    """2022 and 2023: releases that start at Apr-Jun, with Jan-Mar in the previous year's zip."""
    return write_synthetic_raw(tmp_path_factory.mktemp("split"), years=[2022, 2023], n_cu=300)


def test_five_file_release(synthetic_raw):
    files = quarterly_files(YEAR, synthetic_raw)
    assert [path.name for path, _ in files] == ["intrvw24.zip"] * 5
    assert [member.split("/")[-1] for _, member in files] == [
        "fmli241x.csv",
        "fmli242.csv",
        "fmli243.csv",
        "fmli244.csv",
        "fmli251.csv",
    ]
    assert not needs_previous_release(YEAR, synthetic_raw)


def test_first_quarter_comes_from_the_previous_release(split_raw):
    assert needs_previous_release(2023, split_raw)
    for kind in ("fmli", "mtbi", "itbi"):
        files = quarterly_files(2023, split_raw, kind)
        assert [path.name for path, _ in files] == ["intrvw22.zip"] + ["intrvw23.zip"] * 4
        assert [member.split("/")[-1] for _, member in files] == [f"{kind}{q}.csv" for q in (231, 232, 233, 234, 241)]
    # The same Jan-Mar 2023 interviews end 2022 and start 2023.
    last_of_2022 = load_fmli(2022, split_raw).query("SOURCE_FILE == 'fmli231'")
    first_of_2023 = load_fmli(2023, split_raw).query("SOURCE_FILE == 'fmli231'")
    assert len(first_of_2023) > 0 and set(last_of_2022["NEWID"]) == set(first_of_2023["NEWID"])


def test_missing_releases_are_named(split_raw, tmp_path):
    with pytest.raises(FileNotFoundError, match="intrvw19.zip.*cex download --years 2019"):
        quarterly_files(2019, split_raw)
    # The 2023 release alone is not enough for 2023.
    (tmp_path / "intrvw23.zip").write_bytes((split_raw / "intrvw23.zip").read_bytes())
    with pytest.raises(FileNotFoundError, match="Jan-Mar 2023.*intrvw22.zip"):
        quarterly_files(2023, tmp_path)


def test_incomplete_release_is_rejected(split_raw, tmp_path):
    for name in ("intrvw22.zip", "intrvw23.zip"):
        with zipfile.ZipFile(split_raw / name) as src, zipfile.ZipFile(tmp_path / name, "w") as dst:
            for member in src.namelist():
                if not member.endswith("fmli233.csv"):
                    dst.writestr(member, src.read(member))
    with pytest.raises(FileNotFoundError, match="fmli233"):
        quarterly_files(2023, tmp_path)


def test_load_stacks_quarterly_files_in_order(synthetic_raw):
    raw = load_fmli(YEAR, synthetic_raw)
    assert list(raw["SOURCE_FILE"].unique()) == ["fmli241x", "fmli242", "fmli243", "fmli244", "fmli251"]
    assert raw["NEWID"].is_unique
    assert set(COLUMNS) <= set(raw.columns)
    # "." is missing, including in a file with lowercase headers.
    assert raw["REGION"].isna().sum() == 6
    # GROCER* columns in later quarters are read as FDHOME*.
    assert (raw["FDHOMEPQ"] > 0).all() and "GROCERPQ" not in raw


def test_load_rejects_file_missing_a_column(synthetic_raw, tmp_path):
    (tmp_path / "intrvw24.zip").write_bytes((synthetic_raw / "intrvw24.zip").read_bytes())
    rewrite_member(tmp_path / "intrvw24.zip", "fmli243.csv", lambda df: df.drop(columns=["HOUSPQ"]))
    with pytest.raises(KeyError, match="fmli243.*HOUSPQ"):
        load_fmli(YEAR, tmp_path)


def test_ucc_records_come_from_both_files(synthetic_raw):
    records = load_ucc_records(YEAR, synthetic_raw)
    assert list(records.columns) == ["newid", "ref_year", "ref_month", "ucc", "cost", "file"]
    assert set(records["file"]) == {"mtbi", "itbi"}
    # UCCs are six characters; the leading zeros of 005420 survive.
    assert (records["ucc"].str.len() == 6).all() and "005420" in set(records["ucc"])
    assert set(records.loc[records["ucc"] == WAGES_UCC, "file"]) == {"itbi"}
    assert set(records["newid"]) == set(load_fmli(YEAR, synthetic_raw)["NEWID"])
    # Each interview's records cover the 3 months before the interview month.
    one = records[(records["ucc"] == GROCERY_UCC) & (records["newid"] == records["newid"].iloc[0])]
    assert len(one) == 3 and one["ref_month"].nunique() == 3

    only = load_ucc_records(YEAR, synthetic_raw, uccs={GROCERY_UCC})
    assert set(only["ucc"]) == {GROCERY_UCC}
    assert only["cost"].sum() == pytest.approx(records.loc[records["ucc"] == GROCERY_UCC, "cost"].sum())
