import zipfile

import pytest

from cex import stubs

from conftest import FOOD_ONLY_UCC, GROCERY_UCC, ITEMS, TRIP_GROCERY_UCC, YEAR, stub_text


@pytest.fixture(scope="module")
def tree(synthetic_raw):
    return stubs.expenditure_tree(YEAR, synthetic_raw)


def test_read_stub_parses_every_line(synthetic_raw):
    stub = stubs.read_stub(YEAR, synthetic_raw)
    assert stub.iloc[0].to_dict() == {
        "level": 1,
        "title": "Interview stub parameter file",
        "code": "HEADINTR",
        "kind": "H",
        "factor": 1,
        "section": "CUCHARS",
    }
    # A title that wraps onto a continuation line is put back together.
    long = stub[stub["code"] == TRIP_GROCERY_UCC].iloc[0]
    assert long["title"] == "Food prepared by consumer unit on out of town trips, a title long enough to wrap"
    assert set(stub["section"]) == {"CUCHARS", "EXPEND", "FOOD", "INCOME", "ADDENDA"}


def test_tree_holds_only_spending(tree):
    uccs = set(tree.loc[tree["is_item"], "ucc"])
    assert uccs == {item.ucc for item in ITEMS} | {GROCERY_UCC, TRIP_GROCERY_UCC}
    assert "900000" not in uccs and "910050" not in uccs  # income and addenda sections
    mortgage = tree[tree["ucc"] == "220311"].iloc[0]
    assert mortgage["path"] == (
        "Housing",
        "Shelter",
        "Owned dwellings",
        "Mortgage interest and charges",
        "Mortgage interest",
    )
    assert tree.loc[tree["ucc"] == "005420", "factor"].item() == 4


def test_groceries_use_the_all_purchases_ucc(tree):
    # The stub file lists the food-only UCC; the tree swaps in all grocery purchases.
    assert FOOD_ONLY_UCC in stub_text() and GROCERY_UCC not in stub_text()
    at_home = tree[[path[:2] == stubs.FOOD_AT_HOME and is_item for path, is_item in zip(tree["path"], tree["is_item"])]]
    assert list(at_home["ucc"]) == [GROCERY_UCC, TRIP_GROCERY_UCC]


def test_major_categories(tree):
    major = stubs.major_categories(tree)
    assert set(major) == set(stubs.MAJOR) and len(stubs.MAJOR) == 15
    assert major[GROCERY_UCC] == "groceries" and major["790410"] == "food_away"
    assert major["220311"] == major["210110"] == "housing"
    assert major["540000"] == "healthcare"


def test_detail_categories(tree):
    detail = stubs.detail_categories(tree)
    names = stubs.detail_names(tree)
    assert set(detail) == set(names) and len(names) == 31
    assert detail["220311"] == "owned_dwellings" and detail["210110"] == "rented_dwellings"  # shelter is split further
    assert detail["540000"] == "prescription_drugs"  # an item beside groups is its own category
    assert detail["720110"] == detail["440210"] == "apparel"  # kept whole
    assert detail["590230"] == "reading"  # a major category with only items under it is not split
    assert names[:4] == ["groceries", "food_away", "alcohol_home", "alcohol_away"]  # the stub's order


def test_a_retitled_category_keeps_its_name(synthetic_raw, tmp_path):
    text = stub_text().replace("Gasoline and other fuels            ", "Gasoline, other fuels, and motor oil")
    assert text != stub_text()
    with zipfile.ZipFile(tmp_path / "stubs.zip", "w") as zf:
        zf.writestr(f"stubs/CE-HG-Inter-{YEAR}.txt", text)
    assert stubs.detail_categories(stubs.expenditure_tree(YEAR, tmp_path))["470111"] == "gasoline"


def test_labels():
    assert stubs.label("groceries") == "Groceries"
    assert stubs.label("some_new_category") == "Some new category"


def test_unknown_major_category_is_an_error(tmp_path):
    with zipfile.ZipFile(tmp_path / "stubs.zip", "w") as zf:
        zf.writestr(f"stubs/CE-HG-Inter-{YEAR}.txt", stub_text().replace("Reading", "Books  "))
    with pytest.raises(ValueError, match="Books"):
        stubs.expenditure_tree(YEAR, tmp_path)


def test_repeated_ucc_is_an_error(tmp_path):
    lines = stub_text().split("\r\n")
    cigarettes = next(i for i, line in enumerate(lines) if "630110" in line)
    lines.insert(cigarettes, lines[cigarettes])
    with zipfile.ZipFile(tmp_path / "stubs.zip", "w") as zf:
        zf.writestr(f"stubs/CE-HG-Inter-{YEAR}.txt", "\r\n".join(lines))
    with pytest.raises(ValueError, match="630110"):
        stubs.expenditure_tree(YEAR, tmp_path)


def test_missing_stub_files_are_named(synthetic_raw, tmp_path):
    with pytest.raises(FileNotFoundError, match="cex download"):
        stubs.read_stub(YEAR, tmp_path)
    with pytest.raises(FileNotFoundError, match="CE-HG-Inter-1999"):
        stubs.read_stub(1999, synthetic_raw)
