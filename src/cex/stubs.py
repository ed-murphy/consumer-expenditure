"""BLS's "hierarchical groupings" (stub files): which item codes make up which spending category.

Every expense in the Interview survey is recorded under a six-digit item code,
the UCC (Universal Classification Code): 210110 is rent, 470111 gasoline. The
stub file for a year lists the tree of published categories with the UCCs under
each one. `stubs.zip` (downloaded by `cex download`) holds one per year:

    1  1  Average annual expenditures              TOTALEXP  G  1  EXPEND
    1  2    Housing                                HOUSING   G  1  EXPEND
    1  3      Shelter                              SHELTER   G  1  EXPEND
    1  4        Rented dwellings                   RNTDWELL  G  1  EXPEND
    1  5          Rent                             210110    I  1  EXPEND

The columns are: line type (2 continues the previous title), depth in the tree,
title, group code or UCC, G for a group or I for an item, a factor to multiply
the cost by, and the section of the published table.

The factor is 4 for a few items that are asked about only in a household's last
interview and cover 12 months (finance charges on credit cards and loans).
Only a quarter of interviews carry them, so they count four times.

This module reads the part of the tree under "Average annual expenditures" and
cuts it at two depths: the 15 major categories used throughout this package,
and about 30 subcategories ("detail").
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

STUB_LINE = re.compile(r"^1\s+(\d+)\s+(.*?)\s+(\S+)\s+([GISTH])\s+(-?\d+)\s+(\S+)\s*$")
ROOT = "Average annual expenditures"

# All grocery-store purchases, nonfood items included. See `_use_all_grocery_purchases`.
GROCERY_UCC = "790210"
FOOD_ONLY_GROCERY_UCC = "790240"
FOOD_AT_HOME = ("Food", "Food at home")

# Name and label for each category, with its path below the root in the stub
# file. A category has several paths when BLS has retitled it; all of them map
# to the one name, so a category keeps its name across years. Categories not
# listed here get a name made from their title.
_CATEGORIES = [
    ("groceries", "Groceries", [FOOD_AT_HOME]),
    ("food_away", "Food away from home", [("Food", "Food away from home")]),
    ("alcohol", "Alcoholic beverages", [("Alcoholic beverages",)]),
    ("alcohol_home", "Alcohol at home", [("Alcoholic beverages", "At home")]),
    ("alcohol_away", "Alcohol away from home", [("Alcoholic beverages", "Away from home")]),
    ("housing", "Housing", [("Housing",)]),
    ("owned_dwellings", "Owned dwellings", [("Housing", "Shelter", "Owned dwellings")]),
    ("rented_dwellings", "Rented dwellings", [("Housing", "Shelter", "Rented dwellings")]),
    ("other_lodging", "Other lodging", [("Housing", "Shelter", "Other lodging")]),
    ("utilities", "Utilities, fuels, public services", [("Housing", "Utilities, fuels, and public services")]),
    ("household_operations", "Household operations", [("Housing", "Household operations")]),
    ("furnishings", "Furnishings and equipment", [("Housing", "Household furnishings and equipment")]),
    ("apparel", "Apparel and services", [("Apparel and services",)]),
    ("transportation", "Transportation", [("Transportation",)]),
    ("vehicle_purchases", "Vehicle purchases", [("Transportation", "Vehicle purchases (net outlay)")]),
    (
        "gasoline",
        "Gasoline and other fuels",
        [
            ("Transportation", "Gasoline and motor oil"),
            ("Transportation", "Gasoline, other fuels, and motor oil"),
            ("Transportation", "Gasoline and other fuels"),
        ],
    ),
    ("other_vehicle", "Other vehicle expenses", [("Transportation", "Other vehicle expenses")]),
    (
        "public_transportation",
        "Public and other transportation",
        [("Transportation", "Public and other transportation")],
    ),
    ("healthcare", "Healthcare", [("Healthcare",)]),
    ("health_insurance", "Health insurance", [("Healthcare", "Health insurance")]),
    ("medical_services", "Medical services", [("Healthcare", "Medical services")]),
    ("prescription_drugs", "Prescription drugs", [("Healthcare", "Prescription drugs")]),
    ("medical_supplies", "Medical supplies", [("Healthcare", "Medical supplies")]),
    ("entertainment", "Entertainment", [("Entertainment",)]),
    ("fees_admissions", "Fees and admissions", [("Entertainment", "Fees and admissions")]),
    ("audio_visual", "Audio and visual", [("Entertainment", "Audio and visual equipment and services")]),
    ("pets_toys_hobbies", "Pets, toys, hobbies", [("Entertainment", "Pets, toys, hobbies, and playground equipment")]),
    (
        "other_entertainment",
        "Other entertainment",
        [("Entertainment", "Other entertainment supplies, equipment, and services")],
    ),
    ("personal_care", "Personal care", [("Personal care products and services",)]),
    ("reading", "Reading", [("Reading",)]),
    ("education", "Education", [("Education",)]),
    ("tobacco", "Tobacco", [("Tobacco products and smoking supplies",)]),
    ("miscellaneous", "Miscellaneous", [("Miscellaneous",)]),
    ("cash_contributions", "Cash contributions", [("Cash contributions",)]),
    ("insurance_pensions", "Insurance and pensions", [("Personal insurance and pensions",)]),
    (
        "life_insurance",
        "Life and personal insurance",
        [("Personal insurance and pensions", "Life and other personal insurance")],
    ),
    (
        "pensions_social_security",
        "Pensions and Social Security",
        [
            ("Personal insurance and pensions", "Pensions and Social Security"),
            ("Personal insurance and pensions", "Retirement, pensions, and Social Security"),
        ],
    ),
]
NAMES = {path: name for name, _, paths in _CATEGORIES for path in paths}
LABELS = {name: label for name, label, _ in _CATEGORIES} | {"total": "Total spending"}

# The 15 major categories: the groups directly under the root, with food split in two.
MAJOR_PATHS = [path for path in NAMES if len(path) == 1 or path[0] == "Food"]
MAJOR = [NAMES[path] for path in MAJOR_PATHS]

# How the detail cut departs from "the groups directly under each major category":
# shelter is split one level further, and apparel is left whole because BLS
# reorganized its subgroups in 2021.
SPLIT_FURTHER = {("Housing", "Shelter")}
KEEP_WHOLE = {("Apparel and services",)}


def stubs_zip_path(raw_dir: Path) -> Path:
    return Path(raw_dir) / "stubs.zip"


def read_stub(year: int, raw_dir: Path) -> pd.DataFrame:
    """The Interview stub file for `year`, one row per entry, in file order."""
    path = stubs_zip_path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `cex download` (it fetches the category definitions too).")
    member = f"stubs/CE-HG-Inter-{year}.txt"
    with zipfile.ZipFile(path) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(
                f"{path.name} has no {member}. BLS adds a year's file when it publishes that year; "
                "run `cex download --overwrite` to refresh it."
            )
        text = zf.read(member).decode("latin-1")
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("*"):
            continue
        if line.startswith("2") and rows:  # a long title continues on the next line
            rows[-1]["title"] += " " + line[1:].strip()
            continue
        match = STUB_LINE.match(line)
        if not match:
            raise ValueError(f"Cannot read line {number} of {member}: {line!r}")
        level, title, code, kind, factor, section = match.groups()
        rows.append(
            {"level": int(level), "title": title, "code": code, "kind": kind, "factor": int(factor), "section": section}
        )
    return pd.DataFrame(rows)


def expenditure_tree(year: int, raw_dir: Path) -> pd.DataFrame:
    """The spending categories and items under "Average annual expenditures", in the stub's order.

    Columns: `path` (titles from the major category down to this row, as a tuple),
    `title`, `is_item` (a UCC rather than a group), `ucc`, and `factor`.
    """
    stub = read_stub(year, raw_dir)
    starts = stub.index[(stub["title"] == ROOT) & (stub["level"] == 1)]
    if len(starts) != 1:
        raise ValueError(f"Expected one {ROOT!r} entry in the {year} stub file, found {len(starts)}")
    start = starts[0]
    later_sections = stub.index[(stub.index > start) & (stub["level"] == 1)]
    end = later_sections[0] if len(later_sections) else len(stub)

    rows, titles = [], []
    for row in stub.loc[start + 1 : end - 1].itertuples():
        titles = [*titles[: row.level - 2], row.title]  # the root is level 1, so a level-2 row has a path of length 1
        if row.kind not in "GI":
            raise ValueError(f"Unexpected entry type {row.kind!r} under {ROOT!r} in the {year} stub: {row.title!r}")
        is_item = row.kind == "I"
        rows.append(
            {
                "path": tuple(titles),
                "title": row.title,
                "is_item": is_item,
                "ucc": row.code if is_item else None,
                "factor": row.factor,
            }
        )
    tree = _use_all_grocery_purchases(pd.DataFrame(rows))

    items = tree[tree["is_item"]]
    repeated = items.loc[items["ucc"].duplicated(), "ucc"].tolist()
    if repeated:
        raise ValueError(f"UCCs listed more than once under {ROOT!r} in the {year} stub: {repeated}")
    unknown = sorted({path for path in tree["path"] if len(path) == 1} - set(MAJOR_PATHS) - {("Food",)})
    if unknown or not set(MAJOR_PATHS) <= set(tree["path"]):
        raise ValueError(
            f"The major categories in the {year} stub file differ from stubs.NAMES "
            f"(unrecognized: {unknown}). BLS may have renamed one; update NAMES."
        )
    return tree


def _use_all_grocery_purchases(tree: pd.DataFrame) -> pd.DataFrame:
    """Make "Food at home" mean all grocery-store purchases (UCC 790210), in every year.

    Until early 2023 the Interview asked what households usually spend at grocery
    stores (790210) and how much of that was not food, giving food-only purchases
    (790240), which is what the stub files list. The follow-up question was then
    dropped: BLS filled in 790240 as 80% of 790210 for a year, and from Apr-Jun 2024
    it reports only 790210. No stub file lists 790210. Using it throughout is the
    one definition that is observed for every household in every year; the price
    is that this category includes the paper towels and pet food bought with the food.
    See https://www.bls.gov/cex/pumd/errata/errata-2023.htm
    """
    tree = tree[tree["ucc"] != FOOD_ONLY_GROCERY_UCC]
    if (tree["ucc"] == GROCERY_UCC).any():
        return tree.reset_index(drop=True)
    group = tree.index[tree["path"] == FOOD_AT_HOME]
    if len(group) != 1:
        raise ValueError(f"Expected one {' > '.join(FOOD_AT_HOME)} group in the stub file, found {len(group)}")
    title = "Grocery store purchases, including nonfood items"
    grocery = pd.DataFrame(
        [{"path": (*FOOD_AT_HOME, title), "title": title, "is_item": True, "ucc": GROCERY_UCC, "factor": 1}]
    )
    position = tree.index.get_loc(group[0]) + 1
    return pd.concat([tree.iloc[:position], grocery, tree.iloc[position:]], ignore_index=True)


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def category_name(path: tuple[str, ...]) -> str:
    return NAMES.get(path, _slug(path[-1]))


def major_categories(tree: pd.DataFrame) -> pd.Series:
    """Major category name for each UCC (index: UCC)."""
    items = tree[tree["is_item"]]
    paths = [path[:2] if path[0] == "Food" else path[:1] for path in items["path"]]
    return pd.Series([NAMES[path] for path in paths], index=items["ucc"], name="category")


def detail_paths(tree: pd.DataFrame) -> list[tuple[str, ...]]:
    """Where the detail cut falls: the subcategory paths, in the stub's order.

    Each major category is split into the groups directly under it (but see
    SPLIT_FURTHER and KEEP_WHOLE). An item that sits beside such groups, like
    prescription drugs under healthcare, is a subcategory of its own. A major
    category with only items under it (reading, tobacco) is not split.
    """
    is_item = dict(zip(tree["path"], tree["is_item"]))
    children: dict[tuple, list[tuple]] = {}
    for path in tree["path"]:
        children.setdefault(path[:-1], []).append(path)

    def cut(path: tuple, depth: int) -> list[tuple]:
        below = children.get(path, [])
        if is_item[path] or len(path) >= depth or path in KEEP_WHOLE or all(is_item[child] for child in below):
            return [path]
        return [leaf for child in below for leaf in cut(child, depth + (child in SPLIT_FURTHER))]

    return [leaf for major in children[()] for leaf in cut(major, 2)]


def detail_categories(tree: pd.DataFrame) -> pd.Series:
    """Detail category name for each UCC (index: UCC)."""
    cut = set(detail_paths(tree))
    items = tree[tree["is_item"]]
    names = []
    for path in items["path"]:
        ancestors = [path[:n] for n in range(1, len(path) + 1) if path[:n] in cut]
        if len(ancestors) != 1:
            raise ValueError(f"{' > '.join(path)} falls under {len(ancestors)} detail categories")
        names.append(category_name(ancestors[0]))
    return pd.Series(names, index=items["ucc"], name="category")


def detail_names(tree: pd.DataFrame) -> list[str]:
    """The detail category names, in the stub's order."""
    return list(dict.fromkeys(category_name(path) for path in detail_paths(tree)))


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").capitalize())
