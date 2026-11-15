# Working with Consumer Expenditure data: a guide to this repo

This is a walk through the Consumer Expenditure Survey (CE) Interview microdata for someone who wants to model with it. Each section says what you need to know, what goes wrong if you don't, and where this package handles it. The order follows the pipeline: files, time, weights, spending, households, models, prices.

The numbers quoted are for 2024 unless noted and come from the files under `outputs/`.

## 1. What the survey is

BLS runs two surveys under the CE name:

- The **Interview** survey asks households about the last three months, mainly larger and regular expenses: rent, utilities, vehicles, insurance, healthcare. It covers most spending.
- The **Diary** survey has households record every purchase for two weeks. BLS uses it for small, frequent items, chiefly food and apparel.

BLS's published tables combine the two. This package uses the Interview alone, so its numbers match the published ones only for categories BLS takes from the Interview (section 4 shows how closely).

The unit is the **consumer unit** (CU): people who live together and share finances, roughly a household. A CU is interviewed up to four times, three months apart, then leaves the sample. So the data are a short rotating panel: every quarter about a quarter of the CUs are new.

Identifiers:

- `NEWID` identifies one interview. Its last digit is the interview number (1 to 4) and the rest is the CU's id (`CUID`).
- The package keeps both, as `newid` and `cuid` ([clean.py](../src/cex/clean.py)).

## 2. The files

`cex download` fetches three things into `data/raw/`:

| File | What it is |
|---|---|
| `intrvwYY.zip` | One year's Interview release: quarterly csv files of several types |
| `stubs.zip` | BLS's category definitions for every year (section 4) |
| `prices.csv` | CPI series from BLS's public API (section 7) |

Inside a release, the file types this package reads are:

| Type | One row per | Holds |
|---|---|---|
| FMLI | interview | Household traits, income, weights, and ready-made spending totals |
| MTBI | interview × month × item | Every expense, under a six-digit item code (UCC) |
| ITBI | interview × month × item | Income by source, and a few deductions that count as spending |

There are more (MEMI has one row per household member; the `expn` folder has the raw answers to each questionnaire section). They are not used here.

**Pitfall: a release is not a calendar year.** A year of spending needs five quarterly files, because interviews in January to March of the next year still ask about this year. The 2019 and 2024 zips hold all five. The 2020 to 2023 zips hold only four: their January to March files are the last files of the *previous* year's zip. Reading one zip per year silently drops a quarter. `load.quarterly_files` collects the right five ([load.py](../src/cex/load.py)).

## 3. Time: recall periods and calendar years

An interview in month *m* covers months *m* − 3 to *m* − 1. Someone interviewed in February 2024 reports November 2023 through January 2024. Two consequences:

- **Annual estimates count only the months that fall in the year.** That February interview contributes one month to 2024. The package calls this `months_in_scope` (0 to 3) and builds two spending measures for every category:
  - `<category>_in_year`: spending in the recall months that fall in the calendar year, for annual means;
  - `<category>_annual`: spending over all three recall months, times 4, for models.
- **The population is counted the same way.** An interview's weight stands for a number of households for one quarter. Summing `weight × months_in_scope / 12` over a year's five files gives the number of consumer units: 135.76 million, which is BLS's published figure.

MTBI records carry their reference month and year, so the in-year part is exact. FMLI's ready-made totals split each amount into `PQ` (previous calendar quarter) and `CQ` (current), which gives the same split.

## 4. Spending: summary variables, item records, and the category tree

There are two ways to get spending by category.

**The quick way** is FMLI's summary variables: `TOTEXPPQ`/`TOTEXPCQ` for the total, `HOUSPQ`/`HOUSCQ` for housing, and so on. For a first look at one year they are fine.

**The careful way** is to add up the MTBI and ITBI records yourself, using BLS's category tree. This package does that, because the summary variables turned out not to be consistent over time:

| Problem | Where | Size |
|---|---|---|
| Food at home changes from food only (`FDHOME`) to everything bought at grocery stores (`GROCER`) | from Apr–Jun 2024 | +25% in that category |
| Education is nearly empty | Apr–Jun 2021 to Jan–Mar 2023 | about −$1,100 a year |
| Items in BLS's published categories are left out (laundry and dry cleaning, jewelry, some household goods) | every year | about 1% of total spending |

`outputs/<year>/fmli_check.csv` compares the two, quarter by quarter. The run prints the categories that differ.

Rebuilding from items fixes definitions, not the questionnaire. In April 2023 BLS shortened the interview, and some questions changed meaning. Dining out now includes alcohol bought with the meal: food away from home jumps about 30% that quarter, alcohol away from home falls about 65%, and alcohol at home rises. No recoding removes that. **Before reading a trend, plot the category by quarter and look for steps.**

### The category tree

`stubs.zip` has one text file per year listing BLS's published categories as a tree, with the UCCs under each one:

```
Average annual expenditures
  Housing
    Shelter
      Rented dwellings
        210110  Rent
```

[stubs.py](../src/cex/stubs.py) reads the tree and cuts it at two depths:

- the **15 major categories** used throughout (groceries, food away, alcohol, housing, apparel, transportation, healthcare, entertainment, personal care, reading, education, tobacco, miscellaneous, cash contributions, insurance and pensions);
- **31 detail categories** (rent, owned dwellings, utilities, gasoline, vehicle purchases, health insurance, and so on).

Because both come from the same tree, the detail categories add up to the major ones and to the total exactly. `outputs/<year>/detail_means.csv` goes all the way down to single items.

### Things the tree does not tell you

Found by comparing against the summary variables and the published tables ([spending.py](../src/cex/spending.py)):

- **Some spending is in ITBI, the income file.** Deductions for Social Security and pensions are recorded there. Without them, "insurance and pensions" comes out at 6% of its true size.
- **A few items carry a factor of 4.** Finance charges on credit cards and loans are asked only in a household's fourth interview, about the past year. Only a quarter of interviews carry them, so the tree says to count them four times.
- **Groceries are missing from the tree.** The Interview used to ask what a household spends at grocery stores (UCC 790210) and how much of that was not food, giving food only (790240). The second question was dropped in 2023, and no stub file lists 790210. This package defines the category as all grocery-store purchases in every year, the only version observed for every household. It is named `groceries` rather than "food at home" for that reason: it includes the paper towels and pet food.
- **Categories get retitled.** "Gasoline and motor oil" became "Gasoline, other fuels, and motor oil", then "Gasoline and other fuels". `stubs._CATEGORIES` maps every title to one name.
- **MTBI holds more than spending.** Home values, rental values, and asset purchases are in there too. Only UCCs under "Average annual expenditures" count.

### How well it matches BLS

`tests/test_bls_published.py` compares each year with BLS's published table:

| | Agreement, 2019–2024 |
|---|---|
| Consumer units | exact |
| Income before taxes | within 0.7% |
| Cash contributions, 2019–2023 | to the dollar |
| Insurance and pensions | within 0.3% |
| Transportation | within 2% |
| Total spending | 3% below (2019) to 1% above (2024) |

Cash contributions and insurance come only from the Interview, so matching them to the dollar confirms the weights, the calendar-year logic, the choice of files, and the item-level rebuild all at once. The total differs because BLS publishes food, apparel, and small household items from the Diary.

## 5. Weights and uncertainty

**Weights.** `FINLWT21` is the number of households an interview represents. Use it for everything. Unweighted results describe the sample, which over-represents some areas.

**Sampling error.** The sample is clustered, so textbook standard errors are too small. BLS provides 44 **replicate weights** (`WTREP01`–`WTREP44`). Each keeps about half the sample at double weight and drops the rest. The recipe ([survey.py](../src/cex/survey.py)):

1. Compute your estimate with the main weight.
2. Recompute it with each replicate weight.
3. Variance = the sum of squared differences from step 1, divided by 44.

This works for any statistic: a mean, a ratio, a regression coefficient, an elasticity. Every estimate in this package gets its standard error this way.

**Imputed income.** About four in ten interviews have some income filled in by BLS, which does so five times (`FINCBTX1`–`5`) to show how uncertain the fill-in is. Anything that uses income is estimated once per version and combined with Rubin's rules: average the five estimates; the variance is the average replicate variance plus 1.2 times the variance across versions. Using the single averaged variable `FINCBTXM` understates the uncertainty.

**What the public file hides.** Large values are top-coded (replaced by the average of the values above a cutoff). Means are barely affected; the tails of distributions are.

## 6. From interviews to households

One quarter is a noisy picture of a household. Share of interviews with no spending at all in the category:

| Category | Zero in a quarter |
|---|---|
| Vehicle purchases | 95% |
| Education | 89% |
| Tobacco | 85% |
| Alcohol | 59% |
| Cash contributions | 54% |
| Apparel | 18% |
| Groceries | 1% |

A car bought this quarter makes the household look rich in transportation and in total; the other three quarters make it look poor in both. Two responses, both used here:

- **Average each household's interviews** (`clean.household_year`). The models run on one row per CU: its annualized spending averaged over the interviews it has in the year's files, with weights summed accordingly.
- **Instrument total spending** (next section), because averaging does not remove the problem.

A caveat on the averaging: within one year's five files only about 10% of CUs have all four interviews and 44% have just one, because a CU's year in the sample usually straddles two calendar years. Linking CUs across years by `cuid` would give full four-interview households. This package does not do that.

## 7. Models

All in [model.py](../src/cex/model.py), all on household-year data with the same controls (log family size, age, education, region, tenure, urban, sex of the reference person, children, seniors).

### Engel curves: necessities and luxuries

The question: when a household's budget is 1% larger, by what percent does its spending on a category grow? That is the **expenditure elasticity**. Below 1, the category's share of the budget falls as the budget grows (a necessity); above 1, it rises (a luxury).

The model is Working–Leser:

```
budget share of category = a + b × log(total spending) + controls
elasticity = 1 + b / (average budget share)
```

Three things to understand about it:

- **Shares add up to 1, so the *b*'s add up to 0** and the share-weighted elasticities average exactly 1. That is why 1 is the dividing line. It holds for the 15 major categories and for the 31 detail categories.
- **Zeros are not a problem.** A household that bought no tobacco has a share of 0 and stays in. A log-log regression would have to drop it.
- **Total spending must be instrumented.** Measured total spending contains the same noise as its parts: the car purchase raises both transportation and the total, which makes transportation look like more of a luxury than it is. Income moves a household's long-run spending but is unrelated to which quarter the car was bought, so log income serves as the instrument (two-stage least squares; the first-stage F statistic is about 1,200).

Results for 2024 (`outputs/2024/elasticities.csv`): groceries 0.28, housing 0.83, and healthcare 0.90 are necessities; insurance and pensions 2.27, cash contributions 1.70, and entertainment 1.35 are luxuries. Tobacco is negative: households with bigger budgets spend less on it.

### Income elasticities

The same question with respect to *income* instead of total spending, estimated by Poisson pseudo-maximum likelihood (PPML), which models the mean of spending directly and so also keeps the zeros.

These come out much smaller. Total spending itself has an income elasticity of only 0.43, because households smooth: a good or bad year moves income more than spending, and reported income is noisy. So do not compare an income elasticity with 1. Compare it with 0.43, the value for total spending.

### Curvature

`fit_quadratic` adds the square of log total spending, so that a category can be a luxury for small budgets and a necessity for large ones. Both the level and the square are instrumented (with log income and its square). Squaring the fitted value from a first stage instead is a classic mistake, known as the forbidden regression.

In 2024 the curvature is clear for transportation (elasticity 1.45 at the 10th percentile of spending, 0.72 at the 90th) and healthcare (1.14 to 0.56): things that poorer households stretch to afford and richer ones already have. See `outputs/2024/engel_curvature.png`.

A limit of this model family: nothing keeps a predicted share above zero, and the elasticity divides by the share. Where the predicted share nearly vanishes, the package reports no elasticity rather than a wild number.

## 8. Adding prices: whose inflation?

The CE is where the CPI's weights come from. [inflation.py](../src/cex/inflation.py) turns that around: reprice each *group's* basket and see whose cost of living rose most.

```
inflation for group g, year t to t+1 =
    sum over categories of (g's budget share in t) × (the category's price change)
```

The price changes are national CPI series matched to the detail categories ([prices.py](../src/cex/prices.py)). Chaining six years, 2019 to 2025 (`outputs/2019-2024/inflation.csv`), gives 27.0% for all consumer units, against 25.9% for the published CPI. The lowest income fifth is about 1 point above the average and the highest about 1 point below. The largest single reason is rent, which takes more of a small budget and rose faster than average; utilities come second. The top fifth spends more on things whose prices rose less: education, travel, lodging. `inflation_contributions.csv` breaks each gap down by category.

Choices that matter:

- **Owners' housing is priced at rental value.** The CPI treats an owned home as something the owner rents from themselves. The interview asks owners what their home would rent for (`RENTEQVX`); that amount replaces mortgage interest, property tax, and upkeep in the basket. Without this, older owners with no mortgage appear to have almost no housing costs.
- **Some spending has no price.** Cash contributions, life insurance, and pension contributions are left out, as in the CPI.
- **Health insurance is priced with the medical care index**, not the CPI's own health insurance index, which tracks only insurers' margins and is far more volatile.

What it cannot show: every group is assumed to pay the same price change within a category. Differences in where people shop or which varieties they buy are invisible here.

## 9. What this data is not good for

- **Price elasticities.** The survey records dollars, not quantities or prices. Outside prices vary only by region and month, and so does everything else that changes over time. A first attempt at a gasoline demand curve for this repo gave answers that depended entirely on the time controls, so it was left out.
- **Detailed food or clothing.** Those are in the Diary survey.
- **High incomes and large purchases.** Top-coding and a modest sample (under 5,000 interviews a quarter in 2024) limit what can be said about the tails.
- **Long household histories.** Each household is seen for at most a year.
- **Wealth and saving.** Assets are asked once, coarsely. Income minus spending is a poor measure of saving, because both are measured with error and "spending" here includes pension contributions.

## 10. Things to try next

Roughly in order of difficulty:

1. Change the detail cut. `stubs.SPLIT_FURTHER` controls which groups are split one level deeper; add utilities and look at electricity against telephone service.
2. Add a control or a grouping. `clean_fmli` is where FMLI columns become household traits; `summarize.GROUPINGS` lists what the tables are broken down by.
3. Compare consumption and income inequality. Compute a weighted Gini of `total_annual` and of income on household-year data, with replicate-weight standard errors.
4. Link households across years by `cuid` so that the models use all four interviews.
5. Use the member file (MEMI) to build household composition in more detail and estimate how spending scales with family size.
6. Bring in the Diary survey and reproduce a published table exactly. The `CE-HG-Integ-YYYY.txt` files in `stubs.zip` say which survey each item is published from.
