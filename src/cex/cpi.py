"""Consumer Price Index for All Urban Consumers (CPI-U), for comparing dollars across years."""

# CPI-U, U.S. city average, all items, not seasonally adjusted, annual averages
# (1982-84 = 100). BLS series CUUR0000SA0. Add a year once BLS publishes its
# annual average: https://data.bls.gov/timeseries/CUUR0000SA0
CPI_U = {
    2004: 188.9,
    2005: 195.3,
    2006: 201.6,
    2007: 207.342,
    2008: 215.303,
    2009: 214.537,
    2010: 218.056,
    2011: 224.939,
    2012: 229.594,
    2013: 232.957,
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
    2021: 270.970,
    2022: 292.655,
    2023: 304.702,
    2024: 313.689,
}


def cpi(year: int) -> float:
    try:
        return CPI_U[year]
    except KeyError:
        raise KeyError(
            f"No CPI-U annual average for {year} in cex.cpi.CPI_U. "
            "Add it from https://data.bls.gov/timeseries/CUUR0000SA0"
        ) from None


def constant_dollar_factor(year: int, base_year: int) -> float:
    """Multiply `year` dollars by this to express them in `base_year` dollars."""
    return cpi(base_year) / cpi(year)
