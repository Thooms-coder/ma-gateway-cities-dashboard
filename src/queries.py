from sqlalchemy import text
import pandas as pd
from src.db import engine


# ---------------------------
# Core Utility
# ---------------------------

def run_query(query, params=None):
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


# ---------------------------
# City List
# ---------------------------

def get_cities(gateway_only=True):
    query = """
        SELECT place_fips::text, place_name
        FROM gateway_cities
        {}
        ORDER BY place_name;
    """.format(
        "WHERE is_gateway_city = TRUE" if gateway_only else ""
    )

    return run_query(query)


# ---------------------------
# Foreign Born % Trend
# ---------------------------

def get_foreign_born_percent(place_fips):
    query = """
        SELECT 
            fb.year,
            fb.foreign_born_total::float,
            tp.total_pop::float,
            (fb.foreign_born_total::float / tp.total_pop::float) * 100 AS foreign_born_percent
        FROM foreign_born_total fb
        JOIN total_population tp
            ON fb.place_fips::text = tp.place_fips::text
            AND fb.year = tp.year
        WHERE fb.place_fips::text = :place_fips
        ORDER BY fb.year;
    """

    return run_query(query, {"place_fips": place_fips})


# ---------------------------
# Foreign Born by Country
# ---------------------------

def get_foreign_born_by_country(place_fips, year):
    query = """
        SELECT 
            country_label,
            estimate::float AS foreign_born
        FROM foreign_born_by_country
        WHERE place_fips::text = :place_fips
        AND year = :year
        ORDER BY foreign_born DESC;
    """

    return run_query(query, {"place_fips": place_fips, "year": year})


# ---------------------------
# Income (Median Household)
# ---------------------------

def get_income_trend(place_fips):
    query = """
        SELECT 
            year,
            estimate::float AS median_income
        FROM income
        WHERE place_fips::text = :place_fips
        AND variable_label ILIKE '%median%household%income%'
        ORDER BY year;
    """
    return run_query(query, {"place_fips": place_fips})


# ---------------------------
# Poverty Trend
# ---------------------------

def get_poverty_trend(place_fips):
    query = """
        SELECT 
            year,
            poverty_rate::float
        FROM poverty_status
        WHERE place_fips::text = :place_fips
        ORDER BY year;
    """

    return run_query(query, {"place_fips": place_fips})


# ---------------------------
# Gini Trend
# ---------------------------

def get_gini_trend(place_fips):
    query = """
        SELECT 
            year,
            gini_index::float
        FROM gini_index
        WHERE place_fips::text = :place_fips
        ORDER BY year;
    """

    return run_query(query, {"place_fips": place_fips})


# ---------------------------
# Rent Burden %
# ---------------------------

def get_rent_burden_percent(place_fips):
    query = """
        SELECT 
            year,
            (rent_burdened_30plus::float / total_renters::float) * 100 AS rent_burden_percent
        FROM rent_burden
        WHERE place_fips::text = :place_fips
        ORDER BY year;
    """

    return run_query(query, {"place_fips": place_fips})


# ---------------------------
# Employment Rate
# ---------------------------

def get_employment_rate(place_fips):
    query = """
        SELECT 
            year,
            (employed::float / pop_16plus::float) * 100 AS employment_rate
        FROM employment_status
        WHERE place_fips::text = :place_fips
        ORDER BY year;
    """

    return run_query(query, {"place_fips": place_fips})


# ---------------------------
# Housing Tenure Breakdown
# ---------------------------

def get_owner_renter_breakdown(place_fips, year):
    query = """
        SELECT 
            tenure_label,
            estimate
        FROM owner_vs_renter
        WHERE place_fips::text = :place_fips
        AND year = :year;
    """

    return run_query(query, {"place_fips": place_fips, "year": year})


# ---------------------------
# Growth Rate Utility
# ---------------------------

def compute_growth(df, value_col):
    df = df.sort_values("year")
    start = df[value_col].iloc[0]
    end = df[value_col].iloc[-1]
    return ((end - start) / start) * 100 if start != 0 else None


# ---------------------------
# City-Level Growth Comparison
# ---------------------------

def get_foreign_born_growth_all():
    query = """
        SELECT 
            place_fips::text,
            MIN(year) AS start_year,
            MAX(year) AS end_year,
            MIN(foreign_born_total) FILTER (WHERE year = (SELECT MIN(year) FROM foreign_born_total)) AS start_value,
            MAX(foreign_born_total) FILTER (WHERE year = (SELECT MAX(year) FROM foreign_born_total)) AS end_value
        FROM foreign_born_total
        GROUP BY place_fips;
    """

    return run_query(query)


# ---------------------------
# Correlation Dataset Builder
# ---------------------------

def get_correlation_dataset(place_fips):
    query = """
        SELECT 
            fb.year,
            (fb.foreign_born_total::float / tp.total_pop::float) * 100 AS foreign_born_percent,
            inc.estimate::float AS median_income,
            pov.poverty_rate::float AS poverty_rate
        FROM foreign_born_total fb
        JOIN total_population tp
            ON fb.place_fips::text = tp.place_fips::text
            AND fb.year = tp.year
        JOIN income inc
            ON fb.place_fips::text = inc.place_fips::text
            AND fb.year = inc.year
            AND inc.variable_label ILIKE '%Median household income%'
        JOIN poverty_status pov
            ON fb.place_fips::text = pov.place_fips::text
            AND fb.year = pov.year
        WHERE fb.place_fips::text = :place_fips
        ORDER BY fb.year;
    """

    return run_query(query, {"place_fips": place_fips})