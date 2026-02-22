from sqlalchemy import text
import pandas as pd
from src.db import engine

# ---------------------------
# Core Utility
# ---------------------------

def run_query(query, params=None):
    """Executes a SQL query and returns a pandas DataFrame."""
    
    # Convert numpy scalars to native Python types
    if params:
        normalized = {}
        for k, v in params.items():
            if hasattr(v, "item"):   # catches numpy.int64, numpy.float64, etc.
                normalized[k] = v.item()
            else:
                normalized[k] = v
        params = normalized

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)

def compute_growth(df, value_col):
    """Calculates percentage change between first and last recorded years."""
    if df.empty or len(df) < 2:
        return None
    df = df.sort_values("year")
    start = df[value_col].iloc[0]
    end = df[value_col].iloc[-1]
    return ((end - start) / start) * 100 if start != 0 else None

# ---------------------------
# City Registry
# ---------------------------

def get_cities(gateway_only=True):
    query = """
        SELECT place_fips::text, place_name, is_gateway_city
        FROM gateway_cities
        {}
        ORDER BY place_name;
    """.format(
        "WHERE is_gateway_city = TRUE" if gateway_only else ""
    )
    return run_query(query)

# ---------------------------
# Demographic Trends
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

def get_foreign_born_by_country(place_fips, year):
    query = """
        SELECT 
            country_label_estimate AS country_label,
            NULLIF(REGEXP_REPLACE(estimate, '[^0-9.]', '', 'g'), '')::float AS foreign_born
        FROM foreign_born_by_country
        WHERE place_fips::text = :place_fips
        AND year = :year
        ORDER BY foreign_born DESC;
    """
    return run_query(query, {
        "place_fips": place_fips,
        "year": year
    })

# ---------------------------
# Economic Indicators
# ---------------------------

def get_income_trend(place_fips):
    query = """
        SELECT 
            year,
            NULLIF(REGEXP_REPLACE(estimate, '[^0-9.]', '', 'g'), '')::float AS median_income
        FROM income
        WHERE place_fips::text = :place_fips
        AND variable_label IN (
            'Estimate_Households_Median_income_(dollars)', 
            'Households_Estimate_Median_income_(dollars)'
        )
        ORDER BY year;
    """
    return run_query(query, {"place_fips": place_fips})

def get_poverty_trend(place_fips):
    query = """
        SELECT year, poverty_rate::float
        FROM poverty_status
        WHERE place_fips::text = :place_fips
        ORDER BY year;
    """
    return run_query(query, {"place_fips": place_fips})

def get_gini_trend(place_fips):
    query = """
        SELECT year, gini_index::float
        FROM gini_index
        WHERE place_fips::text = :place_fips
        ORDER BY year;
    """
    return run_query(query, {"place_fips": place_fips})

# ---------------------------
# Housing & Employment
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

def get_owner_renter_breakdown(place_fips, year):
    """Provides the distribution of home ownership vs. rental status."""
    query = """
        SELECT 
            tenure_label,
            estimate::float
        FROM owner_vs_renter
        WHERE place_fips::text = :place_fips
        AND year = :year;
    """
    return run_query(query, {"place_fips": place_fips, "year": year})

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
# Investigative Synthesis
# ---------------------------

def get_correlation_dataset(place_fips):
    query = """
        SELECT 
            fb.year,
            (fb.foreign_born_total::float / tp.total_pop::float) * 100 AS foreign_born_percent,
            NULLIF(REGEXP_REPLACE(inc.estimate, '[^0-9.]', '', 'g'), '')::float AS median_income,
            pov.poverty_rate::float AS poverty_rate
        FROM foreign_born_total fb
        JOIN total_population tp
            ON fb.place_fips::text = tp.place_fips::text AND fb.year = tp.year
        JOIN income inc
            ON fb.place_fips::text = inc.place_fips::text AND fb.year = inc.year
            AND inc.variable_label IN (
                'Estimate_Households_Median_income_(dollars)', 
                'Households_Estimate_Median_income_(dollars)'
            )
        JOIN poverty_status pov
            ON fb.place_fips::text = pov.place_fips::text AND fb.year = pov.year
        WHERE fb.place_fips::text = :place_fips
        ORDER BY fb.year;
    """
    return run_query(query, {"place_fips": place_fips})

def get_foreign_born_growth_all():
    """Aggregates first-and-last year values across all municipalities for comparison."""
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