import pandas as pd
from src.db import engine

def get_foreign_born_trend(place_fips):
    query = """
        SELECT acs_end_year,
               foreign_born_percent
        FROM foreign_born_summary
        WHERE place_fips = %s
        ORDER BY acs_end_year
    """
    return pd.read_sql(query, engine, params=[place_fips])