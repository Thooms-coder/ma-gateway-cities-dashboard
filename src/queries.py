from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from sqlalchemy import text

from src.db import engine


# ============================================================
# Core Utility
# ============================================================

def _normalize_params(params: Optional[dict]) -> Optional[dict]:
    """Convert numpy scalars (and similar) to native Python types."""
    if not params:
        return params
    normalized = {}
    for k, v in params.items():
        if hasattr(v, "item"):
            normalized[k] = v.item()
        else:
            normalized[k] = v
    return normalized


def run_query(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Executes a SQL query and returns a pandas DataFrame."""
    params = _normalize_params(params)
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def compute_growth(df: pd.DataFrame, year_col: str, value_col: str) -> Optional[float]:
    """Percent change between first and last rows (ordered by year_col)."""
    if df is None or df.empty or len(df) < 2:
        return None
    d = df.sort_values(year_col)
    start = d[value_col].iloc[0]
    end = d[value_col].iloc[-1]
    if pd.isna(start) or start == 0 or pd.isna(end):
        return None
    return ((end - start) / start) * 100.0


# ============================================================
# Table Names (single source of truth)
# ============================================================

ACS_PLACE = "public.acs_place_data"
ACS_STATE = "public.acs_state_data"
CITY_REGISTRY = "public.gateway_cities"


# ============================================================
# City Registry
# ============================================================

def get_cities(gateway_only: bool = True) -> pd.DataFrame:
    query = f"""
        SELECT place_fips::text, place_name, is_gateway_city
        FROM {CITY_REGISTRY}
        {"WHERE is_gateway_city = TRUE" if gateway_only else ""}
        ORDER BY place_name;
    """
    return run_query(query)


def get_gateway_fips() -> pd.DataFrame:
    query = f"""
        SELECT place_fips::text
        FROM {CITY_REGISTRY}
        WHERE is_gateway_city = TRUE;
    """
    return run_query(query)


# ============================================================
# Warehouse Discovery / Metadata
# ============================================================

def get_years(scope: str = "place") -> pd.DataFrame:
    """
    Returns available ACS end years.
    scope: 'place' or 'state'
    """
    table = ACS_PLACE if scope == "place" else ACS_STATE
    query = f"""
        SELECT DISTINCT acs_end_year, acs_period
        FROM {table}
        ORDER BY acs_end_year;
    """
    return run_query(query)


def get_source_tables(scope: str = "place") -> pd.DataFrame:
    table = ACS_PLACE if scope == "place" else ACS_STATE
    query = f"""
        SELECT source_table, COUNT(DISTINCT variable_id) AS variables
        FROM {table}
        GROUP BY source_table
        ORDER BY source_table;
    """
    return run_query(query)


def list_variables(
    source_table: Optional[str] = None,
    unit: Optional[str] = None,
    is_percent: Optional[bool] = None,
    scope: str = "place",
    limit: int = 5000,
) -> pd.DataFrame:
    """
    Lists distinct variables with labels and metadata.
    Useful for building dropdowns / mapping dictionaries.
    """
    table = ACS_PLACE if scope == "place" else ACS_STATE

    where = ["1=1"]
    params = {}

    if source_table:
        where.append("source_table = :source_table")
        params["source_table"] = source_table

    if unit:
        where.append("unit = :unit")
        params["unit"] = unit

    if is_percent is not None:
        where.append("is_percent = :is_percent")
        params["is_percent"] = is_percent

    query = f"""
        SELECT
            variable_id,
            MAX(variable_label) AS variable_label,
            MAX(unit) AS unit,
            MAX(is_percent) AS is_percent,
            MAX(source_table) AS source_table
        FROM {table}
        WHERE {" AND ".join(where)}
        GROUP BY variable_id
        ORDER BY source_table, variable_id
        LIMIT :limit;
    """
    params["limit"] = int(limit)
    return run_query(query, params)


def search_variables(
    needle: str,
    scope: str = "place",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Search variable labels (ILIKE) for building UI search boxes.
    """
    table = ACS_PLACE if scope == "place" else ACS_STATE
    query = f"""
        SELECT
            variable_id,
            variable_label,
            unit,
            is_percent,
            source_table
        FROM {table}
        WHERE variable_label ILIKE :q
        GROUP BY variable_id, variable_label, unit, is_percent, source_table
        ORDER BY source_table, variable_id
        LIMIT :limit;
    """
    return run_query(query, {"q": f"%{needle}%", "limit": int(limit)})


# ============================================================
# Core Fetchers (single variable)
# ============================================================

def get_place_variable_trend(
    place_fips: str,
    variable_id: str,
    include_moe: bool = True,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Time series for one variable_id for one place.
    Returns columns: acs_end_year, estimate, moe(optional), unit, is_percent, variable_label, source_table
    """
    where = [
        "place_fips::text = :place_fips",
        "variable_id = :variable_id",
    ]
    params = {"place_fips": str(place_fips), "variable_id": variable_id}

    if start_year is not None:
        where.append("acs_end_year >= :start_year")
        params["start_year"] = int(start_year)
    if end_year is not None:
        where.append("acs_end_year <= :end_year")
        params["end_year"] = int(end_year)

    cols = """
        acs_end_year,
        estimate::float AS estimate,
        unit,
        is_percent,
        variable_label,
        source_table
    """
    if include_moe:
        cols = cols.replace("estimate::float AS estimate,", "estimate::float AS estimate, moe::float AS moe,")

    query = f"""
        SELECT {cols}
        FROM {ACS_PLACE}
        WHERE {" AND ".join(where)}
        ORDER BY acs_end_year;
    """
    return run_query(query, params)

def get_place_source_table_year(
    place_fips: str,
    source_table: str,
    year: int,
    include_moe: bool = True,
) -> pd.DataFrame:
    """
    Fetch ALL variables for a given source_table
    for a specific place and ACS year.

    Used for:
        - B05006 country-of-origin breakdown
        - full table inspection
        - hierarchical variable extraction

    Returns:
        acs_end_year,
        variable_id,
        estimate,
        (optional) moe,
        unit,
        is_percent,
        variable_label,
        source_table
    """

    cols = """
        acs_end_year,
        variable_id,
        estimate::float AS estimate,
        unit,
        is_percent,
        variable_label,
        source_table
    """

    if include_moe:
        cols = cols.replace(
            "estimate::float AS estimate,",
            "estimate::float AS estimate, moe::float AS moe,"
        )

    query = f"""
        SELECT {cols}
        FROM {ACS_PLACE}
        WHERE place_fips::text = :place_fips
          AND source_table = :source_table
          AND acs_end_year = :year
        ORDER BY variable_id;
    """

    return run_query(
        query,
        {
            "place_fips": str(place_fips),
            "source_table": source_table,
            "year": int(year),
        },
    )

def get_state_variable_trend(
    state_fips: str = "25",
    variable_id: str = "",
    include_moe: bool = True,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Time series for one variable_id for the state.
    state_fips defaults to Massachusetts "25".
    """
    where = [
        "place_fips::text = :state_fips",
        "variable_id = :variable_id",
    ]
    params = {"state_fips": str(state_fips), "variable_id": variable_id}

    if start_year is not None:
        where.append("acs_end_year >= :start_year")
        params["start_year"] = int(start_year)
    if end_year is not None:
        where.append("acs_end_year <= :end_year")
        params["end_year"] = int(end_year)

    cols = """
        acs_end_year,
        estimate::float AS estimate,
        unit,
        is_percent,
        variable_label,
        source_table
    """
    if include_moe:
        cols = cols.replace("estimate::float AS estimate,", "estimate::float AS estimate, moe::float AS moe,")

    query = f"""
        SELECT {cols}
        FROM {ACS_STATE}
        WHERE {" AND ".join(where)}
        ORDER BY acs_end_year;
    """
    return run_query(query, params)


def get_place_latest_value(place_fips: str, variable_id: str) -> pd.DataFrame:
    """
    Latest observation for a place and variable.
    """
    query = f"""
        SELECT
            acs_end_year,
            estimate::float AS estimate,
            moe::float AS moe,
            unit,
            is_percent,
            variable_label,
            source_table
        FROM {ACS_PLACE}
        WHERE place_fips::text = :place_fips
          AND variable_id = :variable_id
        ORDER BY acs_end_year DESC
        LIMIT 1;
    """
    return run_query(query, {"place_fips": str(place_fips), "variable_id": variable_id})


def get_state_latest_value(state_fips: str, variable_id: str) -> pd.DataFrame:
    query = f"""
        SELECT
            acs_end_year,
            estimate::float AS estimate,
            moe::float AS moe,
            unit,
            is_percent,
            variable_label,
            source_table
        FROM {ACS_STATE}
        WHERE place_fips::text = :state_fips
          AND variable_id = :variable_id
        ORDER BY acs_end_year DESC
        LIMIT 1;
    """
    return run_query(query, {"state_fips": str(state_fips), "variable_id": variable_id})


# ============================================================
# City vs State Comparison (same variable)
# ============================================================

def get_place_vs_state_trend(
    place_fips: str,
    variable_id: str,
    state_fips: str = "25",
    include_moe: bool = True,
) -> pd.DataFrame:
    """
    Joins place trend and state trend on year for the same variable_id.
    Returns acs_end_year, city_value, state_value (and optional moe columns).
    """
    city_cols = "p.estimate::float AS city_value"
    state_cols = "s.estimate::float AS state_value"
    moe_cols = ""
    if include_moe:
        moe_cols = ", p.moe::float AS city_moe, s.moe::float AS state_moe"

    query = f"""
        SELECT
            p.acs_end_year,
            {city_cols},
            {state_cols}
            {moe_cols},
            p.unit,
            p.is_percent,
            p.variable_label,
            p.source_table
        FROM {ACS_PLACE} p
        JOIN {ACS_STATE} s
          ON p.acs_end_year = s.acs_end_year
         AND p.variable_id = s.variable_id
        WHERE p.place_fips::text = :place_fips
          AND s.place_fips::text = :state_fips
          AND p.variable_id = :variable_id
        ORDER BY p.acs_end_year;
    """
    return run_query(query, {"place_fips": str(place_fips), "state_fips": str(state_fips), "variable_id": variable_id})


# ============================================================
# Multi-variable Panels (correlation/scatter-ready)
# ============================================================

def get_panel(
    place_fips: str,
    variables: Sequence[str],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns a wide panel by year for a set of variables for one place.
    Output: one row per year, columns named by variable_id with values = estimate
    """
    if not variables:
        return pd.DataFrame()

    where = [
        "place_fips::text = :place_fips",
        "variable_id = ANY(:vars)",
    ]
    params = {"place_fips": str(place_fips), "vars": list(variables)}

    if start_year is not None:
        where.append("acs_end_year >= :start_year")
        params["start_year"] = int(start_year)
    if end_year is not None:
        where.append("acs_end_year <= :end_year")
        params["end_year"] = int(end_year)

    query = f"""
        SELECT acs_end_year, variable_id, estimate::float AS estimate
        FROM {ACS_PLACE}
        WHERE {" AND ".join(where)}
        ORDER BY acs_end_year, variable_id;
    """
    long_df = run_query(query, params)
    if long_df.empty:
        return long_df

    wide = (
        long_df.pivot_table(index="acs_end_year", columns="variable_id", values="estimate", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values("acs_end_year")
    )
    return wide


def get_scatter_dataset(
    place_fips: str,
    x_variable: str,
    y_variable: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns a year-aligned scatter dataset for a place with x and y variables.
    """
    panel = get_panel(place_fips, [x_variable, y_variable], start_year=start_year, end_year=end_year)
    if panel.empty:
        return panel
    # Ensure both exist
    for v in [x_variable, y_variable]:
        if v not in panel.columns:
            return pd.DataFrame()
    return panel[["acs_end_year", x_variable, y_variable]].rename(
        columns={"acs_end_year": "year", x_variable: "x", y_variable: "y"}
    )


# ============================================================
# Cross-city Analysis (rankings / distributions)
# ============================================================

def get_top_n_places_for_variable(
    variable_id: str,
    year: int,
    n: int = 25,
    gateway_only: bool = True,
    descending: bool = True,
) -> pd.DataFrame:
    """
    Ranks places by a variable for a given year.
    Joins to gateway_cities for place_name and gateway filter.
    """
    order = "DESC" if descending else "ASC"
    where_gateway = "AND gc.is_gateway_city = TRUE" if gateway_only else ""

    query = f"""
        SELECT
            d.place_fips::text,
            gc.place_name,
            d.acs_end_year,
            d.estimate::float AS estimate,
            d.unit,
            d.is_percent,
            d.variable_label,
            d.source_table
        FROM {ACS_PLACE} d
        JOIN {CITY_REGISTRY} gc
          ON d.place_fips::text = gc.place_fips::text
        WHERE d.acs_end_year = :year
          AND d.variable_id = :variable_id
          {where_gateway}
        ORDER BY d.estimate {order} NULLS LAST
        LIMIT :n;
    """
    return run_query(query, {"year": int(year), "variable_id": variable_id, "n": int(n)})


def get_distribution_for_variable(
    variable_id: str,
    year: int,
    gateway_only: bool = True,
) -> pd.DataFrame:
    query = f"""
        SELECT
            d.place_fips::text,
            gc.place_name,
            d.acs_end_year,
            d.estimate::float AS estimate,
            d.unit,
            d.is_percent,
            d.variable_label,
            d.source_table
        FROM {ACS_PLACE} d
        JOIN {CITY_REGISTRY} gc
          ON d.place_fips::text = gc.place_fips::text
        WHERE 1=1
          {"AND gc.is_gateway_city = TRUE" if gateway_only else ""}
          AND d.acs_end_year = :year
          AND d.variable_id = :variable_id
        ORDER BY d.estimate NULLS LAST;
    """
    return run_query(query, {"year": int(year), "variable_id": variable_id})

# ============================================================
# Convenience: "classic" metrics using variable_ids
# ============================================================
# These map legacy ACS variable_ids to readable metric keys.
# Only include variables that are safe for journalist use.

DEFAULT_VARIABLES: Dict[str, str] = {

    # =========================
    # Population
    # =========================
    "total_population": "B01003_001E",
    "total_population_profile": "S0101_C01_001E",

    # =========================
    # Income & Poverty
    # =========================
    "median_household_income": "S1901_C01_012E",
    "poverty_rate": "S1701_C03_001E",
    "child_poverty_rate": "S1701_C03_002E",

    # =========================
    # Inequality
    # =========================
    "gini_index": "B19083_001E",

    # =========================
    # Labor Market
    # =========================
    "unemployment_rate": "B23025_005E",   # used in combination with labor force
    "labor_force_total": "B23025_003E",

    # =========================
    # Housing
    # =========================
    "median_home_value": "B25077_001E",
    "rent_burden_base": "B25070_001E",   # used for derived burden calculations
    "percent_renters": "B25003_002E",
    "vacant_units": "B25002_003E",

    # =========================
    # EDUCATION (S1501)
    # =========================
    # Percent high school graduate or higher (25+)
    "hs_or_higher_pct": "S1501_C01_014E",

    # Percent bachelor's degree or higher (25+)
    "ba_or_higher_pct": "S1501_C01_015E",

    # Median earnings (25+ with earnings)
    "median_earnings_25plus": "S1501_C01_059E",
}

def get_metric_trend(place_fips: str, metric_key: str) -> pd.DataFrame:
    """
    Wrapper: metric_key must exist in DEFAULT_VARIABLES.
    Returns ACS-based trend (raw warehouse).
    """
    if metric_key not in DEFAULT_VARIABLES:
        raise KeyError(
            f"Unknown metric_key '{metric_key}'. "
            "Add it to DEFAULT_VARIABLES or use Gateway metric functions."
        )
    return get_place_variable_trend(place_fips, DEFAULT_VARIABLES[metric_key])


def get_metric_vs_state(
    place_fips: str,
    metric_key: str,
    state_fips: str = "25"
) -> pd.DataFrame:
    """
    Returns place vs state comparison for raw ACS variables.
    """
    if metric_key not in DEFAULT_VARIABLES:
        raise KeyError(
            f"Unknown metric_key '{metric_key}'. "
            "Add it to DEFAULT_VARIABLES or use Gateway metric functions."
        )
    return get_place_vs_state_trend(
        place_fips,
        DEFAULT_VARIABLES[metric_key],
        state_fips=state_fips
    )

# ============================================================
# Guardrails: Quick sanity checks (optional)
# ============================================================

def sanity_check_place_key_duplicates(limit: int = 20) -> pd.DataFrame:
    """
    Detect duplicate (place_fips, year, variable_id) rows.
    Should always return empty in a clean warehouse.
    """
    query = f"""
        SELECT place_fips, acs_end_year, variable_id, COUNT(*) AS n
        FROM {ACS_PLACE}
        GROUP BY place_fips, acs_end_year, variable_id
        HAVING COUNT(*) > 1
        LIMIT :limit;
    """
    return run_query(query, {"limit": int(limit)})


def sanity_check_negative_moe(scope: str = "place") -> pd.DataFrame:
    """
    Detect negative margins of error (data corruption indicator).
    """
    table = ACS_PLACE if scope == "place" else ACS_STATE
    query = f"""
        SELECT COUNT(*) AS negative_moe_rows
        FROM {table}
        WHERE moe < 0;
    """
    return run_query(query)

# ============================================================
# Journalist Layer: Gateway Metrics + Story Angles
# ============================================================

GATEWAY_METRICS = "public.gateway_metrics"
STATE_METRICS = "public.state_metrics_yearly"
METRIC_CATALOG = "public.metric_catalog"


def get_metric_catalog() -> pd.DataFrame:
    return run_query(f"""
        SELECT metric_key, metric_label, theme, unit, description, format_hint
        FROM {METRIC_CATALOG}
        ORDER BY theme, metric_label;
    """)


def get_latest_year_available() -> int:
    df = run_query(f"SELECT MAX(year) AS y FROM {GATEWAY_METRICS};")
    if df.empty or pd.isna(df["y"].iloc[0]):
        return 0
    return int(df["y"].iloc[0])

def get_available_gateway_years() -> List[int]:
    df = run_query(f"""
        SELECT DISTINCT year
        FROM {GATEWAY_METRICS}
        WHERE year IS NOT NULL
        ORDER BY year;
    """)
    if df is None or df.empty:
        return []
    return [int(y) for y in df["year"].tolist()]

def get_gateway_metric_trend(place_fips: str, metric_key: str) -> pd.DataFrame:
    return run_query(f"""
        SELECT year,
               metric_value AS value,
               delta_5yr,
               delta_10yr,
               rank_within_gateway,
               rank_change_5yr
        FROM {GATEWAY_METRICS}
        WHERE place_fips::text = :place_fips
          AND metric_key = :metric_key
        ORDER BY year;
    """, {"place_fips": str(place_fips), "metric_key": metric_key})


def get_state_metric_trend(metric_key: str) -> pd.DataFrame:
    return run_query(f"""
        SELECT year,
               metric_value AS value
        FROM {STATE_METRICS}
        WHERE metric_key = :metric_key
        ORDER BY year;
    """, {"metric_key": metric_key})


def get_gateway_metric_snapshot(place_fips: str, metric_key: str, year: Optional[int] = None) -> pd.DataFrame:
    if year is None:
        # keep old behavior (latest) for callers that don't pass year
        return run_query(f"""
            SELECT year,
                   metric_value AS value,
                   delta_5yr,
                   delta_10yr,
                   rank_within_gateway,
                   rank_change_5yr
            FROM {GATEWAY_METRICS}
            WHERE place_fips::text = :place_fips
              AND metric_key = :metric_key
            ORDER BY year DESC
            LIMIT 1;
        """, {"place_fips": str(place_fips), "metric_key": metric_key})

    return run_query(f"""
        SELECT year,
               metric_value AS value,
               delta_5yr,
               delta_10yr,
               rank_within_gateway,
               rank_change_5yr
        FROM {GATEWAY_METRICS}
        WHERE place_fips::text = :place_fips
          AND metric_key = :metric_key
          AND year = :year
        LIMIT 1;
    """, {"place_fips": str(place_fips), "metric_key": metric_key, "year": int(year)})


def get_gateway_ranking(metric_key: str, year: int) -> pd.DataFrame:
    return run_query(f"""
        SELECT place_fips::text,
               place_name,
               metric_value AS value,
               rank_within_gateway
        FROM {GATEWAY_METRICS}
        WHERE metric_key = :metric_key
          AND year = :year
        ORDER BY value DESC NULLS LAST;
    """, {"metric_key": metric_key, "year": int(year)})


def get_gateway_scatter(metric_x: str, metric_y: str, year: int) -> pd.DataFrame:
    return run_query(f"""
        SELECT
          x.place_fips::text,
          x.place_name,
          x.metric_value AS x,
          y.metric_value AS y
        FROM {GATEWAY_METRICS} x
        JOIN {GATEWAY_METRICS} y
          ON x.place_fips = y.place_fips
         AND x.year = y.year
        WHERE x.metric_key = :metric_x
          AND y.metric_key = :metric_y
          AND x.year = :year
        ORDER BY x.metric_value NULLS LAST;
    """, {
        "metric_x": metric_x,
        "metric_y": metric_y,
        "year": int(year),
    })