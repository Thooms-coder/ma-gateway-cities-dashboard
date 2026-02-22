import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import AgGrid #, GridOptionsBuilder, GridUpdateMode
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd
import re
import textwrap
import pycountry
import requests
from openai import OpenAI

# --- Queries Import ---
from src.queries import (
    get_all_cities,
    get_gateway_fips,
    get_foreign_born_percent,
    get_foreign_born_by_country,
    get_income_trend,
    get_poverty_trend
)

# Mocking the missing queries for UI architecture purposes
def get_housing_burden(fips):
    return pd.DataFrame({"year": [2010, 2015, 2020, 2024], "rent_burden_percent": [30, 32, 35, 38]})

# Design System Colors (Colorblind Safe: Cobalt & Vermilion)
COLOR_TARGET = "#005ab5"
COLOR_BASE = "#dc3220"
COLOR_BG_LIGHT = "#f4f5f6"
COLOR_TEXT = "#2c2f33"
COLOR_BOSTON = "#009E73"  # Colorblind-safe bluish green

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    st.markdown(f"""
    <style>
    /* Marker is invisible */
    .section-card-marker {{
        display: none;
    }}

    /*
      IMPORTANT:
      Streamlit doesn't wrap components inside your <div>.
      So we style the *Streamlit container block* that contains a marker.
      This removes the "mystery rectangles" and makes true cards.
    */
    div[data-testid="stVerticalBlock"]:has(.section-card-marker) {{
        background: #ffffff;
        padding: 35px;
        border-radius: 2px;
        border: 1px solid #e1e4e8;
        margin-bottom: 30px;
    }}

    .map-legend {{
        display: flex;
        gap: 25px;
        justify-content: flex-start;
        padding: 15px 0;
        font-size: 0.85rem;
        font-family: "Public Sans", sans-serif;
        color: #586069;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; }}
    .dot {{ height: 12px; width: 12px; border-radius: 2px; display: inline-block; }}
    </style>
    """, unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# Load & Process Geo Data
# --------------------------------------------------
@st.cache_data
def load_ma_map():
    with open("data/ma_municipalities.geojson") as f:
        return json.load(f)

ma_geo = load_ma_map()

def normalize(name):
    return str(name).strip().upper()

def clean_place_label(name: str) -> str:
    s = str(name).replace(", Massachusetts", "").strip()
    s = re.sub(r"\b(city|town)\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s

def country_to_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None
    
def get_geo_bounds(geojson):
    lats, lons = [], []
    def extract_coords(coords):
        if isinstance(coords[0], list):
            for c in coords:
                extract_coords(c)
        else:
            lons.append(coords[0])
            lats.append(coords[1])
    for feature in geojson["features"]:
        extract_coords(feature["geometry"]["coordinates"])
    return min(lats), max(lats), min(lons), max(lons)

min_lat, max_lat, min_lon, max_lon = get_geo_bounds(ma_geo)
center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

# --------------------------------------------------
# City Data Initialization
# --------------------------------------------------
cities = get_all_cities()
gateway_fips = set(get_gateway_fips()["place_fips"])

# --------------------------------------------------
# Allowed Cities (Gateway + Boston + Cambridge)
# --------------------------------------------------

# Add Boston and Cambridge explicitly
extra_cities = cities[
    cities["place_name"].isin([
        "Boston city, Massachusetts",
        "Cambridge city, Massachusetts"
    ])
]

allowed_cities_df = pd.concat([
    cities[cities["place_fips"].isin(gateway_fips)],
    extra_cities
]).drop_duplicates()

allowed_fips = set(allowed_cities_df["place_fips"])

boston_cambridge_names = set(
    normalize(clean_place_label(name))
    for name in extra_cities["place_name"]
)

# Ensure consistent type for lookup + query params
if "place_fips" in cities.columns:
    cities["place_fips"] = cities["place_fips"].astype(str)

# Build normalized town → FIPS lookup (ALL municipalities)
town_fips_map = {
    normalize(clean_place_label(name)): fips
    for name, fips in zip(cities["place_name"], cities["place_fips"])
}

# Build allowed town name set (for map highlighting)
allowed_town_names = set(
    normalize(clean_place_label(name))
    for name in allowed_cities_df["place_name"]
)

locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]

# Dropdown shows ONLY gateway cities
city_options = (
    cities[cities["place_fips"].isin(gateway_fips)]["place_name"]
    .sort_values()
    .tolist()
)

# Keep this for your lede text; ensure it stays in sync with actual primary selection
if "selected_city" not in st.session_state:
    st.session_state.selected_city = city_options[0]

# --------------------------------------------------
# Header & Control View
# --------------------------------------------------
st.markdown("""
<section class="hero">
    <h1>Gateway Cities Investigative Dashboard</h1>
    <div class="accent-line"></div>
    <p>
        A longitudinal analysis of immigration patterns, demographic transitions,
        and structural economic shifts across Massachusetts municipalities.
        Source: American Community Survey (ACS) 2010–2024.
    </p>
</section>
""", unsafe_allow_html=True)

col_search, col_export = st.columns([3, 1])
with col_search:
    st.markdown("""
**Explore Gateway Cities**  

Select a Gateway City to observe Foreign-Born Populations trends. 

Selecting more than one Gateway City allows you to compare their demographic and economic trends side by side below. 

    """)
    
    available_options = sorted(
        set(city_options + st.session_state.get("selected_cities", []))
    )

    selected_cities = st.multiselect(
        "Compare Municipalities (Max 3)",
        options=available_options,
        default=st.session_state.get("selected_cities", [city_options[0]]),
        max_selections=3,
        key="selected_cities"
    )

    if not selected_cities:
        selected_cities = [city_options[0]]

selected_fips = {
    city: str(cities[cities["place_name"] == city]["place_fips"].values[0])
    for city in selected_cities
}

primary_city = selected_cities[0]
primary_fips = selected_fips[primary_city]

# --- LOGIC FIX: keep lede city in sync with current primary selection
st.session_state.selected_city = primary_city

city_data = {}

for city, fips in selected_fips.items():

    df_fb = get_foreign_born_percent(fips)
    df_income = get_income_trend(fips)
    df_poverty = get_poverty_trend(fips)

    for df in [df_fb, df_income, df_poverty]:
        if not df.empty:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df.dropna(subset=["year"], inplace=True)

    # --- LOGIC FIX: strict overlap dataset for structural panels (NO outer+interpolation fabrication)
    # This is the dataset used for trajectory/export (needs all 3 series).
    df_struct = (
        df_fb
        .merge(df_income, on="year", how="inner")
        .merge(df_poverty, on="year", how="inner")
        .sort_values("year")
        .reset_index(drop=True)
    )

    city_data[city] = {
        "fb": df_fb,
        "income": df_income,
        "poverty": df_poverty,
        "struct": df_struct
    }

with col_export:
    st.markdown("<br>", unsafe_allow_html=True)

    if primary_city in city_data:
        export_df = city_data[primary_city]["struct"]

        if not export_df.empty:
            csv = export_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label=f"Download {primary_city} Dataset (CSV)",
                data=csv,
                file_name=f"{primary_city.replace(' ', '_')}_longitudinal_data.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================================================
# SECTION 1: GEOGRAPHIC CONTEXT
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Geographic Context")

    @st.cache_data
    def build_map(geojson, locations, allowed_names, town_fips_map_local,
                place_fips, c_lat, c_lon, boston_cambridge_names):

        z_values = []
        selected_index = None

        for i, town_name in enumerate(locations):
            town_norm = normalize(town_name)

            # Selected municipality
            if (
                town_norm in town_fips_map_local and
                town_fips_map_local[town_norm] == place_fips
            ):
                z_values.append(3)
                selected_index = i

            # Boston & Cambridge
            elif town_norm in boston_cambridge_names:
                z_values.append(2)

            # Gateway cities
            elif town_norm in allowed_names:
                z_values.append(1)

            # All other municipalities
            else:
                z_values.append(0)

        trace = go.Choroplethmapbox(
            geojson=geojson,
            locations=locations,
            z=z_values,
            featureidkey="properties.TOWN",
            colorscale=[
                [0.0, "#E9ECEF"],
                [0.25, "#E9ECEF"],
                [0.25, COLOR_BASE],
                [0.5, COLOR_BASE],
                [0.5, COLOR_BOSTON],
                [0.75, COLOR_BOSTON],
                [0.75, COLOR_TARGET],
                [1.0, COLOR_TARGET]
            ],
            zmin=0,
            zmax=3,
            showscale=False,
            marker_line_width=1.0,
            marker_line_color="rgba(60,65,75,0.5)",
            hovertemplate="<b>%{location}</b><extra></extra>",
            selectedpoints=[selected_index] if selected_index is not None else None,
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.8))
        )

        fig = go.Figure(trace)

        fig.update_layout(
            clickmode="event+select",
            mapbox=dict(
                style="white-bg",
                center=dict(lat=c_lat, lon=c_lon),
                zoom=8
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=825
        )

        return fig

    fig_map = build_map(
        ma_geo,
        locations,
        allowed_town_names,
        town_fips_map,
        primary_fips,
        center_lat,
        center_lon,
        boston_cambridge_names
    )
    
    map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_select")

    st.markdown(f"""
        <div class="map-legend">
            <div class="legend-item">
                <span class="dot" style="background-color: {COLOR_TARGET};"></span>
                Selected Municipality
            </div>
            <div class="legend-item">
                <span class="dot" style="background-color: {COLOR_BOSTON};"></span>
                Boston & Cambridge
            </div>
            <div class="legend-item">
                <span class="dot" style="background-color: {COLOR_BASE};"></span>
                Gateway Cities
            </div>
            <div class="legend-item">
                <span class="dot" style="background-color: #E9ECEF; border: 1px solid #ccc;"></span>
                Other Municipalities
            </div>
        </div>
    """, unsafe_allow_html=True)

    if map_event and "selection" in map_event and map_event["selection"]["points"]:
        clicked_town = map_event["selection"]["points"][0]["location"]
        town_norm = normalize(clicked_town)

        if (
            town_norm in town_fips_map and
            town_fips_map[town_norm] in allowed_fips
        ):
            new_fips = town_fips_map[town_norm]
            new_city = cities[cities["place_fips"] == new_fips]["place_name"].iloc[0]

            if new_city not in st.session_state["selected_cities"]:
                if len(st.session_state["selected_cities"]) < 3:
                    st.session_state["selected_cities"].append(new_city)
                else:
                    st.session_state["selected_cities"] = [
                        st.session_state["selected_cities"][0],
                        new_city
                    ]
                st.rerun()

    st.markdown('<hr style="border: 0; border-top: 1px solid #e1e4e8; margin: 30px 0;">', unsafe_allow_html=True)

    # --- LOGIC FIX: use primary city fb series (not leaked loop variable)
    df_primary_fb = city_data.get(primary_city, {}).get("fb", pd.DataFrame())

    if not df_primary_fb.empty:
        latest_percent = df_primary_fb["foreign_born_percent"].iloc[-1]
        start_val = df_primary_fb["foreign_born_percent"].iloc[0]
        growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else 0

        col_kpi, col_lede = st.columns([1, 2.5])
        with col_kpi:
            st.markdown(
                f'<div class="kpi-container"><div class="kpi-label">Foreign-Born Base</div><div class="kpi-value">{latest_percent:.1f}%</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="kpi-container"><div class="kpi-label">Period Growth Rate</div><div class="kpi-value">{growth:.1f}%</div></div>',
                unsafe_allow_html=True
            )
        with col_lede:
            trend_word = "surged" if growth > 10 else "grown" if growth > 0 else "declined"
            st.markdown(textwrap.dedent(f"""
            <div style="font-family: 'Lora', serif; font-size:1.15rem; line-height: 1.7; color: #333; padding: 10px 20px;">
            Over the observed period, the foreign-born population in <b>{st.session_state.selected_city}</b> has {trend_word} by {abs(growth):.1f}%, now representing {latest_percent:.1f}% of the total community. This demographic shift provides the foundation for examining localized economic transitions, housing pressures, and wealth distribution.
            </div>
            """), unsafe_allow_html=True)

# ==================================================
# SECTION 3: ECONOMIC INDICATORS
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Economic Health & Poverty Status")

    col_ts1, col_ts2 = st.columns(2)

    # -------------------------
    # Median Income Comparison
    # -------------------------
    with col_ts1:
        fig_inc = go.Figure()

        for city, data in city_data.items():
            # Use income series directly (no dependence on overlap)
            df = data.get("income", pd.DataFrame())
            if not df.empty and "median_income" in df.columns:
                fig_inc.add_trace(go.Scatter(
                    x=df["year"],
                    y=df["median_income"],
                    mode="lines",
                    name=city,
                    line=dict(width=3),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Year: %{x}<br>"
                        "Income: $%{y:,.0f}<extra></extra>"
                    )
                ))

        fig_inc.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Year",
            yaxis_title="Median Household Income ($)",
            font=dict(family="Public Sans"),
            height=500,
            legend=dict(title=""),
            hoverlabel=dict(
                bgcolor="#ffffff",
                font=dict(color="#2c2f33", family="Public Sans", size=14),
                bordercolor="#e1e4e8",
                align="left"
            )
        )

        st.plotly_chart(fig_inc, use_container_width=True)

    # -------------------------
    # Poverty Rate Comparison
    # -------------------------
    with col_ts2:
        fig_pov = go.Figure()

        for city, data in city_data.items():
            # Use poverty series directly (no dependence on overlap)
            df = data.get("poverty", pd.DataFrame())
            if not df.empty and "poverty_rate" in df.columns:
                fig_pov.add_trace(go.Scatter(
                    x=df["year"],
                    y=df["poverty_rate"],
                    mode="lines",
                    name=city,
                    line=dict(width=3),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Year: %{x}<br>"
                        "Poverty Rate: %{y:.1f}%<extra></extra>"
                    )
                ))

        fig_pov.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Year",
            yaxis_title="Poverty Rate (%)",
            font=dict(family="Public Sans"),
            height=500,
            legend=dict(title=""),
            hoverlabel=dict(
                bgcolor="#ffffff",
                font=dict(color="#2c2f33", family="Public Sans", size=14),
                bordercolor="#e1e4e8",
                align="left"
            )
        )

        st.plotly_chart(fig_pov, use_container_width=True)

# ==================================================
# SECTION 4: TRAJECTORY ANALYSIS
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Structural Trajectory: Income vs. Immigration")
    st.markdown(
        "<p style='color: #586069; font-size: 0.95rem;'>"
        "This connected scatterplot traces each municipality’s economic and demographic movement year-over-year. "
        "Upward and rightward movement indicates simultaneous growth in median income and foreign-born population."
        "</p>",
        unsafe_allow_html=True
    )

    fig_traj = go.Figure()

    for city, data in city_data.items():
        # This plot requires overlap between FB% and Income.
        # You originally used df_struct; we keep that intent but now it's strict overlap of all 3.
        df = data.get("struct", pd.DataFrame())

        if not df.empty and len(df) > 1 and "foreign_born_percent" in df.columns and "median_income" in df.columns:
            fig_traj.add_trace(go.Scatter(
                x=df["foreign_born_percent"],
                y=df["median_income"],
                mode='markers',
                name=city,
                marker=dict(size=8),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Foreign-Born: %{x:.1f}%<br>"
                    "Income: $%{y:,.0f}<extra></extra>"
                ),
                text=df["year"]
            ))

    fig_traj.update_layout(
        template="plotly_white",
        xaxis_title="Foreign-Born Population (%)",
        yaxis_title="Median Household Income ($)",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Public Sans"),
        legend=dict(title="Municipality"),
        height=550,
        hoverlabel=dict(
            bgcolor="#ffffff",
            font=dict(color="#2c2f33", family="Public Sans", size=14),
            bordercolor="#e1e4e8",
            align="left"
        )
    )

    st.plotly_chart(fig_traj, use_container_width=True)

# ==================================================
# SECTION 5: Tables
# ==================================================

with st.container():
    st.markdown('### Our Data')
    st.markdown(f"#### Foreign-Born Population for {primary_city.split(',')[0]}")
    df_fb = get_foreign_born_percent(fips)
    df_fb.columns = df_fb.columns.str.replace("_", " ").str.title()
    AgGrid(df_fb, fit_columns_on_grid_load=True)

    st.markdown(f"#### Poverty Rate for {primary_city.split(',')[0]}")
    df_pov = data.get("poverty", pd.DataFrame())
    df_pov.columns = df_pov.columns.str.replace("_", " ").str.title()
    AgGrid(df_pov, fit_columns_on_grid_load=True)

    st.markdown(f"#### Income and Immigration for {primary_city.split(',')[0]}")
    df_inc = data.get("struct", pd.DataFrame())
    df_inc.columns = df_inc.columns.str.replace("_", " ").str.title()
    AgGrid(df_inc, fit_columns_on_grid_load=True)

# ==================================================
# SECTION 6: METHODOLOGY
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("#### Data Responsibility & Methodology")
    st.markdown(textwrap.dedent("""<div style="font-size: 0.9rem; line-height: 1.6; color: #586069;">
    <strong>1. Transparency & Accuracy:</strong> All figures are derived directly from the U.S. Census American Community Survey (ACS) 5-Year Estimates. Margins of error (MOE) are preserved in the backend.<br><br>"""), unsafe_allow_html=True)
    st.link_button("Open gateway city report", url="https://www.census.gov/programs-surveys/acs/data.html")
    st.markdown(textwrap.dedent("""<div style="font-size: 0.9rem; line-height: 1.6; color: #586069;"> <strong>2. Journalistic Framing:</strong> This platform avoids causal claims without rigorous statistical testing. Correlation visualized across demographic and economic panels is intended to surface trends for localized reporting, rather than draw definitive conclusions.<br><br>
    <strong>3. Limitations:</strong> ACS 5-year estimatses smooth out short-term volatility. Data represented here should be cross-referenced with local municipal records where applicable.
    </div>
    """), unsafe_allow_html=True)
    
# ==================================================
# SECTION 7: CUSTOM QUERY
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Want to find something else?")

    CENSUS_VARIABLES = """
## POPULATION
B01003_001E: Total population

## RACE & ETHNICITY
B02001_002E: White alone
B02001_003E: Black or African American alone
B02001_004E: American Indian and Alaska Native alone
B02001_005E: Asian alone
B02001_006E: Native Hawaiian and Other Pacific Islander alone
B02001_007E: Some other race alone
B03002_003E: White alone, not Hispanic or Latino
B03002_012E: Hispanic or Latino (any race)
B03001_003E: Hispanic or Latino total

## NATIVITY & FOREIGN-BORN
B05002_001E: Total population (nativity)
B05002_002E: Native-born population
B05002_013E: Foreign-born population total
B05001_001E: Total population (citizenship)
B05001_005E: U.S. citizen by naturalization
B05001_006E: Not a U.S. citizen
B05002_014E: Foreign-born: naturalized citizen
B05002_021E: Foreign-born: not a citizen

## PLACE OF BIRTH (FOREIGN-BORN)
B05006_001E: Total foreign-born population (place of birth)
B05006_002E: Foreign-born from Europe
B05006_047E: Foreign-born from Asia
B05006_091E: Foreign-born from Africa
B05006_100E: Foreign-born from Oceania
B05006_101E: Foreign-born from Latin America
B05006_123E: Foreign-born from Northern America

## YEAR OF ENTRY
B05005_001E: Total foreign-born (year of entry)
B05005_002E: Entered 2010 or later
B05005_006E: Entered 2000 to 2009
B05005_009E: Entered before 2000

## GEOGRAPHIC MOBILITY (MIGRATION)
B07001_001E: Total population 1 year and over (mobility)
B07001_017E: Lived in same house 1 year ago (did not move)
B07001_033E: Moved within same county
B07001_049E: Moved from different county, same state
B07001_065E: Moved from different state
B07001_081E: Moved from abroad
B07003_004E: Male movers from different state
B07003_007E: Female movers from different state
B07013_001E: Total population in occupied housing units (mobility)
B07013_003E: Moved in same county — renters

## INCOME
B19013_001E: Median household income (all households)
B19013B_001E: Median household income — Black or African American households
B19013D_001E: Median household income — Asian households
B19013H_001E: Median household income — White non-Hispanic households
B19013I_001E: Median household income — Hispanic or Latino households
B19301_001E: Per capita income
B19083_001E: Gini index of income inequality
B19001_001E: Total households (household income distribution)
B19001_002E: Households with income less than $10,000
B19001_011E: Households with income $50,000 to $59,999
B19001_014E: Households with income $100,000 to $124,999
B19001_017E: Households with income $200,000 or more

## POVERTY
B17001_001E: Total population (poverty status)
B17001_002E: Population below poverty level
B17001_031E: Population at or above poverty level
C17002_001E: Total (ratio of income to poverty level)
C17002_002E: Under 0.50 (deep poverty)
C17002_003E: 0.50 to 0.99 (below poverty)
C17002_004E: 1.00 to 1.24 (near poverty)
C17002_008E: 2.00 and over (200%+ of poverty line)

## HOUSING & RENT BURDEN
B25070_001E: Total renter-occupied units (gross rent as % of income)
B25070_007E: Gross rent 30.0 to 34.9% of income (rent burdened)
B25070_008E: Gross rent 35.0 to 39.9% of income
B25070_009E: Gross rent 40.0 to 49.9% of income
B25070_010E: Gross rent 50% or more of income (severely rent burdened)
B25064_001E: Median gross rent (dollars)
B25003_001E: Total occupied housing units (tenure)
B25003_002E: Owner-occupied housing units
B25003_003E: Renter-occupied housing units

## EMPLOYMENT
B23025_001E: Total civilian population 16 years and over
B23025_002E: In labor force
B23025_004E: Employed (civilian labor force)
B23025_005E: Unemployed
B23025_007E: Not in labor force

## EDUCATION
B15003_001E: Total population 25 years and over (educational attainment)
B15003_017E: Population with high school diploma (or equivalent)
B15003_022E: Population with bachelor's degree
B15003_023E: Population with master's degree
B15003_025E: Population with doctorate degree
"""

    CENSUS_QUERY_SYSTEM_PROMPT = f"""
You are a Census data assistant helping journalists explore Massachusetts data.
Given a plain English question, return ONLY a valid JSON object (no markdown, no explanation) with:

- "variables": list of ACS variable codes to fetch (from the list below)
- "year": integer year (use 2022 unless the user specifies)
- "geo": Census API geo string, one of:
    "county:*&in=state:25"  (all MA counties)
    "place:*&in=state:25"   (all MA cities/towns)
- "chart_type": one of "bar", "line", "scatter", "pie"
- "x_col": column name for x-axis (usually "NAME")
- "y_col": the primary variable code to plot
- "title": a descriptive chart title
- "x_label": x-axis label
- "y_label": y-axis label

Available variables:
{CENSUS_VARIABLES}

Example output:
{{
    "variables": ["B19013_001E"],
    "year": 2022,
    "geo": "county:*&in=state:25",
    "chart_type": "bar",
    "x_col": "NAME",
    "y_col": "B19013_001E",
    "title": "Median Household Income by County in MA (2022)",
    "x_label": "County",
    "y_label": "Median Household Income ($)"
}}

Only use variable codes from the list above. If the question is unrelated to Census data, return:
{{"error": "I can only answer questions about Census data for Massachusetts."}}
"""

    def fetch_census_data(variables, geo, year):
        base_url = f"https://api.census.gov/data/{year}/acs/acs5"
        get_cols = ",".join(variables) + ",NAME"
        params = {"get": get_cols}
        for part in geo.split("&"):
            if part.startswith("in="):
                params["in"] = part[3:]
            else:
                params["for"] = part.replace("for=", "")
        census_api_key = st.secrets.get("CENSUS_API_KEY", None)
        if census_api_key:
            params["key"] = census_api_key
        r = requests.get(base_url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        for v in variables:
            if v in df.columns:
                df[v] = pd.to_numeric(df[v], errors="coerce")
        if "NAME" in df.columns:
            df["NAME"] = df["NAME"].str.replace(", Massachusetts", "", regex=False)
        return df

    def ask_deepseek(question):
        client = OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": CENSUS_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    question = st.text_input(
        "",
        placeholder="e.g. Which counties have the highest median income? Show me migration trends by county.",
    )
    submit = st.button("Search")

    if submit and question:
        with st.spinner("Thinking..."):
            try:
                query = ask_deepseek(question)
            except Exception as e:
                st.error(f"AI error: {e}")
                query = None

        if query:
            if "error" in query:
                st.warning(query["error"])
            else:
                with st.spinner("Fetching Census data..."):
                    try:
                        df = fetch_census_data(query["variables"], query["geo"], query["year"])
                    except Exception as e:
                        st.error(f"Census API error: {e}")
                        df = None

                if df is not None and not df.empty:
                    x = query.get("x_col", "NAME")
                    y = query.get("y_col", query["variables"][0])
                    title = query.get("title", "Census Data")
                    x_label = query.get("x_label", x)
                    y_label = query.get("y_label", y)
                    chart_type = query.get("chart_type", "bar")

                    df_sorted = df.dropna(subset=[y]).sort_values(y, ascending=True)

                    if chart_type == "bar":
                        fig = px.bar(df_sorted, x=y, y=x, orientation="h",
                                     title=title, labels={y: y_label, x: x_label})
                    elif chart_type == "scatter":
                        fig = px.scatter(df_sorted, x=x, y=y,
                                         title=title, labels={y: y_label, x: x_label},
                                         hover_name=x if x == "NAME" else None)
                    elif chart_type == "pie":
                        fig = px.pie(df_sorted, values=y, names=x, title=title)
                    else:
                        fig = px.line(df_sorted, x=x, y=y,
                                      title=title, labels={y: y_label, x: x_label})

                    st.plotly_chart(fig, use_container_width=True)
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False),
                        file_name="census_data.csv",
                        mime="text/csv",
                    )
                elif df is not None:
                    st.warning("No data returned for that query.")