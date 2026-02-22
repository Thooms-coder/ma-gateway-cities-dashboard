import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd

# --- Assuming these exist in your src.queries based on README ---
from src.queries import (
    get_cities,
    get_foreign_born_percent,
    get_income_trend,
    get_poverty_trend,
    # get_housing_burden,   <-- YOU WILL NEED TO CREATE THESE
    # get_employment_trend  <-- YOU WILL NEED TO CREATE THESE
)

# Mocking the new queries for UI architecture purposes
def get_housing_burden(fips): return pd.DataFrame({"year": [2010, 2015, 2020, 2024], "rent_burden_percent": [30, 32, 35, 38]})
def get_employment_trend(fips): return pd.DataFrame({"year": [2010, 2015, 2020, 2024], "unemployment_rate": [8.5, 6.0, 7.5, 4.2]})

# --------------------------------------------------
# Page Config 
# --------------------------------------------------

st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --------------------------------------------------
# Load Custom CSS
# --------------------------------------------------

def load_css():
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback inline CSS for the custom legend and kpis
        st.markdown("""
        <style>
        .kpi-container { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #8b0000; margin-bottom: 10px; }
        .kpi-label { font-size: 0.9rem; color: #555; text-transform: uppercase; }
        .kpi-value { font-size: 1.8rem; font-weight: bold; color: #111; }
        .map-legend { display: flex; gap: 20px; justify-content: center; padding: 10px; font-size: 0.9rem; }
        .legend-item { display: flex; align-items: center; gap: 5px; }
        .dot { height: 12px; width: 12px; border-radius: 50%; display: inline-block; }
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
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

# --------------------------------------------------
# City Data Initialization
# --------------------------------------------------

cities = get_cities(gateway_only=False)
gateway_cities = get_cities(gateway_only=True)

gateway_names = set(
    normalize(
        n.replace(", Massachusetts", "")
         .replace(" city", "")
         .replace(" City", "")
         .replace(" Town city", "")
         .replace(" Town", "")
         .strip()
    )
    for n in gateway_cities["place_name"]
)

locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]
city_options = cities["place_name"].tolist()

if "selected_city" not in st.session_state:
    st.session_state.selected_city = city_options[0]

# --------------------------------------------------
# Hero Section
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>Gateway Cities Investigative Dashboard</h1>
    <div class="accent-line"></div>
    <p>
    A longitudinal analysis of immigration patterns, demographic transitions, 
    and structural economic shifts across Massachusetts municipalities. (Source: ACS 2010-2024)
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Global Control 
# --------------------------------------------------

st.markdown("### Search Target Municipality")
selected_city = st.selectbox(
    "Target Municipality",
    options=city_options,
    label_visibility="collapsed",
    key="selected_city" 
)

selected_city_norm = normalize(st.session_state.selected_city)
place_fips = cities[cities["place_name"] == st.session_state.selected_city]["place_fips"].values[0]

# Pre-fetch core data
df_fb = get_foreign_born_percent(place_fips)
df_income = get_income_trend(place_fips)
df_poverty = get_poverty_trend(place_fips)

for df in [df_fb, df_income, df_poverty]:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df.dropna(subset=["year"], inplace=True)

df_struct = df_fb.merge(df_income, on="year", how="outer").merge(df_poverty, on="year", how="outer").sort_values("year").reset_index(drop=True)
df_struct = df_struct.interpolate(method="linear", limit_direction="both").dropna()

# --------------------------------------------------
# Architecture: Multi-Tab Layout
# --------------------------------------------------

tab_geo, tab_econ, tab_housing, tab_multi, tab_ethics = st.tabs([
    "📍 Geographic Overview", 
    "📈 Economic Indicators", 
    "🏠 Housing & Transport",
    "🔍 Multidimensional Analysis",
    "⚖️ Methodology & Ethics"
])

# ==================================================
# TAB 1: GEOGRAPHIC OVERVIEW
# ==================================================
with tab_geo:
    st.markdown("#### Municipality Selection & Demographics")
    
    # Map rendering function - Fixed cache mutation issue
    @st.cache_data
    def build_map(geojson, locations, gw_names, selected_norm, c_lat, c_lon):
        z_values = []
        for town_name in locations:
            town_norm = normalize(town_name)
            if town_norm == selected_norm:
                z_values.append(2) # Black
            elif town_norm in gw_names:
                z_values.append(1) # Dark Red
            else:
                z_values.append(0) # Light Gray

        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson, locations=locations, z=z_values,
            featureidkey="properties.TOWN",
            colorscale=[[0.0, "#e5e5e5"], [0.499, "#e5e5e5"], [0.5, "#8b0000"], [0.999, "#8b0000"], [1.0, "#111111"]],
            zmin=0, zmax=2, showscale=False, marker_line_width=0.5, marker_line_color="#ffffff",
            hovertemplate="<b>%{location}</b><extra></extra>"
        ))

        fig.update_layout(
            clickmode="event+select",
            mapbox=dict(style="white-bg", center=dict(lat=c_lat, lon=c_lon), zoom=7.5),
            margin=dict(l=0, r=0, t=0, b=0), height=550
        )
        return fig

    fig_map = build_map(ma_geo, locations, gateway_names, selected_city_norm, center_lat, center_lon)
    
    map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_select")
    
    # Custom HTML Legend
    st.markdown("""
        <div class="map-legend">
            <div class="legend-item"><span class="dot" style="background-color: #111111;"></span> Selected Municipality</div>
            <div class="legend-item"><span class="dot" style="background-color: #8b0000;"></span> Gateway City</div>
            <div class="legend-item"><span class="dot" style="background-color: #e5e5e5; border: 1px solid #ccc;"></span> Other MA Town</div>
        </div>
    """, unsafe_allow_html=True)

    if map_event and "selection" in map_event and map_event["selection"]["points"]:
        clicked_town = map_event["selection"]["points"][0]["location"]
        matched_city = cities[cities["place_name"].str.upper().str.contains(clicked_town.upper())]
        if not matched_city.empty and matched_city["place_name"].iloc[0] != st.session_state.selected_city:
            st.session_state.selected_city = matched_city["place_name"].iloc[0]
            st.rerun()

    st.markdown("---")
    st.markdown("#### Demographic Lede (B05006)")
    if len(df_fb) > 0:
        latest_percent = df_fb["foreign_born_percent"].iloc[-1]
        start_val = df_fb["foreign_born_percent"].iloc[0]
        growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else 0
        
        col_kpi, col_lede = st.columns([1, 2])
        with col_kpi:
            st.markdown(f'<div class="kpi-container"><div class="kpi-label">Foreign-Born % (Latest)</div><div class="kpi-value">{latest_percent:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-container"><div class="kpi-label">Growth Rate</div><div class="kpi-value">{growth:.1f}%</div></div>', unsafe_allow_html=True)
        with col_lede:
            trend_word = "surged" if growth > 10 else "grown" if growth > 0 else "declined"
            st.markdown(f"""
            <div style="background:#f4f4f4; padding:20px; border-left:4px solid #8b0000; height: 100%; font-size:1.05rem; line-height: 1.6;">
            <strong>Journalist Briefing:</strong> Over the observed period, the foreign-born population in {st.session_state.selected_city} has {trend_word} by {abs(growth):.1f}%, now representing {latest_percent:.1f}% of the total community.
            </div>
            """, unsafe_allow_html=True)

# ==================================================
# TAB 2: ECONOMIC INDICATORS
# ==================================================
with tab_econ:
    st.markdown(f"### Economic Health & Poverty Status: {st.session_state.selected_city}")
    st.markdown("Analyzing ACS Tables S1901 (Income), S1701 (Poverty), and B23025 (Employment).")
    
    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        fig_inc = px.line(df_struct, x="year", y="median_income", title="Median Household Income ($)")
        fig_inc.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_inc, use_container_width=True)

    with col_ts2:
        fig_pov = px.line(df_struct, x="year", y="poverty_rate", title="Poverty Rate (%)")
        fig_pov.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pov, use_container_width=True)
        
    st.info("💡 Next Step for Devs: Connect `get_employment_trend(fips)` to Supabase and render unemployment tracking here.")

# ==================================================
# TAB 3: HOUSING & TRANSPORT
# ==================================================
with tab_housing:
    st.markdown(f"### Housing Burden & Infrastructure: {st.session_state.selected_city}")
    st.markdown("Analyzing ACS Tables B25070 (Rent Burden) and B08126 (Travel Time).")
    
    # MOCKED DATA VISUALIZATION
    df_housing_mock = get_housing_burden(place_fips)
    
    fig_house = px.bar(df_housing_mock, x="year", y="rent_burden_percent", title="Rent as % of Income (Mocked Data)")
    fig_house.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_house, use_container_width=True)
    
    st.info("💡 Next Step for Devs: Connect `get_housing_burden(fips)` to Supabase. Correlate rapid foreign-born growth with spikes in rent burden to identify displacement risks.")

# ==================================================
# TAB 4: MULTIDIMENSIONAL ANALYSIS (OUTLIERS)
# ==================================================
with tab_multi:
    st.markdown(f"### Structural Correlation & Outlier Detection: {st.session_state.selected_city}")
    
    st.markdown("This visualization traces the relationship between time, immigration percentage, median income, and poverty rate to identify systemic divergence from statewide averages.")
    
    if len(df_struct) > 1:
        fig_par = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df_struct['year'], 
                    colorscale='Tealrose',
                    showscale=True,
                    cmin=df_struct['year'].min(),
                    cmax=df_struct['year'].max()
                ),
                dimensions=[
                    dict(range=[df_struct["year"].min(), df_struct["year"].max()], label="Year", values=df_struct["year"]),
                    dict(range=[df_struct["foreign_born_percent"].min(), df_struct["foreign_born_percent"].max()], label="Foreign-Born %", values=df_struct["foreign_born_percent"]),
                    dict(range=[df_struct["median_income"].min(), df_struct["median_income"].max()], label="Median Income", values=df_struct["median_income"], tickformat="$,.0f"),
                    dict(range=[df_struct["poverty_rate"].min(), df_struct["poverty_rate"].max()], label="Poverty Rate (%)", values=df_struct["poverty_rate"])
                ]
            )
        )
        fig_par.update_layout(margin=dict(l=40, r=40, t=40, b=20), height=400)
        st.plotly_chart(fig_par, use_container_width=True)

# ==================================================
# TAB 5: METHODOLOGY & ETHICS
# ==================================================
with tab_ethics:
    st.markdown("### Ethical & Responsible Data Use")
    st.markdown("""
    This project synthesizes **U.S. Census American Community Survey (ACS) 5-Year Estimates (2010–2024)**.
    
    **1. Transparency & Accuracy**
    * All tables documented directly from census.gov.
    * Margins of error (MOE) are preserved in the backend to prevent over-indexing on marginal shifts.
    
    **2. Fairness & Interpretation**
    * We avoid making causal claims without rigorous statistical testing.
    * This dashboard separates correlation from interpretation, surfacing trends for journalistic investigation rather than drawing definitive conclusions.
    
    **3. Limitations**
    * ACS 5-year estimates smooth out short-term volatility.
    * Margins of error can be large for small cities. Shifts must be evaluated alongside local reporting contexts.
    """)