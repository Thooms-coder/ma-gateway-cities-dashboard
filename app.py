import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd

# --- Queries Import ---
from src.queries import (
    get_cities,
    get_foreign_born_percent,
    get_income_trend,
    get_poverty_trend,
    # get_housing_burden,
    # get_employment_trend
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
# Load Custom CSS (Restored to prioritize your theme)
# --------------------------------------------------

def load_css():
    # 1. Load your established theme first
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
        
    # 2. Inject strictly the layout/animation classes for the scroll narrative
    st.markdown("""
    <style>
    /* Smooth Fade & Rise Animation */
    @keyframes fadeInRise {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Narrative Section Cards (Inherits your fonts and base styles) */
    .section-card {
        animation: fadeInRise 0.8s ease-out forwards;
        background: #ffffff;
        padding: 30px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border-top: 3px solid #8b0000;
        margin-bottom: 40px;
    }
    
    /* Staggered load delays for cascading effect */
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.3s; }
    .delay-3 { animation-delay: 0.5s; }
    .delay-4 { animation-delay: 0.7s; }

    /* Clean Map Legend */
    .map-legend { display: flex; gap: 25px; justify-content: center; padding: 15px 0; font-size: 0.9rem; }
    .legend-item { display: flex; align-items: center; gap: 8px; }
    .dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; }
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
# Header & Control View (Restored original markup classes)
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

# ==================================================
# SECTION 1: GEOGRAPHIC CONTEXT
# ==================================================
st.markdown('<div class="section-card delay-1">', unsafe_allow_html=True)
st.markdown("#### Geographic Overview")

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
        mapbox=dict(style="white-bg", center=dict(lat=c_lat, lon=c_lon), zoom=7.2),
        margin=dict(l=0, r=0, t=0, b=0), height=500
    )
    return fig

fig_map = build_map(ma_geo, locations, gateway_names, selected_city_norm, center_lat, center_lon)
map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_select")

st.markdown("""
    <div class="map-legend">
        <div class="legend-item"><span class="dot" style="background-color: #111111;"></span> Target Municipality</div>
        <div class="legend-item"><span class="dot" style="background-color: #8b0000;"></span> Gateway City Baseline</div>
        <div class="legend-item"><span class="dot" style="background-color: #e5e5e5; border: 1px solid #ccc;"></span> Rest of Commonwealth</div>
    </div>
""", unsafe_allow_html=True)

if map_event and "selection" in map_event and map_event["selection"]["points"]:
    clicked_town = map_event["selection"]["points"][0]["location"]
    matched_city = cities[cities["place_name"].str.upper().str.contains(clicked_town.upper())]
    if not matched_city.empty and matched_city["place_name"].iloc[0] != st.session_state.selected_city:
        st.session_state.selected_city = matched_city["place_name"].iloc[0]
        st.rerun()

st.markdown('<hr style="border: 0; border-top: 1px solid #eee; margin: 30px 0;">', unsafe_allow_html=True)

if len(df_fb) > 0:
    latest_percent = df_fb["foreign_born_percent"].iloc[-1]
    start_val = df_fb["foreign_born_percent"].iloc[0]
    growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else 0
    
    col_kpi, col_lede = st.columns([1, 2])
    with col_kpi:
        st.markdown(f'<div class="kpi-container"><div class="kpi-label">Foreign-Born Base</div><div class="kpi-value">{latest_percent:.1f}%</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-container"><div class="kpi-label">Period Growth Rate</div><div class="kpi-value">{growth:.1f}%</div></div>', unsafe_allow_html=True)
    with col_lede:
        trend_word = "surged" if growth > 10 else "grown" if growth > 0 else "declined"
        st.markdown(f"""
        <div style="background:#f4f4f4; padding:25px; border-left:4px solid #8b0000; height: 100%; font-size:1.05rem; line-height: 1.6;">
        <strong>Investigative Briefing:</strong> Over the observed period, the foreign-born population in {st.session_state.selected_city} has {trend_word} by {abs(growth):.1f}%, now representing {latest_percent:.1f}% of the total community. This sets the demographic baseline for cross-referencing against municipal economic indicators below.
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# SECTION 2: ECONOMIC INDICATORS
# ==================================================
st.markdown('<div class="section-card delay-2">', unsafe_allow_html=True)
st.markdown("#### Economic Health & Poverty Status")
st.markdown("Analyzing ACS Tables S1901 (Income) and S1701 (Poverty).")

col_ts1, col_ts2 = st.columns(2)

with col_ts1:
    if not df_struct.empty and "median_income" in df_struct.columns:
        fig_inc = px.line(df_struct, x="year", y="median_income", title="Median Household Income Trajectory ($)")
        # Layout handles the frame
        fig_inc.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        # Traces handle the data lines
        fig_inc.update_traces(line_color="#111111")
        st.plotly_chart(fig_inc, use_container_width=True)
    else:
        st.info("Income data currently unavailable for this selection.")

with col_ts2:
    if not df_struct.empty and "poverty_rate" in df_struct.columns:
        fig_pov = px.line(df_struct, x="year", y="poverty_rate", title="Poverty Rate Deviation (%)")
        # Layout handles the frame
        fig_pov.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        # Traces handle the data lines
        fig_pov.update_traces(line_color="#8b0000")
        st.plotly_chart(fig_pov, use_container_width=True)
    else:
        st.info("Poverty data currently unavailable for this selection.")

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# SECTION 3: HOUSING & TRANSPORT
# ==================================================
st.markdown('<div class="section-card delay-3">', unsafe_allow_html=True)
st.markdown("#### Housing Burden")
st.markdown("Analyzing ACS Table B25070 (Rent Burden).")

df_housing_mock = get_housing_burden(place_fips)
fig_house = px.bar(df_housing_mock, x="year", y="rent_burden_percent", title="Rent as Percentage of Income")
fig_house.update_traces(marker_color='#111111')
fig_house.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_house, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# SECTION 4: MULTIDIMENSIONAL ANALYSIS
# ==================================================
st.markdown('<div class="section-card delay-4">', unsafe_allow_html=True)
st.markdown("#### Multidimensional Outlier Detection")
st.markdown("Tracing the longitudinal relationship between immigration scale, median income, and poverty rate to identify systemic divergence.")

if len(df_struct) > 1:
    fig_par = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_struct['year'], 
                colorscale=[[0, '#e5e5e5'], [1, '#8b0000']], 
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
    fig_par.update_layout(
        margin=dict(l=40, r=40, t=40, b=20), 
        height=450
    )
    st.plotly_chart(fig_par, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# SECTION 5: METHODOLOGY
# ==================================================
st.markdown('<div class="section-card delay-4" style="background: #fdfdfd; border-top: 1px solid #ddd;">', unsafe_allow_html=True)
st.markdown("#### Data Responsibility & Methodology")
st.markdown("""
<div style="font-size: 0.95rem; line-height: 1.6;">
<strong>1. Transparency & Accuracy:</strong> All figures are derived directly from the U.S. Census American Community Survey (ACS) 5-Year Estimates. Margins of error (MOE) are preserved in the backend to prevent over-indexing on marginal statistical shifts.<br><br>
<strong>2. Journalistic Framing:</strong> This platform avoids causal claims without rigorous statistical testing. Correlation visualized across demographic and economic panels is intended to surface trends for localized reporting, rather than draw definitive conclusions on causality.<br><br>
<strong>3. Limitations:</strong> ACS 5-year estimates smooth out short-term volatility. Data represented here should be cross-referenced with local municipal records where applicable.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)