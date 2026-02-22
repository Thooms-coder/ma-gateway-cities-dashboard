import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd
from src.queries import (
    get_cities,
    get_foreign_born_percent,
    get_income_trend,
    get_poverty_trend,
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="GBH | Gateway Cities",
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
        pass # Failsafe in case the CSS file is missing

load_css()

# --------------------------------------------------
# Load Massachusetts GeoJSON
# --------------------------------------------------

@st.cache_data
def load_ma_map():
    with open("data/ma_municipalities.geojson") as f:
        return json.load(f)

ma_geo = load_ma_map()

def normalize(name):
    return str(name).strip().upper()

# --------------------------------------------------
# Compute Geo Bounds
# --------------------------------------------------

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
# Sidebar Controls
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Display Options")
    show_income = st.toggle("Income Trend", value=False)
    show_poverty = st.toggle("Poverty Trend", value=False)
    show_markers = st.toggle("Markers", value=True)
    smooth_lines = st.toggle("Smooth Lines", value=False)

# --------------------------------------------------
# Hero Section
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-inner">
        <h1>Gateway Cities</h1>
        <div class="accent-line"></div>
        <p>
        A longitudinal investigation of immigration patterns,
        economic transformation, and structural inequality across Massachusetts.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# City Data
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

# --------------------------------------------------
# Build Map
# --------------------------------------------------

@st.cache_resource
def build_base_map(geojson, locations, center_lat, center_lon):
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=locations,
        z=[0] * len(locations),
        featureidkey="properties.TOWN",
        colorscale=[
            [0.0, "#e5e5e5"],
            [0.499, "#e5e5e5"],
            [0.5, "#E10600"],
            [0.999, "#E10600"],
            [1.0, "#111111"],
        ],
        zmin=0,
        zmax=2,
        showscale=False,
        marker_line_width=0.7,
        marker_line_color="#bbbbbb",
        hovertemplate="<b>%{location}</b><extra></extra>"
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode="markers",
        marker=dict(size=12, color="#111111"),
        name="Selected City"
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode="markers",
        marker=dict(size=12, color="#E10600"),
        name="Gateway City"
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode="markers",
        marker=dict(size=12, color="#e5e5e5"),
        name="Other Municipality"
    ))

    fig.update_layout(
        clickmode="event+select",
        mapbox=dict(
            style="white-bg",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8.3,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=1050,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dddddd",
            borderwidth=1
        )
    )

    return fig

base_fig = build_base_map(ma_geo, locations, center_lat, center_lon)

# --------------------------------------------------
# Show Map FIRST (before selector)
# --------------------------------------------------

selected_city = st.session_state.get("city_selector", cities["place_name"].iloc[0])
selected_city_norm = normalize(selected_city)

z_values = []
for town_name in locations:
    town_norm = normalize(town_name)
    if town_norm == selected_city_norm:
        z_values.append(2)
    elif town_norm in gateway_names:
        z_values.append(1)
    else:
        z_values.append(0)

# Work on a copy of the cached figure
fig_map = go.Figure(base_fig)
fig_map.data[0].z = z_values

event = st.plotly_chart(
    fig_map,
    use_container_width=True,
    on_select="rerun",
    key="map"
)

# --------------------------------------------------
# Handle Map Click Selection
# --------------------------------------------------

if event and "selection" in event and event["selection"]["points"]:
    clicked_town = event["selection"]["points"][0]["location"]

    matched_city = cities[
        cities["place_name"].str.upper().str.contains(clicked_town.upper())
    ]

    if not matched_city.empty:
        st.session_state["city_selector"] = matched_city["place_name"].iloc[0]
        st.rerun()

# --------------------------------------------------
# City Selector (NOW BELOW MAP)
# --------------------------------------------------

selected_city = st.selectbox(
    "Select City",
    cities["place_name"],
    index=0,
    label_visibility="collapsed",
    key="city_selector"
)

selected_city_norm = normalize(selected_city)

# --------------------------------------------------
# Data Section
# --------------------------------------------------

st.markdown('<div class="section">', unsafe_allow_html=True)

place_fips = cities[cities["place_name"] == selected_city]["place_fips"].values[0]

# ---------------------------
# Core Datasets
# ---------------------------

df_fb = get_foreign_born_percent(place_fips)
df_income = get_income_trend(place_fips)
df_poverty = get_poverty_trend(place_fips)

# Ensure numeric years and clear NaNs
df_fb["year"] = pd.to_numeric(df_fb["year"], errors="coerce")
df_income["year"] = pd.to_numeric(df_income["year"], errors="coerce")
df_poverty["year"] = pd.to_numeric(df_poverty["year"], errors="coerce")

df_fb = df_fb.dropna(subset=["year"])
df_income = df_income.dropna(subset=["year"])
df_poverty = df_poverty.dropna(subset=["year"])

# ---------------------------
# Overlap for structural chart (needs all 3)
# ---------------------------

df_struct = (
    df_fb
    .merge(df_income, on="year", how="outer")
    .merge(df_poverty, on="year", how="outer")
    .sort_values("year")
    .reset_index(drop=True)
)

if len(df_struct) > 0:
    df_struct["foreign_born_percent"] = df_struct["foreign_born_percent"].interpolate(method="linear", limit_direction="both")
    df_struct["median_income"] = df_struct["median_income"].interpolate(method="linear", limit_direction="both")
    df_struct["poverty_rate"] = df_struct["poverty_rate"].interpolate(method="linear", limit_direction="both")
    df_struct = df_struct.dropna(subset=["foreign_born_percent", "median_income", "poverty_rate"])

# ---------------------------
# Overlap for scatter (needs 2)
# ---------------------------

df_scatter = (
    df_fb
    .merge(df_poverty, on="year", how="outer")
    .sort_values("year")
    .reset_index(drop=True)
)

if len(df_scatter) > 0:
    df_scatter["foreign_born_percent"] = df_scatter["foreign_born_percent"].interpolate(method="linear", limit_direction="both")
    df_scatter["poverty_rate"] = df_scatter["poverty_rate"].interpolate(method="linear", limit_direction="both")
    df_scatter = df_scatter.dropna(subset=["foreign_born_percent", "poverty_rate"])

# ---------------------------
# Headline Metrics
# ---------------------------

if len(df_fb) > 0:
    latest_percent = df_fb["foreign_born_percent"].iloc[-1]
    start_val = df_fb["foreign_born_percent"].iloc[0]
    end_val = df_fb["foreign_born_percent"].iloc[-1]
    growth = ((end_val - start_val) / start_val) * 100 if start_val != 0 else np.nan
else:
    latest_percent = np.nan
    growth = np.nan

m1, m2 = st.columns(2)
m1.metric("Current Foreign-Born %", f"{latest_percent:.1f}%" if pd.notna(latest_percent) else "N/A")
m2.metric("Growth Since Start", f"{growth:.1f}%" if pd.notna(growth) else "N/A")

# ==================================================
# 1️⃣ STRUCTURAL SHIFT (Indexed Comparison)
# ==================================================

df_indexed = df_struct.copy()

# Robust check: explicitly ensure length is greater than zero
if len(df_indexed) > 0:
    base_fb = df_indexed["foreign_born_percent"].iloc[0]
    base_inc = df_indexed["median_income"].iloc[0]
    base_pov = df_indexed["poverty_rate"].iloc[0]

    df_indexed["Immigration Index"] = (
        df_indexed["foreign_born_percent"] / base_fb * 100
        if base_fb != 0 else np.nan
    )

    df_indexed["Income Index"] = (
        df_indexed["median_income"] / base_inc * 100
        if base_inc != 0 else np.nan
    )

    df_indexed["Poverty Index"] = (
        df_indexed["poverty_rate"] / base_pov * 100
        if base_pov != 0 else np.nan
    )

    fig_struct = go.Figure()

    fig_struct.add_trace(go.Scatter(
        x=df_indexed["year"],
        y=df_indexed["Immigration Index"],
        mode="lines+markers" if show_markers else "lines",
        name="Immigration (Indexed)",
        line=dict(color="#E10600")
    ))

    fig_struct.add_trace(go.Scatter(
        x=df_indexed["year"],
        y=df_indexed["Income Index"],
        mode="lines",
        name="Income (Indexed)",
        line=dict(color="#111111")
    ))

    fig_struct.add_trace(go.Scatter(
        x=df_indexed["year"],
        y=df_indexed["Poverty Index"],
        mode="lines",
        name="Poverty (Indexed)",
        line=dict(color="#888888")
    ))

    fig_struct.update_layout(
        template="plotly_white",
        title=f"Structural Shift (Base Year = 100) — {selected_city}",
        yaxis_title="Index (Base Year = 100)",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_struct, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Insufficient historical data to generate the Structural Shift index chart for this city.")

# ==================================================
# 2️⃣ IMMIGRATION vs POVERTY DYNAMIC RELATIONSHIP
# ==================================================

if len(df_scatter) > 1:
    fig_scatter = px.scatter(
        df_scatter,
        x="foreign_born_percent",
        y="poverty_rate",
        color="year",
        trendline="ols"
    )

    fig_scatter.update_layout(
        template="plotly_white",
        title=f"Immigration vs Poverty — {selected_city}",
        xaxis_title="Foreign-Born %",
        yaxis_title="Poverty Rate (%)",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Insufficient data to plot the Immigration vs Poverty relationship.")

# ==================================================
# 3️⃣ RAW IMMIGRATION TREND (Original)
# ==================================================

if len(df_fb) > 0:
    fig_fb = px.line(
        df_fb,
        x="year",
        y="foreign_born_percent",
        markers=show_markers,
    )

    if smooth_lines:
        fig_fb.update_traces(line_shape="spline")

    fig_fb.update_layout(
        template="plotly_white",
        title=f"Foreign-Born Population (%) — {selected_city}",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_fb, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Foreign-Born Population data is not available for this city.")

# ==================================================
# OPTIONAL: INCOME & POVERTY TOGGLES
# ==================================================

if show_income:
    if len(df_income) > 0:
        fig_income = px.line(
            df_income,
            x="year",
            y="median_income",
            markers=show_markers,
        )

        fig_income.update_layout(
            template="plotly_white",
            title=f"Median Household Income — {selected_city}",
        )

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_income, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Income data is not available for this city.")

if show_poverty:
    if len(df_poverty) > 0:
        fig_poverty = px.line(
            df_poverty,
            x="year",
            y="poverty_rate",
            markers=show_markers,
        )

        fig_poverty.update_layout(
            template="plotly_white",
            title=f"Poverty Rate — {selected_city}",
        )

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_poverty, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Poverty data is not available for this city.")

st.markdown('</div>', unsafe_allow_html=True)