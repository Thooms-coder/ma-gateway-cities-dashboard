import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
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
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
# Sidebar Controls (ADD ONLY)
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Display Options")

    # NEW (does not affect default behavior)
    compare_mode = st.toggle("Compare Multiple Cities", value=False)
    map_metric_mode = st.toggle("Color Map by Metric", value=False)
    map_year = st.slider("Map Year", 2010, 2024, 2024)

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
# Build Map (UNCHANGED)
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
# Map Logic (STRICTLY ADDITIVE)
# --------------------------------------------------

if compare_mode:
    selected_cities = st.multiselect(
        "Select Cities",
        cities["place_name"],
        default=[cities["place_name"].iloc[0]]
    )
else:
    selected_cities = [
        st.session_state.get("city_selector", cities["place_name"].iloc[0])
    ]

selected_norms = set(normalize(c) for c in selected_cities)

z_values = []

if not map_metric_mode:
    # EXACT ORIGINAL BEHAVIOR
    for town_name in locations:
        town_norm = normalize(town_name)
        if town_norm in selected_norms:
            z_values.append(2)
        elif town_norm in gateway_names:
            z_values.append(1)
        else:
            z_values.append(0)
else:
    # Metric coloring (foreign-born % only to keep correctness)
    values = []
    for town_name in locations:
        town_norm = normalize(town_name)
        match = cities[cities["place_name"].str.upper().str.contains(town_norm)]
        if not match.empty:
            pf = match["place_fips"].values[0]
            df = get_foreign_born_percent(pf)
            row = df[df["year"] == map_year]
            if not row.empty:
                values.append(row["foreign_born_percent"].values[0])
            else:
                values.append(0)
        else:
            values.append(0)
    z_values = values
    base_fig.data[0].showscale = True
    base_fig.data[0].colorscale = "Reds"

fig_map = base_fig
fig_map.data[0].z = z_values

st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------
# City Selector (UNCHANGED)
# --------------------------------------------------

if not compare_mode:
    selected_city = st.selectbox(
        "Select City",
        cities["place_name"],
        index=0,
        label_visibility="collapsed",
        key="city_selector"
    )
    selected_cities = [selected_city]

# --------------------------------------------------
# Data Section (ENHANCED BUT SAFE)
# --------------------------------------------------

st.markdown('<div class="section">', unsafe_allow_html=True)

fig_fb = go.Figure()

for city in selected_cities:
    pf = cities[cities["place_name"] == city]["place_fips"].values[0]
    df_fb = get_foreign_born_percent(pf)

    fig_fb.add_trace(go.Scatter(
        x=df_fb["year"],
        y=df_fb["foreign_born_percent"],
        mode="lines+markers" if show_markers else "lines",
        name=city
    ))

if smooth_lines:
    fig_fb.update_traces(line_shape="spline")

fig_fb.update_layout(
    template="plotly_white",
    title="Foreign-Born Population (%)",
    font=dict(family="Inter"),
    title_font=dict(size=20),
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_fb, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)