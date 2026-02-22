import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd

from src.queries import (
    get_cities,
    get_foreign_born_percent,
    get_foreign_born_by_country,
    get_income_trend,
    get_poverty_trend
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css():
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# --------------------------------------------------
# Load GeoJSON
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
center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

# --------------------------------------------------
# Load Cities
# --------------------------------------------------
cities = get_cities(gateway_only=False)
gateway_cities = get_cities(gateway_only=True)

cities["place_fips"] = cities["place_fips"].astype(str)
gateway_cities["place_fips"] = gateway_cities["place_fips"].astype(str)

town_fips_map = {
    normalize(
        name.replace(", Massachusetts", "")
            .replace(" city", "")
            .replace(" City", "")
            .replace(" Town", "")
            .strip()
    ): fips
    for name, fips in zip(cities["place_name"], cities["place_fips"])
}

gateway_names = set(
    normalize(
        n.replace(", Massachusetts", "")
         .replace(" city", "")
         .replace(" City", "")
         .replace(" Town", "")
         .strip()
    )
    for n in gateway_cities["place_name"]
)

locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]
city_options = cities["place_name"].tolist()

if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = [city_options[0]]

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<section class="hero">
    <h1>Gateway Cities Investigative Dashboard</h1>
    <div class="accent-line"></div>
</section>
""", unsafe_allow_html=True)

# --------------------------------------------------
# City Selection
# --------------------------------------------------
selected_cities = st.multiselect(
    "Compare Municipalities (Max 3)",
    options=city_options,
    default=st.session_state.selected_cities,
    max_selections=3,
    key="selected_cities"
)

if not selected_cities:
    selected_cities = [city_options[0]]

primary_city = selected_cities[0]

selected_fips = {
    city: cities[cities["place_name"] == city]["place_fips"].values[0]
    for city in selected_cities
}

# --------------------------------------------------
# Load City Data
# --------------------------------------------------
city_data = {}

for city, fips in selected_fips.items():

    df_fb = get_foreign_born_percent(fips)
    df_income = get_income_trend(fips)
    df_poverty = get_poverty_trend(fips)

    for df in [df_fb, df_income, df_poverty]:
        if not df.empty:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df.dropna(subset=["year"], inplace=True)

    # STRICT OVERLAP ONLY (no interpolation)
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

# --------------------------------------------------
# Map Builder (fixed dependency)
# --------------------------------------------------
@st.cache_data
def build_map(geojson, locations, gw_names, town_fips_map, place_fips, c_lat, c_lon):

    z_values = []
    selected_index = None

    for i, town_name in enumerate(locations):
        town_norm = normalize(town_name)

        if town_norm in town_fips_map and town_fips_map[town_norm] == place_fips:
            z_values.append(2)
            selected_index = i
        elif town_norm in gw_names:
            z_values.append(1)
        else:
            z_values.append(0)

    trace = go.Choroplethmapbox(
        geojson=geojson,
        locations=locations,
        z=z_values,
        featureidkey="properties.TOWN",
        colorscale=[
            [0.0, "#e9ecef"],
            [0.499, "#e9ecef"],
            [0.5, "#dc3220"],
            [0.999, "#dc3220"],
            [1.0, "#005ab5"]
        ],
        zmin=0,
        zmax=2,
        showscale=False,
        marker_line_width=1.2,
        marker_line_color="rgba(60, 65, 75, 0.65)",
        hovertemplate="<b>%{location}</b><extra></extra>"
    )

    fig = go.Figure(trace)

    fig.update_layout(
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
    gateway_names,
    town_fips_map,
    selected_fips[primary_city],
    center_lat,
    center_lon
)

st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------
# KPI Section (fixed city reference)
# --------------------------------------------------
df_primary_fb = city_data[primary_city]["fb"]

if not df_primary_fb.empty and len(df_primary_fb) > 1:

    latest_percent = df_primary_fb["foreign_born_percent"].iloc[-1]
    start_val = df_primary_fb["foreign_born_percent"].iloc[0]

    growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else None

    st.metric("Foreign-Born Population (%)", f"{latest_percent:.1f}%")

    if growth is not None:
        st.metric("Growth Rate (%)", f"{growth:.1f}%")

# --------------------------------------------------
# Economic Charts (no fake interpolation)
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    fig_inc = go.Figure()
    for city, data in city_data.items():
        df = data["income"]
        if not df.empty:
            fig_inc.add_trace(go.Scatter(
                x=df["year"],
                y=df["median_income"],
                mode="lines",
                name=city
            ))
    st.plotly_chart(fig_inc, use_container_width=True)

with col2:
    fig_pov = go.Figure()
    for city, data in city_data.items():
        df = data["poverty"]
        if not df.empty:
            fig_pov.add_trace(go.Scatter(
                x=df["year"],
                y=df["poverty_rate"],
                mode="lines",
                name=city
            ))
    st.plotly_chart(fig_pov, use_container_width=True)

# --------------------------------------------------
# Trajectory (requires strict overlap)
# --------------------------------------------------
fig_traj = go.Figure()

for city, data in city_data.items():
    df = data["struct"]
    if not df.empty and len(df) > 1:
        fig_traj.add_trace(go.Scatter(
            x=df["foreign_born_percent"],
            y=df["median_income"],
            mode='lines+markers',
            name=city,
            text=df["year"]
        ))

st.plotly_chart(fig_traj, use_container_width=True)