import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
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

# --------------------------------------------------
# Prepare Locations Once
# --------------------------------------------------

locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]

# --------------------------------------------------
# Cached Base Map (Static Geometry + Legend)
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

    # Manual Legend
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


# Build static base once
base_fig = build_base_map(ma_geo, locations, center_lat, center_lon)

# --------------------------------------------------
# Default Selection
# --------------------------------------------------

selected_city = cities["place_name"].iloc[0]
selected_city_norm = normalize(selected_city)

# Compute dynamic z values
z_values = []

for town_name in locations:
    town_norm = normalize(town_name)

    if town_norm == selected_city_norm:
        z_values.append(2)
    elif town_norm in gateway_names:
        z_values.append(1)
    else:
        z_values.append(0)

# Update cached base map safely
fig_map = base_fig.copy()
fig_map.data[0].z = z_values

st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------
# Dropdown Below Map
# --------------------------------------------------

selected_city = st.selectbox(
    "Select City",
    cities["place_name"],
    index=cities["place_name"].tolist().index(selected_city),
    label_visibility="collapsed"
)

selected_city_norm = normalize(selected_city)

# --------------------------------------------------
# Data Section
# --------------------------------------------------

st.markdown('<div class="section">', unsafe_allow_html=True)

place_fips = cities[cities["place_name"] == selected_city]["place_fips"].values[0]

df_fb = get_foreign_born_percent(place_fips)

latest_percent = df_fb["foreign_born_percent"].iloc[-1]
growth = (
    (df_fb["foreign_born_percent"].iloc[-1] - df_fb["foreign_born_percent"].iloc[0])
    / df_fb["foreign_born_percent"].iloc[0]
) * 100

m1, m2 = st.columns(2)
m1.metric("Current Foreign-Born %", f"{latest_percent:.1f}%")
m2.metric("Growth Since Start", f"{growth:.1f}%")

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
    font=dict(family="Inter"),
    title_font=dict(size=20),
    margin=dict(l=20, r=20, t=60, b=20),
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_fb, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Optional Income
# --------------------------------------------------

if show_income:
    df_income = get_income_trend(place_fips)

    fig_income = px.line(
        df_income,
        x="year",
        y="median_income",
        markers=show_markers,
    )

    if smooth_lines:
        fig_income.update_traces(line_shape="spline")

    fig_income.update_layout(
        template="plotly_white",
        title=f"Median Household Income — {selected_city}",
        font=dict(family="Inter"),
        title_font=dict(size=18),
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_income, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Optional Poverty
# --------------------------------------------------

if show_poverty:
    df_poverty = get_poverty_trend(place_fips)

    fig_poverty = px.line(
        df_poverty,
        x="year",
        y="poverty_rate",
        markers=show_markers,
    )

    if smooth_lines:
        fig_poverty.update_traces(line_shape="spline")

    fig_poverty.update_layout(
        template="plotly_white",
        title=f"Poverty Rate — {selected_city}",
        font=dict(family="Inter"),
        title_font=dict(size=18),
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_poverty, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)