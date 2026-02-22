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
# Page Config (MUST be first Streamlit command)
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
    <h1>Gateway Cities</h1>
    <div class="accent-line"></div>
    <p>
    A longitudinal investigation of immigration patterns,
    economic transformation, and structural inequality across Massachusetts.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Get City Data
# --------------------------------------------------

cities = get_cities(gateway_only=False)
gateway_cities = get_cities(gateway_only=True)

gateway_names = set(normalize(n) for n in gateway_cities["place_name"])

# --------------------------------------------------
# Map Section (Fades in under hero)
# --------------------------------------------------

st.markdown('<div class="map-container">', unsafe_allow_html=True)

selected_city = st.selectbox(
    "Select City",
    cities["place_name"],
    label_visibility="collapsed"
)

selected_city_norm = normalize(selected_city)

locations = []
z_values = []

for feature in ma_geo["features"]:
    town_name = feature["properties"]["TOWN"]
    locations.append(town_name)

    if normalize(town_name) == selected_city_norm:
        z_values.append(2)  # Selected city
    elif normalize(town_name) in gateway_names:
        z_values.append(1)  # Gateway city
    else:
        z_values.append(0)  # Non-gateway

fig_map = go.Figure(go.Choroplethmapbox(
    geojson=ma_geo,
    locations=locations,
    z=z_values,
    featureidkey="properties.TOWN",
    colorscale=[
        [0.0, "#f2f2f2"],
        [0.5, "#E10600"],
        [1.0, "#111111"]
    ],
    zmin=0,
    zmax=2,
    marker_line_width=0.6,
    marker_line_color="#222",
    showscale=False,
    hovertemplate="<b>%{location}</b><extra></extra>"
))

fig_map.update_layout(
    mapbox=dict(
        style="white-bg",
        fitbounds="locations"
    ),
    margin=dict(l=0, r=0, t=0, b=0),
)

st.markdown('<div class="fullscreen-map">', unsafe_allow_html=True)
st.plotly_chart(fig_map, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

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

# --------------------------------------------------
# Insight Block
# --------------------------------------------------

st.markdown("""
### Investigative Insight

Use this interface to explore how foreign-born population growth aligns
with structural economic indicators across time. Toggle contextual layers
to identify divergence, acceleration, or structural shifts.
""")

st.markdown('</div>', unsafe_allow_html=True)