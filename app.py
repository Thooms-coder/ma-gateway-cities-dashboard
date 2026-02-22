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
    get_foreign_born_by_country,
    get_income_trend,
    get_poverty_trend
)

# Mocking the missing queries for UI architecture purposes
def get_housing_burden(fips): return pd.DataFrame({"year": [2010, 2015, 2020, 2024], "rent_burden_percent": [30, 32, 35, 38]})

# Design System Colors (Colorblind Safe: Cobalt & Vermilion)
COLOR_TARGET = "#005ab5"
COLOR_BASE = "#dc3220"
COLOR_BG_LIGHT = "#f4f5f6"
COLOR_TEXT = "#2c2f33"

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
cities = get_cities(gateway_only=False)
gateway_cities = get_cities(gateway_only=True)

# Build normalized town → FIPS lookup
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
        n.replace(", Massachusetts", "").replace(" city", "").replace(" City", "").replace(" Town", "").strip()
    ) for n in gateway_cities["place_name"]
)

locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]
city_options = cities["place_name"].tolist()

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
    selected_city = st.selectbox(
        "Search Target Municipality",
        options=city_options,
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

df_struct = (
    df_fb
    .merge(df_income, on="year", how="outer")
    .merge(df_poverty, on="year", how="outer")
    .sort_values("year")
    .reset_index(drop=True)
)
df_struct = df_struct.interpolate(method="linear", limit_direction="both").dropna()

with col_export:
    st.markdown("<br>", unsafe_allow_html=True)
    if not df_struct.empty:
        csv = df_struct.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Master Dataset (CSV)",
            data=csv,
            file_name=f"{selected_city_norm}_longitudinal_data.csv",
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
    def build_map(geojson, locations, gw_names, place_fips, c_lat, c_lon):

        z_values = []
        selected_index = None

        for i, town_name in enumerate(locations):
            town_norm = normalize(town_name)

            # Match by FIPS (correct, reliable match)
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
                [0.5, COLOR_BASE],
                [0.999, COLOR_BASE],
                [1.0, COLOR_TARGET]
            ],
            zmin=0,
            zmax=2,
            showscale=False,
            marker_line_width=0.5,
            marker_line_color="#ffffff",
            hovertemplate="<b>%{location}</b><extra></extra>",
            selectedpoints=[selected_index] if selected_index is not None else None,
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.7))
        )

        fig = go.Figure(trace)

        fig.update_layout(
            clickmode="event+select",
            mapbox=dict(
                style="white-bg",
                center=dict(lat=c_lat, lon=c_lon),
                zoom=7.2
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=450
        )

        return fig

    fig_map = build_map(ma_geo, locations, gateway_names, place_fips, center_lat, center_lon)
    map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_select")

    st.markdown(f"""
        <div class="map-legend">
            <div class="legend-item"><span class="dot" style="background-color: {COLOR_TARGET};"></span> Target Municipality</div>
            <div class="legend-item"><span class="dot" style="background-color: {COLOR_BASE};"></span> Gateway City Baseline</div>
            <div class="legend-item"><span class="dot" style="background-color: #e9ecef; border: 1px solid #ccc;"></span> Rest of Commonwealth</div>
        </div>
    """, unsafe_allow_html=True)

    if map_event and "selection" in map_event and map_event["selection"]["points"]:
        clicked_town = map_event["selection"]["points"][0]["location"]
        matched_city = cities[cities["place_name"].str.upper().str.contains(clicked_town.upper())]
        if not matched_city.empty and matched_city["place_name"].iloc[0] != st.session_state.selected_city:
            st.session_state.selected_city = matched_city["place_name"].iloc[0]
            st.rerun()

    st.markdown('<hr style="border: 0; border-top: 1px solid #e1e4e8; margin: 30px 0;">', unsafe_allow_html=True)

    if len(df_fb) > 0:
        latest_percent = df_fb["foreign_born_percent"].iloc[-1]
        start_val = df_fb["foreign_born_percent"].iloc[0]
        growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else 0

        col_kpi, col_lede = st.columns([1, 2.5])
        with col_kpi:
            st.markdown(f'<div class="kpi-container"><div class="kpi-label">Foreign-Born Base</div><div class="kpi-value">{latest_percent:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-container"><div class="kpi-label">Period Growth Rate</div><div class="kpi-value">{growth:.1f}%</div></div>', unsafe_allow_html=True)
        with col_lede:
            trend_word = "surged" if growth > 10 else "grown" if growth > 0 else "declined"
            st.markdown(f"""
            <div style="font-family: 'Lora', serif; font-size:1.15rem; line-height: 1.7; color: #333; padding: 10px 20px;">
            Over the observed period, the foreign-born population in <b>{st.session_state.selected_city}</b> has {trend_word} by {abs(growth):.1f}%, now representing {latest_percent:.1f}% of the total community. This demographic shift provides the foundation for examining localized economic transitions, housing pressures, and wealth distribution.
            </div>
            """, unsafe_allow_html=True)

# ==================================================
# SECTION 2: DEMOGRAPHIC ORIGINS
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Demographic Origins")

    latest_year = df_fb["year"].max() if not df_fb.empty else 2024
    df_origins = get_foreign_born_by_country(place_fips, latest_year)

    if not df_origins.empty:
        df_origins_top = df_origins.head(10).sort_values("foreign_born", ascending=True)
        fig_origins = px.bar(
            df_origins_top,
            x="foreign_born",
            y="country_label",
            orientation='h',
            title=f"Top 10 Origin Countries ({latest_year})"
        )
        fig_origins.update_traces(marker_color=COLOR_TARGET)
        fig_origins.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Population Estimate",
            yaxis_title="",
            font=dict(family="Public Sans", color=COLOR_TEXT)
        )
        st.plotly_chart(fig_origins, use_container_width=True)
    else:
        st.info("Country of origin breakdown currently unavailable for this municipality.")

# ==================================================
# SECTION 3: ECONOMIC INDICATORS
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Economic Health & Poverty Status")

    col_ts1, col_ts2 = st.columns(2)

    with col_ts1:
        if not df_struct.empty and "median_income" in df_struct.columns:
            fig_inc = px.line(df_struct, x="year", y="median_income", title="Median Household Income ($)")
            fig_inc.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Public Sans"))
            fig_inc.update_traces(line_color=COLOR_TARGET, line_width=3)
            st.plotly_chart(fig_inc, use_container_width=True)

    with col_ts2:
        if not df_struct.empty and "poverty_rate" in df_struct.columns:
            fig_pov = px.line(df_struct, x="year", y="poverty_rate", title="Poverty Rate Deviation (%)")
            fig_pov.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Public Sans"))
            fig_pov.update_traces(line_color=COLOR_BASE, line_width=3)
            st.plotly_chart(fig_pov, use_container_width=True)

# ==================================================
# SECTION 4: TRAJECTORY ANALYSIS
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Structural Trajectory: Income vs. Immigration")
    st.markdown("<p style='color: #586069; font-size: 0.95rem;'>This connected scatterplot traces the municipality's economic and demographic movement year-over-year. A consistent upward and rightward trajectory indicates simultaneous growth in median income and foreign-born population.</p>", unsafe_allow_html=True)

    if len(df_struct) > 1:
        fig_traj = go.Figure()

        fig_traj.add_trace(go.Scatter(
            x=df_struct["foreign_born_percent"],
            y=df_struct["median_income"],
            mode='lines+markers+text',
            text=df_struct["year"],
            textposition="top center",
            marker=dict(size=8, color=df_struct["year"], colorscale="Blues", showscale=False),
            line=dict(color=COLOR_TARGET, width=2),
            hovertemplate="<b>%{text}</b><br>Foreign-Born: %{x:.1f}%<br>Income: $%{y:,.0f}<extra></extra>"
        ))

        fig_traj.add_trace(go.Scatter(
            x=[df_struct["foreign_born_percent"].iloc[0], df_struct["foreign_born_percent"].iloc[-1]],
            y=[df_struct["median_income"].iloc[0], df_struct["median_income"].iloc[-1]],
            mode='markers',
            marker=dict(size=12, color=[COLOR_BASE, COLOR_TARGET], symbol=['circle-open', 'circle']),
            hoverinfo='skip',
            showlegend=False
        ))

        fig_traj.update_layout(
            template="plotly_white",
            xaxis_title="Foreign-Born Population (%)",
            yaxis_title="Median Household Income ($)",
            margin=dict(l=40, r=40, t=40, b=40),
            font=dict(family="Public Sans"),
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig_traj, use_container_width=True)

# ==================================================
# SECTION 5: METHODOLOGY
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("#### Data Responsibility & Methodology")
    st.markdown("""
    <div style="font-size: 0.9rem; line-height: 1.6; color: #586069;">
    <strong>1. Transparency & Accuracy:</strong> All figures are derived directly from the U.S. Census American Community Survey (ACS) 5-Year Estimates. Margins of error (MOE) are preserved in the backend.<br><br>
    <strong>2. Journalistic Framing:</strong> This platform avoids causal claims without rigorous statistical testing. Correlation visualized across demographic and economic panels is intended to surface trends for localized reporting, rather than draw definitive conclusions.<br><br>
    <strong>3. Limitations:</strong> ACS 5-year estimates smooth out short-term volatility. Data represented here should be cross-referenced with local municipal records where applicable.
    </div>
    """, unsafe_allow_html=True)