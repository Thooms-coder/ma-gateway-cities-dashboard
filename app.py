import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd
import re
import textwrap

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

        # right before the if town_norm in town_fips_map:
        st.write("clicked_town:", clicked_town, "town_norm:", town_norm, "in_map:", town_norm in town_fips_map)

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
# SECTION 2: DEMOGRAPHIC ORIGINS
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Demographic Origins")

    primary_city = selected_cities[0]
    primary_fips = selected_fips[primary_city]

    # Get latest available year from already-fetched data
    df_primary = city_data[primary_city]["fb"]

    if not df_primary.empty:
        latest_year = df_primary["year"].max()
    else:
        latest_year = None

    if latest_year is not None:
        df_origins = get_foreign_born_by_country(primary_fips, latest_year)

        if not df_origins.empty:
            df_origins_top = (
                df_origins
                .head(10)
                .sort_values("foreign_born", ascending=True)
            )

            fig_origins = px.bar(
                df_origins_top,
                x="foreign_born",
                y="country_label",
                orientation='h',
                title=f"Top 10 Origin Countries — {primary_city} ({latest_year})"
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
            st.info(f"Country of origin breakdown unavailable for {primary_city}.")
    else:
        st.info(f"No foreign-born data available for {primary_city}.")

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
                    line=dict(width=3)
                ))

        fig_inc.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Year",
            yaxis_title="Median Household Income ($)",
            font=dict(family="Public Sans"),
            height=500,
            legend=dict(title="")
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
                    line=dict(width=3)
                ))

        fig_pov.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Year",
            yaxis_title="Poverty Rate (%)",
            font=dict(family="Public Sans"),
            height=500,
            legend=dict(title="")
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
        height=550
    )

    st.plotly_chart(fig_traj, use_container_width=True)

# ==================================================
# SECTION 5: METHODOLOGY
# ==================================================
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("#### Data Responsibility & Methodology")
    st.markdown(textwrap.dedent("""<div style="font-size: 0.9rem; line-height: 1.6; color: #586069;">
    <strong>1. Transparency & Accuracy:</strong> All figures are derived directly from the U.S. Census American Community Survey (ACS) 5-Year Estimates. Margins of error (MOE) are preserved in the backend.<br><br>"""), unsafe_allow_html=True)
    st.link_button("Open gateway city report", "https://www.census.gov/programs-surveys/acs/data.html")
    st.markdown(textwrap.dedent("""<strong>2. Journalistic Framing:</strong> This platform avoids causal claims without rigorous statistical testing. Correlation visualized across demographic and economic panels is intended to surface trends for localized reporting, rather than draw definitive conclusions.<br><br>
    <strong>3. Limitations:</strong> ACS 5-year estimates smooth out short-term volatility. Data represented here should be cross-referenced with local municipal records where applicable.
    </div>
    """), unsafe_allow_html=True)