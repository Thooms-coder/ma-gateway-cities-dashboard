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
        pass

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

def get_abbreviation(name):
    """Generates a 2-letter abbreviation for map labels."""
    clean_name = str(name).replace(" city", "").replace(" Town", "").strip()
    words = clean_name.split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    return clean_name[:2].upper()

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

# Global Session State for synchronized selection
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
    and structural economic shifts across Massachusetts municipalities.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Global Control (Syncs Map & Data)
# --------------------------------------------------

st.markdown("### Search Target Municipality")
# Binding the selectbox directly to session_state using the key parameter
selected_city = st.selectbox(
    "Target Municipality",
    options=city_options,
    label_visibility="collapsed",
    key="selected_city" 
)

selected_city_norm = normalize(st.session_state.selected_city)
place_fips = cities[cities["place_name"] == st.session_state.selected_city]["place_fips"].values[0]

# --------------------------------------------------
# Architecture: Multi-Tab Layout
# --------------------------------------------------

tab_demo, tab_ethics = st.tabs([
    "Demographics & Economy", 
    "Methodology & Ethics"
])

# ==================================================
# TAB 1: DEMOGRAPHICS & ECONOMY
# ==================================================
with tab_demo:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # --- Map & Metrics Layout ---
    col_map, col_data = st.columns([1.2, 1])

    with col_map:
        st.markdown("#### Geographic Context")
        
        @st.cache_resource
        def build_base_map(geojson, locations, center_lat, center_lon):
            fig = go.Figure(go.Choroplethmapbox(
                geojson=geojson, locations=locations, z=[0] * len(locations),
                featureidkey="properties.TOWN",
                colorscale=[[0.0, "#e5e5e5"], [0.499, "#e5e5e5"], [0.5, "#8b0000"], [0.999, "#8b0000"], [1.0, "#111111"]],
                zmin=0, zmax=2, showscale=False, marker_line_width=0.5, marker_line_color="#ffffff",
                hovertemplate="<b>%{location}</b><extra></extra>"
            ))

            gw_locs = [loc for loc in locations if normalize(loc) in gateway_names]
            gw_texts = [get_abbreviation(loc) for loc in gw_locs]
            
            fig.add_trace(go.Scattermapbox(
                lat=[None]*len(gw_locs), lon=[None]*len(gw_locs), 
                mode="markers", marker=dict(size=10, color="#8b0000"), name="Gateway City"
            ))

            fig.update_layout(
                clickmode="event+select",
                mapbox=dict(style="white-bg", center=dict(lat=center_lat, lon=center_lon), zoom=7),
                margin=dict(l=0, r=0, t=0, b=0), height=700,
                legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.9)")
            )
            return fig

        base_fig = build_base_map(ma_geo, locations, center_lat, center_lon)
        
        z_values = []
        for town_name in locations:
            town_norm = normalize(town_name)
            if town_norm == selected_city_norm:
                z_values.append(2)
            elif town_norm in gateway_names:
                z_values.append(1)
            else:
                z_values.append(0)

        fig_map = go.Figure(base_fig)
        fig_map.data[0].z = z_values

        map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_select")
        
        # Handle map click to instantly sync dropdown
        if map_event and "selection" in map_event and map_event["selection"]["points"]:
            clicked_town = map_event["selection"]["points"][0]["location"]
            matched_city = cities[cities["place_name"].str.upper().str.contains(clicked_town.upper())]
            
            if not matched_city.empty and matched_city["place_name"].iloc[0] != st.session_state.selected_city:
                st.session_state.selected_city = matched_city["place_name"].iloc[0]
                st.rerun()

    # --- Data Fetching & Processing ---
    df_fb = get_foreign_born_percent(place_fips)
    df_income = get_income_trend(place_fips)
    df_poverty = get_poverty_trend(place_fips)

    for df in [df_fb, df_income, df_poverty]:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df.dropna(subset=["year"], inplace=True)

    df_struct = df_fb.merge(df_income, on="year", how="outer").merge(df_poverty, on="year", how="outer").sort_values("year").reset_index(drop=True)
    df_struct = df_struct.interpolate(method="linear", limit_direction="both").dropna()

    with col_data:
        st.markdown("#### Analytical Brief")
        if len(df_fb) > 0:
            latest_percent = df_fb["foreign_born_percent"].iloc[-1]
            start_val = df_fb["foreign_born_percent"].iloc[0]
            growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else 0
            
            if growth > 15:
                st.warning(f"Rapid Demographic Shift: Foreign-born population expanded by {growth:.1f}%. Cross-reference housing capacity.")
            elif growth < 0:
                st.info(f"Population Contraction: Foreign-born population declined by {abs(growth):.1f}%.")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="kpi-container"><div class="kpi-label">Foreign-Born %</div><div class="kpi-value">{latest_percent:.1f}%</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="kpi-container"><div class="kpi-label">Growth (Full Period)</div><div class="kpi-value">{growth:.1f}%</div></div>', unsafe_allow_html=True)

            trend_word = "surged" if growth > 10 else "grown" if growth > 0 else "declined"
            st.markdown(f"""
            <div style="background:#f4f4f4; padding:15px; border-left:3px solid #111; margin-top:10px; font-size:0.95rem;">
            <strong>Auto-Generated Lede:</strong> Over the observed period, the foreign-born population in {st.session_state.selected_city} has {trend_word} by {abs(growth):.1f}%, now representing {latest_percent:.1f}% of the total community. Investigative focus should track how this growth correlates with median income and poverty trajectories.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # --- Advanced Visualization: Parallel Coordinates ---
    
    if len(df_struct) > 1:
        st.markdown(f"#### Multidimensional Socioeconomic Analysis: {st.session_state.selected_city}")
        st.markdown("This visualization traces the relationship between time, immigration percentage, median income, and poverty rate. Following a single line horizontally reveals the socioeconomic state of the city for a specific year.")
        
        fig_par = px.parallel_coordinates(
            df_struct, 
            color="year", 
            dimensions=[
                dict(range=[min(df_struct["year"]), max(df_struct["year"])], label="Year", values=df_struct["year"]),
                dict(range=[min(df_struct["foreign_born_percent"]), max(df_struct["foreign_born_percent"])], label="Foreign-Born %", values=df_struct["foreign_born_percent"]),
                dict(range=[min(df_struct["median_income"]), max(df_struct["median_income"])], label="Median Income", values=df_struct["median_income"], tickformat=",.0f", tickprefix="$"),
                dict(range=[min(df_struct["poverty_rate"]), max(df_struct["poverty_rate"])], label="Poverty Rate (%)", values=df_struct["poverty_rate"])
            ],
            color_continuous_scale=px.colors.diverging.Tealrose,
        )
        fig_par.update_layout(margin=dict(l=40, r=40, t=40, b=20), height=400)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_par, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Time-Series Section ---
        st.markdown("#### Longitudinal Trends")
        col_ts1, col_ts2 = st.columns(2)
        
        with col_ts1:
            fig_inc = px.line(df_struct, x="year", y="median_income", title="Median Income Trend")
            fig_inc.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_inc, use_container_width=True)

        with col_ts2:
            fig_pov = px.line(df_struct, x="year", y="poverty_rate", title="Poverty Rate Trend")
            fig_pov.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pov, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# TAB 2: METHODOLOGY & ETHICS
# ==================================================
with tab_ethics:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("### Ethics, Transparency, & Data Responsibility")
    st.markdown("""
    This application is designed in strict accordance with the CivicHacks Ethics & Responsibility rubric.

    **1. Data Privacy & Security**
    * All demographic and economic data is sourced exclusively from the U.S. Census Bureau's American Community Survey (ACS) 5-Year Estimates.
    * No personally identifiable information (PII) or microdata is exposed or stored. All data is aggregated at the municipal level, ensuring complete privacy compliance.

    **2. Fairness & Human-Centeredness**
    * We explicitly separate correlation from interpretation. The platform visualizes the concurrent growth of foreign-born populations and economic indicators to identify structural realities, **not** to draw causal links or stigmatize communities.
    * Language across the dashboard is carefully constructed to maintain an investigative, journalistic, and objective standard.

    **3. Limitations & Transparency**
    * **ACS Margins of Error:** 5-Year estimates smooth out short-term volatility. Margins of Error (MOE) can be significant for smaller municipalities. The trends presented here are indicative of broader structural shifts and should be considered alongside localized reporting before publishing claims based on slight percentage differences.
    """)
    st.markdown('</div>', unsafe_allow_html=True)