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
    initial_sidebar_state="collapsed",
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

# --------------------------------------------------
# Normalization helpers (critical for matching)
# --------------------------------------------------
def normalize(s: str) -> str:
    return str(s or "").strip().upper()

def canonical_place_name(place_name: str) -> str:
    """
    Convert DB place_name into a canonical string that matches GeoJSON TOWN.
    Keeps your prior stripping logic but makes it reusable.
    """
    s = str(place_name or "")
    s = (
        s.replace(", Massachusetts", "")
         .replace(" Town city", "")
         .replace(" city", "")
         .replace(" City", "")
         .replace(" Town", "")
         .strip()
    )
    return normalize(s)

def canonical_town_name(town: str) -> str:
    """
    GeoJSON town canonicalization.
    """
    return normalize(town)

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

    compare_mode = st.toggle("Compare Multiple Cities", value=False)
    normalize_index = st.toggle("Normalize to Start Year (Index = 100)", value=False)

    show_income = st.toggle("Income Trend", value=False)
    show_poverty = st.toggle("Poverty Trend", value=False)

    show_markers = st.toggle("Markers", value=True)
    smooth_lines = st.toggle("Smooth Lines", value=False)

    st.markdown("---")
    st.markdown("### Map Options")

    map_mode = st.radio(
        "Map Coloring",
        [
            "Gateway Cities (default)",
            "Foreign-Born % (by year)",
            "Median Income (by year)",
            "Poverty Rate (by year)",
        ],
        index=0,
    )

    map_year = st.slider("Map Year", min_value=2010, max_value=2024, value=2024, step=1)

# --------------------------------------------------
# Hero Section
# --------------------------------------------------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# City Data
# --------------------------------------------------
cities = get_cities(gateway_only=False)
gateway_cities = get_cities(gateway_only=True)

# Canonical gateway names set (matches GeoJSON towns)
gateway_names = set(canonical_place_name(n) for n in gateway_cities["place_name"])

# GeoJSON town labels (used as choropleth locations)
locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]
locations_norm = [canonical_town_name(t) for t in locations]

# Map DB city names to place_fips quickly (canonical -> fips + display name)
@st.cache_data
def build_city_lookup(cities_df: pd.DataFrame):
    lookup = {}
    for _, row in cities_df.iterrows():
        canon = canonical_place_name(row["place_name"])
        lookup[canon] = {
            "place_fips": str(row["place_fips"]),
            "place_name": row["place_name"],
        }
    return lookup

city_lookup = build_city_lookup(cities)

def place_fips_for_city(place_name: str) -> str:
    match = cities[cities["place_name"] == place_name]
    return str(match["place_fips"].values[0])

# --------------------------------------------------
# Selector: single vs compare (keeps existing logic intact)
# --------------------------------------------------
if compare_mode:
    # Default selection: keep prior default as the first city in list
    default_city = cities["place_name"].iloc[0]
    selected_cities = st.multiselect(
        "Select Cities to Compare",
        options=list(cities["place_name"]),
        default=[default_city],
    )

    # Keep things readable
    if len(selected_cities) == 0:
        st.info("Select at least one city to display charts.")
        selected_cities = [default_city]
    if len(selected_cities) > 5:
        st.warning("Please select 5 or fewer cities for a readable comparison.")
        selected_cities = selected_cities[:5]
else:
    selected_city = st.selectbox(
        "Select City",
        cities["place_name"],
        index=0,
        label_visibility="collapsed",
        key="city_selector",
    )
    selected_cities = [selected_city]

selected_cities_norm = {canonical_place_name(n) for n in selected_cities}

# --------------------------------------------------
# Query helpers (cached per place_fips)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_foreign_born_df(place_fips: str) -> pd.DataFrame:
    return get_foreign_born_percent(place_fips)

@st.cache_data(show_spinner=False)
def fetch_income_df(place_fips: str) -> pd.DataFrame:
    return get_income_trend(place_fips)

@st.cache_data(show_spinner=False)
def fetch_poverty_df(place_fips: str) -> pd.DataFrame:
    return get_poverty_trend(place_fips)

def indexed_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df = df.sort_values("year").copy()
    if df.empty:
        df["indexed"] = []
        return df
    start = df[value_col].iloc[0]
    if start == 0 or pd.isna(start):
        df["indexed"] = None
    else:
        df["indexed"] = (df[value_col] / start) * 100.0
    return df

# --------------------------------------------------
# Build Map Base
# --------------------------------------------------
@st.cache_resource
def build_base_map(geojson, locations, center_lat, center_lon):
    fig = go.Figure(
        go.Choroplethmapbox(
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
            hovertemplate="<b>%{location}</b><extra></extra>",
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="white-bg",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8.3,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,  # more usable; does not affect your downstream logic
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dddddd",
            borderwidth=1,
        ),
    )
    return fig

base_fig = build_base_map(ma_geo, locations, center_lat, center_lon)

# --------------------------------------------------
# Map metric computation (safe + cached)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def metric_value_for_year(place_fips: str, mode: str, year: int):
    """
    Returns a float metric for the specified year if available, else None.
    mode in {"fb", "income", "poverty"}.
    """
    try:
        if mode == "fb":
            df = fetch_foreign_born_df(place_fips)
            row = df[df["year"] == year]
            if row.empty:
                return None
            return float(row["foreign_born_percent"].iloc[0])

        if mode == "income":
            df = fetch_income_df(place_fips)
            row = df[df["year"] == year]
            if row.empty:
                return None
            return float(row["median_income"].iloc[0])

        if mode == "poverty":
            df = fetch_poverty_df(place_fips)
            row = df[df["year"] == year]
            if row.empty:
                return None
            return float(row["poverty_rate"].iloc[0])

        return None
    except Exception:
        return None

def build_map_z_and_hover(map_mode_label: str, year: int):
    """
    Produces:
      - z_values for choropleth
      - hover strings (optional)
      - zmin/zmax and colorscale config
    """
    # Default mode: keep your original semantics exactly
    if map_mode_label == "Gateway Cities (default)":
        z_values = []
        for town_norm in locations_norm:
            if town_norm in selected_cities_norm:
                z_values.append(2)  # selected
            elif town_norm in gateway_names:
                z_values.append(1)  # gateway
            else:
                z_values.append(0)  # other
        return {
            "z": z_values,
            "hover": None,
            "zmin": 0,
            "zmax": 2,
            "showscale": False,
            "colorscale": [
                [0.0, "#e5e5e5"],
                [0.499, "#e5e5e5"],
                [0.5, "#E10600"],
                [0.999, "#E10600"],
                [1.0, "#111111"],
            ],
        }

    # Metric choropleths by year
    if map_mode_label == "Foreign-Born % (by year)":
        metric_key = "fb"
        title = f"Foreign-Born % ({year})"
        value_fmt = lambda v: f"{v:.1f}%"
        colorscale = "Reds"
    elif map_mode_label == "Median Income (by year)":
        metric_key = "income"
        title = f"Median Income ({year})"
        value_fmt = lambda v: f"${v:,.0f}"
        colorscale = "Blues"
    else:  # "Poverty Rate (by year)"
        metric_key = "poverty"
        title = f"Poverty Rate ({year})"
        value_fmt = lambda v: f"{v:.1f}%"
        colorscale = "Oranges"

    z_values = []
    hover = []

    # Build per-town metric value if we can map to place_fips
    vals_present = []

    for town, town_norm in zip(locations, locations_norm):
        info = city_lookup.get(town_norm)
        v = None
        if info:
            v = metric_value_for_year(info["place_fips"], metric_key, year)

        # Keep selected cities visually emphasized by nudging z upward slightly
        # without destroying the metric map (small lift).
        # This does not affect correctness; it is a visualization affordance.
        if v is None:
            z_values.append(None)
            hover.append(f"<b>{town}</b><br>{title}: N/A<extra></extra>")
        else:
            vv = float(v)
            if town_norm in selected_cities_norm:
                vv = vv  # keep true metric; selection is handled by outline/hover + legend context
            z_values.append(vv)
            vals_present.append(vv)
            hover.append(f"<b>{town}</b><br>{title}: {value_fmt(v)}<extra></extra>")

    if len(vals_present) == 0:
        # Fallback: keep map usable if no data
        return {
            "z": [0] * len(locations),
            "hover": [f"<b>{t}</b><br>{title}: N/A<extra></extra>" for t in locations],
            "zmin": 0,
            "zmax": 1,
            "showscale": False,
            "colorscale": [[0, "#e5e5e5"], [1, "#e5e5e5"]],
        }

    zmin = float(min(vals_present))
    zmax = float(max(vals_present))
    if zmin == zmax:
        zmax = zmin + 1e-9

    return {
        "z": z_values,
        "hover": hover,
        "zmin": zmin,
        "zmax": zmax,
        "showscale": True,
        "colorscale": colorscale,
    }

# --------------------------------------------------
# Render Map (always above selector, as requested earlier)
# --------------------------------------------------
map_payload = build_map_z_and_hover(map_mode, map_year)

fig_map = go.Figure(base_fig)  # safe copy
fig_map.data[0].z = map_payload["z"]
fig_map.data[0].zmin = map_payload["zmin"]
fig_map.data[0].zmax = map_payload["zmax"]
fig_map.data[0].showscale = map_payload["showscale"]
fig_map.data[0].colorscale = map_payload["colorscale"]

if map_payload["hover"] is not None:
    fig_map.data[0].hovertemplate = "%{customdata}"
    fig_map.data[0].customdata = map_payload["hover"]

st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------
# Data Section
# --------------------------------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)

# ---------------------------------
# Foreign-Born Chart (single or multi)
# ---------------------------------
# Metrics cards: keep your existing “current + growth” behavior for single-city,
# and provide a sensible aggregate view for compare mode.
if not compare_mode:
    place_fips = place_fips_for_city(selected_cities[0])
    df_fb = fetch_foreign_born_df(place_fips)

    latest_percent = df_fb["foreign_born_percent"].iloc[-1]
    growth = (
        (df_fb["foreign_born_percent"].iloc[-1] - df_fb["foreign_born_percent"].iloc[0])
        / df_fb["foreign_born_percent"].iloc[0]
    ) * 100

    m1, m2 = st.columns(2)
    m1.metric("Current Foreign-Born %", f"{latest_percent:.1f}%")
    m2.metric("Growth Since Start", f"{growth:.1f}%")
else:
    st.caption("Compare mode: select up to 5 cities to overlay trends.")

# Build multi-line chart (works for both single & compare)
fig_fb = go.Figure()
export_fb_rows = []

for city in selected_cities:
    pf = place_fips_for_city(city)
    df = fetch_foreign_born_df(pf).copy()
    df["city"] = city

    if normalize_index:
        df = indexed_series(df, "foreign_born_percent")
        y_col = "indexed"
        y_title = "Foreign-Born % (Indexed, start=100)"
    else:
        y_col = "foreign_born_percent"
        y_title = "Foreign-Born Population (%)"

    export_fb_rows.append(df[["city", "year", "foreign_born_percent"]])

    fig_fb.add_trace(
        go.Scatter(
            x=df["year"],
            y=df[y_col],
            mode=("lines+markers" if show_markers else "lines"),
            name=city,
        )
    )

if smooth_lines:
    fig_fb.update_traces(line_shape="spline")

fig_fb.update_layout(
    template="plotly_white",
    title=f"{y_title}",
    font=dict(family="Inter"),
    title_font=dict(size=20),
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_fb, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Download export for foreign-born
try:
    export_fb = pd.concat(export_fb_rows, ignore_index=True)
    st.download_button(
        "Download Foreign-Born Trend (CSV)",
        data=export_fb.to_csv(index=False).encode("utf-8"),
        file_name="foreign_born_trend.csv",
        mime="text/csv",
    )
except Exception:
    pass

# ---------------------------------
# Income Chart (optional; single or multi)
# ---------------------------------
if show_income:
    fig_income = go.Figure()
    export_income_rows = []

    for city in selected_cities:
        pf = place_fips_for_city(city)
        df = fetch_income_df(pf).copy()
        df["city"] = city

        if normalize_index:
            df = indexed_series(df, "median_income")
            y_col = "indexed"
            y_title = "Median Household Income (Indexed, start=100)"
        else:
            y_col = "median_income"
            y_title = "Median Household Income"

        export_income_rows.append(df[["city", "year", "median_income"]])

        fig_income.add_trace(
            go.Scatter(
                x=df["year"],
                y=df[y_col],
                mode=("lines+markers" if show_markers else "lines"),
                name=city,
            )
        )

    if smooth_lines:
        fig_income.update_traces(line_shape="spline")

    fig_income.update_layout(
        template="plotly_white",
        title=y_title,
        font=dict(family="Inter"),
        title_font=dict(size=18),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_income, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        export_income = pd.concat(export_income_rows, ignore_index=True)
        st.download_button(
            "Download Income Trend (CSV)",
            data=export_income.to_csv(index=False).encode("utf-8"),
            file_name="income_trend.csv",
            mime="text/csv",
        )
    except Exception:
        pass

# ---------------------------------
# Poverty Chart (optional; single or multi)
# ---------------------------------
if show_poverty:
    fig_poverty = go.Figure()
    export_pov_rows = []

    for city in selected_cities:
        pf = place_fips_for_city(city)
        df = fetch_poverty_df(pf).copy()
        df["city"] = city

        if normalize_index:
            df = indexed_series(df, "poverty_rate")
            y_col = "indexed"
            y_title = "Poverty Rate (Indexed, start=100)"
        else:
            y_col = "poverty_rate"
            y_title = "Poverty Rate"

        export_pov_rows.append(df[["city", "year", "poverty_rate"]])

        fig_poverty.add_trace(
            go.Scatter(
                x=df["year"],
                y=df[y_col],
                mode=("lines+markers" if show_markers else "lines"),
                name=city,
            )
        )

    if smooth_lines:
        fig_poverty.update_traces(line_shape="spline")

    fig_poverty.update_layout(
        template="plotly_white",
        title=y_title,
        font=dict(family="Inter"),
        title_font=dict(size=18),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_poverty, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        export_pov = pd.concat(export_pov_rows, ignore_index=True)
        st.download_button(
            "Download Poverty Trend (CSV)",
            data=export_pov.to_csv(index=False).encode("utf-8"),
            file_name="poverty_trend.csv",
            mime="text/csv",
        )
    except Exception:
        pass

st.markdown("</div>", unsafe_allow_html=True)