import streamlit as st
from st_aggrid import AgGrid
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd
import re
import textwrap

# Country mapping (best effort)
import pycountry

from src.queries import (
    get_cities,
    get_gateway_fips,
    get_place_variable_trend,
    get_place_source_table_year
)

# --------------------------------------------------
# ACS Variable IDs (Warehouse-Controlled)
# --------------------------------------------------
VAR_FOREIGN_BORN_TOTAL = "B05002_013E"   # Foreign-born total
VAR_TOTAL_POP         = "B01003_001E"    # Total population
VAR_MEDIAN_INCOME     = "S1901_C01_012E" # Median household income (ACS Subject Table)
VAR_POVERTY_RATE      = "S1701_C03_001E" # Percent below poverty level (subject table series)

# B05006 (place of birth) – used for choropleth + top origins
SOURCE_BIRTH_TABLE = "B05006"

# --------------------------------------------------
# Design System Colors
# --------------------------------------------------
COLOR_TARGET = "#005ab5"   # selected
COLOR_BASE   = "#dc3220"   # gateway
COLOR_BG     = "#f4f5f6"
COLOR_TEXT   = "#2c2f33"
COLOR_BOSTON = "#009E73"   # Boston/Cambridge highlight

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CSS
# --------------------------------------------------
def load_css():
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    st.markdown("""
    <style>
    .section-card-marker { display:none; }

    div[data-testid="stVerticalBlock"]:has(.section-card-marker) {
        background: #ffffff;
        padding: 26px;
        border-radius: 6px;
        border: 1px solid #e1e4e8;
        margin-bottom: 22px;
    }

    .map-legend {
        display:flex;
        gap:18px;
        flex-wrap:wrap;
        padding: 10px 0 0 0;
        font-size: 0.85rem;
        color:#586069;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    .legend-item { display:flex; align-items:center; gap:8px; }
    .dot { height: 12px; width: 12px; border-radius: 2px; display:inline-block; }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# GeoJSON
# --------------------------------------------------
@st.cache_data
def load_ma_map():
    with open("data/ma_municipalities.geojson") as f:
        return json.load(f)

ma_geo = load_ma_map()

def normalize(name: str) -> str:
    return str(name).strip().upper()

def clean_place_label(name: str) -> str:
    # "Boston city, Massachusetts" -> "Boston"
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

    for feat in geojson["features"]:
        extract_coords(feat["geometry"]["coordinates"])

    return min(lats), max(lats), min(lons), max(lons)

min_lat, max_lat, min_lon, max_lon = get_geo_bounds(ma_geo)
center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

# --------------------------------------------------
# Cities / Gateway Registry
# --------------------------------------------------
cities = get_cities(gateway_only=False)
cities["place_fips"] = cities["place_fips"].astype(str)

gateway_fips = set(get_gateway_fips()["place_fips"].astype(str))

# Boston + Cambridge are highlighted (not gateway, but we show them)
EXTRA_CITY_NAMES = {"Boston city, Massachusetts", "Cambridge city, Massachusetts"}
extra_cities = cities[cities["place_name"].isin(list(EXTRA_CITY_NAMES))].copy()
extra_fips = set(extra_cities["place_fips"].astype(str))

# Dropdown shows ONLY gateway cities (as before)
city_options = (
    cities[cities["place_fips"].isin(gateway_fips)]["place_name"]
    .sort_values()
    .tolist()
)
if not city_options:
    st.error("No gateway cities found in gateway_cities table. Check is_gateway_city flags.")
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<section class="hero">
    <h1>Gateway Cities Investigative Dashboard</h1>
    <div class="accent-line"></div>
    <p>
        Longitudinal analysis of demographic change and economic structure across Massachusetts municipalities.
        Source: ACS 5-year estimates (2010–2024).
    </p>
</section>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Controls + Export
# --------------------------------------------------
col_search, col_export = st.columns([3, 1])

with col_search:
    st.markdown("""
**Explore Gateway Cities**

Select up to 3 Gateway Cities for side-by-side trend comparisons.
""")

    available_options = sorted(set(city_options + st.session_state.get("selected_cities", [])))

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
st.session_state.selected_city = primary_city

# --------------------------------------------------
# Helpers: trends from warehouse
# --------------------------------------------------
def _ensure_year(df: pd.DataFrame, year_col: str = "acs_end_year") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["year"] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=["year"]).sort_values("year")
    return out

def safe_trend(place_fips: str, variable_id: str, value_name: str) -> pd.DataFrame:
    df = get_place_variable_trend(place_fips, variable_id)
    if df is None or df.empty:
        return pd.DataFrame(columns=["year", value_name])
    df = _ensure_year(df, "acs_end_year")
    if df.empty:
        return pd.DataFrame(columns=["year", value_name])
    # expected columns: estimate, moe possibly
    df[value_name] = pd.to_numeric(df.get("estimate"), errors="coerce")
    df = df.dropna(subset=[value_name])
    return df[["year", value_name]].reset_index(drop=True)

def foreign_born_percent(place_fips: str) -> pd.DataFrame:
    fb = get_place_variable_trend(place_fips, VAR_FOREIGN_BORN_TOTAL)
    tp = get_place_variable_trend(place_fips, VAR_TOTAL_POP)
    if fb is None or tp is None or fb.empty or tp.empty:
        return pd.DataFrame(columns=["year", "foreign_born_percent"])

    fb = _ensure_year(fb, "acs_end_year")
    tp = _ensure_year(tp, "acs_end_year")
    if fb.empty or tp.empty:
        return pd.DataFrame(columns=["year", "foreign_born_percent"])

    fb["estimate_fb"] = pd.to_numeric(fb.get("estimate"), errors="coerce")
    tp["estimate_pop"] = pd.to_numeric(tp.get("estimate"), errors="coerce")

    df = fb[["year", "estimate_fb"]].merge(tp[["year", "estimate_pop"]], on="year", how="inner")
    df = df.dropna(subset=["estimate_fb", "estimate_pop"])
    df = df[df["estimate_pop"] != 0]
    df["foreign_born_percent"] = (df["estimate_fb"] / df["estimate_pop"]) * 100
    return df[["year", "foreign_born_percent"]].sort_values("year").reset_index(drop=True)

def latest_year_for_place(place_fips: str) -> int | None:
    df = get_place_variable_trend(place_fips, VAR_TOTAL_POP)
    if df is None or df.empty:
        return None
    yrs = pd.to_numeric(df["acs_end_year"], errors="coerce").dropna()
    return int(yrs.max()) if not yrs.empty else None

# --------------------------------------------------
# Load data per selected city (safe)
# --------------------------------------------------
city_data = {}

for city, fips in selected_fips.items():
    df_fb = foreign_born_percent(fips)
    df_income = safe_trend(fips, VAR_MEDIAN_INCOME, "median_income")
    df_poverty = safe_trend(fips, VAR_POVERTY_RATE, "poverty_rate")

    # strict overlap for structural panels (no fabricated years)
    if not df_fb.empty and not df_income.empty and not df_poverty.empty:
        df_struct = (
            df_fb.merge(df_income, on="year", how="inner")
                .merge(df_poverty, on="year", how="inner")
                .sort_values("year")
                .reset_index(drop=True)
        )
    else:
        df_struct = pd.DataFrame(columns=["year", "foreign_born_percent", "median_income", "poverty_rate"])

    city_data[city] = {
        "fb": df_fb,
        "income": df_income,
        "poverty": df_poverty,
        "struct": df_struct
    }

# Export (primary city overlap dataset)
with col_export:
    st.markdown("<br>", unsafe_allow_html=True)
    export_df = city_data.get(primary_city, {}).get("struct", pd.DataFrame())
    if not export_df.empty:
        st.download_button(
            label=f"Download {primary_city.split(',')[0]} Dataset (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{primary_city.replace(' ', '_')}_struct_overlap.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("No overlap dataset to export for this city.")

# --------------------------------------------------
# SECTION 1: MAP (with Boston/Cambridge highlight)
# --------------------------------------------------
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Geographic Context")

    # Build town -> fips map from registry names
    town_fips_map = {
        normalize(clean_place_label(name)): fips
        for name, fips in zip(cities["place_name"], cities["place_fips"])
    }

    # Allowed set for highlighting gateway cities
    allowed_gateway_names = set(
        normalize(clean_place_label(n))
        for n in cities[cities["place_fips"].isin(gateway_fips)]["place_name"]
    )

    boston_cambridge_names = set(
        normalize(clean_place_label(n))
        for n in extra_cities["place_name"]
    )

    locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]

    @st.cache_data
    def build_map(geojson, locations, allowed_gateway_names, town_fips_map_local,
                  selected_fips, c_lat, c_lon, boston_cambridge_names):

        z_values = []
        selected_index = None

        for i, town_name in enumerate(locations):
            town_norm = normalize(town_name)

            # selected municipality
            if town_norm in town_fips_map_local and town_fips_map_local[town_norm] == selected_fips:
                z_values.append(3)
                selected_index = i
            # Boston/Cambridge
            elif town_norm in boston_cambridge_names:
                z_values.append(2)
            # gateway cities
            elif town_norm in allowed_gateway_names:
                z_values.append(1)
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
                [0.5,  COLOR_BASE],
                [0.5,  COLOR_BOSTON],
                [0.75, COLOR_BOSTON],
                [0.75, COLOR_TARGET],
                [1.0,  COLOR_TARGET],
            ],
            zmin=0,
            zmax=3,
            showscale=False,
            marker_line_width=1.0,
            marker_line_color="rgba(60,65,75,0.5)",
            hovertemplate="<b>%{location}</b><extra></extra>",
            selectedpoints=[selected_index] if selected_index is not None else None,
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.85)),
        )

        fig = go.Figure(trace)
        fig.update_layout(
            clickmode="event+select",
            mapbox=dict(style="white-bg", center=dict(lat=c_lat, lon=c_lon), zoom=8),
            margin=dict(l=0, r=0, t=0, b=0),
            height=825,
        )
        return fig

    fig_map = build_map(
        ma_geo,
        locations,
        allowed_gateway_names,
        town_fips_map,
        primary_fips,
        center_lat,
        center_lon,
        boston_cambridge_names
    )

    map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_select")

    st.markdown(f"""
        <div class="map-legend">
            <div class="legend-item"><span class="dot" style="background:{COLOR_TARGET};"></span>Selected Municipality</div>
            <div class="legend-item"><span class="dot" style="background:{COLOR_BOSTON};"></span>Boston & Cambridge</div>
            <div class="legend-item"><span class="dot" style="background:{COLOR_BASE};"></span>Gateway Cities</div>
            <div class="legend-item"><span class="dot" style="background:#E9ECEF; border:1px solid #ccc;"></span>Other Municipalities</div>
        </div>
    """, unsafe_allow_html=True)

    # click-to-add: gateway cities only
    if map_event and "selection" in map_event and map_event["selection"]["points"]:
        clicked_town = map_event["selection"]["points"][0]["location"]
        town_norm = normalize(clicked_town)

        if town_norm in town_fips_map:
            new_fips = town_fips_map[town_norm]
            # only allow gateway picks via map
            if new_fips in gateway_fips:
                new_city = cities[cities["place_fips"] == new_fips]["place_name"].iloc[0]
                if new_city not in st.session_state["selected_cities"]:
                    if len(st.session_state["selected_cities"]) < 3:
                        st.session_state["selected_cities"].append(new_city)
                    else:
                        st.session_state["selected_cities"] = [st.session_state["selected_cities"][0], new_city]
                    st.rerun()

    # KPI narrative
    df_primary_fb = city_data.get(primary_city, {}).get("fb", pd.DataFrame())
    if not df_primary_fb.empty and len(df_primary_fb) >= 2:
        latest_percent = float(df_primary_fb["foreign_born_percent"].iloc[-1])
        start_val = float(df_primary_fb["foreign_born_percent"].iloc[0])
        growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else 0

        col_kpi, col_lede = st.columns([1, 2.5])
        with col_kpi:
            st.metric("Foreign-Born Base", f"{latest_percent:.1f}%")
            st.metric("Period Growth", f"{growth:.1f}%")
        with col_lede:
            trend_word = "surged" if growth > 10 else "grown" if growth > 0 else "declined"
            st.markdown(textwrap.dedent(f"""
            <div style="font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; font-size:1.05rem; line-height: 1.65; color: #333; padding: 6px 0;">
            Over the observed period, the foreign-born population in <b>{primary_city}</b> has {trend_word} by {abs(growth):.1f}%, now representing {latest_percent:.1f}% of residents.
            </div>
            """), unsafe_allow_html=True)
    else:
        st.info("Foreign-born time series not available for the selected municipality.")

# --------------------------------------------------
# SECTION 2: ECONOMIC INDICATORS
# --------------------------------------------------
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Economic Health & Poverty Status")

    col_ts1, col_ts2 = st.columns(2)

    with col_ts1:
        fig_inc = go.Figure()
        for city, data in city_data.items():
            df = data.get("income", pd.DataFrame())
            if not df.empty:
                fig_inc.add_trace(go.Scatter(
                    x=df["year"], y=df["median_income"], mode="lines", name=city,
                    hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Income: $%{y:,.0f}<extra></extra>"
                ))
        fig_inc.update_layout(
            template="plotly_white",
            height=480,
            xaxis_title="Year",
            yaxis_title="Median Household Income ($)",
            legend=dict(title="")
        )
        st.plotly_chart(fig_inc, use_container_width=True)

    with col_ts2:
        fig_pov = go.Figure()
        for city, data in city_data.items():
            df = data.get("poverty", pd.DataFrame())
            if not df.empty:
                fig_pov.add_trace(go.Scatter(
                    x=df["year"], y=df["poverty_rate"], mode="lines", name=city,
                    hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Poverty Rate: %{y:.1f}%<extra></extra>"
                ))
        fig_pov.update_layout(
            template="plotly_white",
            height=480,
            xaxis_title="Year",
            yaxis_title="Poverty Rate (%)",
            legend=dict(title="")
        )
        st.plotly_chart(fig_pov, use_container_width=True)

# --------------------------------------------------
# SECTION 3: TRAJECTORY
# --------------------------------------------------
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Structural Trajectory: Income vs. Immigration")

    fig_traj = go.Figure()
    for city, data in city_data.items():
        df = data.get("struct", pd.DataFrame())
        if not df.empty and len(df) > 1:
            fig_traj.add_trace(go.Scatter(
                x=df["foreign_born_percent"],
                y=df["median_income"],
                mode="markers",
                name=city,
                text=df["year"],
                hovertemplate="<b>%{fullData.name}</b><br>Year: %{text}<br>Foreign-Born: %{x:.1f}%<br>Income: $%{y:,.0f}<extra></extra>"
            ))

    fig_traj.update_layout(
        template="plotly_white",
        height=520,
        xaxis_title="Foreign-Born Population (%)",
        yaxis_title="Median Household Income ($)",
        legend=dict(title="Municipality")
    )
    st.plotly_chart(fig_traj, use_container_width=True)

# --------------------------------------------------
# SECTION 4: STRUCTURAL REGRESSION (simple OLS)
# --------------------------------------------------
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Structural Regression Panel")

    st.markdown(
        "<p style='color:#586069;'>OLS on the strict-overlap panel: "
        "<code>median_income ~ foreign_born_percent + poverty_rate</code> "
        "(per municipality).</p>",
        unsafe_allow_html=True
    )

    rows = []
    for city, data in city_data.items():
        df = data.get("struct", pd.DataFrame()).dropna()
        if df is None or df.empty or len(df) < 4:
            rows.append({"city": city, "n": int(len(df)) if df is not None else 0, "r2": None,
                         "b_foreign_born": None, "b_poverty": None, "intercept": None})
            continue

        y = df["median_income"].to_numpy(dtype=float)
        X = np.column_stack([
            np.ones(len(df)),
            df["foreign_born_percent"].to_numpy(dtype=float),
            df["poverty_rate"].to_numpy(dtype=float),
        ])

        # beta = (X'X)^-1 X'y
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
            rows.append({
                "city": city,
                "n": int(len(df)),
                "r2": float(r2) if r2 is not None else None,
                "intercept": float(beta[0]),
                "b_foreign_born": float(beta[1]),
                "b_poverty": float(beta[2]),
            })
        except Exception:
            rows.append({"city": city, "n": int(len(df)), "r2": None,
                         "b_foreign_born": None, "b_poverty": None, "intercept": None})

    reg_df = pd.DataFrame(rows).sort_values(["r2", "n"], ascending=[False, False])
    AgGrid(reg_df, fit_columns_on_grid_load=True)

# --------------------------------------------------
# SECTION 5: COUNTRY-OF-ORIGIN CHOROPLETH (B05006 best-effort)
# --------------------------------------------------
def label_to_country_name(variable_label: str) -> str | None:
    """
    Attempt to extract a country-ish token from the B05006 variable_label.
    We drop broad regions and non-sovereign aggregates.
    """
    if not isinstance(variable_label, str) or not variable_label.strip():
        return None

    # label examples often like: "Total | Africa | Ethiopia"
    s = variable_label.strip()

    # remove common prefixes
    s = re.sub(r"^Total\s*\|\s*", "", s)
    s = re.sub(r"^Foreign-born\s*\|\s*", "", s, flags=re.IGNORECASE)

    # split hierarchy
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return None

    last = parts[-1]

    # reject obvious aggregates
    bad = {
        "Total", "Europe", "Asia", "Africa", "Oceania", "Latin America", "Northern America",
        "Other", "Other areas", "Other Europe", "Other Asia", "Other Africa",
        "Other Oceania", "Other Latin America", "Other Northern America",
        "Other and unspecified", "Other and unspecified areas"
    }
    if last in bad:
        return None

    # many labels include "Other ..." or "Total foreign-born population"
    if "total foreign-born" in last.lower():
        return None
    if last.lower().startswith("other"):
        return None

    return last

def country_to_iso3(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except Exception:
        return None

with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Global Origins of Foreign-Born Population (Best-Effort)")

    latest_year = latest_year_for_place(primary_fips)
    if latest_year is None:
        st.warning("Could not determine latest year for this place.")
        st.stop()

    # Pull B05006 rows for the latest year by filtering via get_place_variable_trend? (variable-by-variable is expensive)
    # We rely on a convention: get_place_variable_trend supports a 3rd argument source_table OR we fallback to a small sample.
    # If your function does not support source_table filtering, this section will show a controlled message.

    try:
        # Try: if your get_place_variable_trend supports "source_table" by passing variable_id=None and source_table=...
        # If it doesn't, it will throw and we handle cleanly.

        df_b05006 = get_place_source_table_year(
            primary_fips,
            SOURCE_BIRTH_TABLE,
            latest_year
        )
    except Exception:
        df_b05006 = pd.DataFrame()

    if df_b05006 is None or df_b05006.empty:
        st.info(
            "Country-of-origin choropleth requires a source_table-capable query. "
            "If your queries.py doesn't support it yet, add a function to fetch "
            "all B05006 rows for (place_fips, year)."
        )
    else:
        # Expect cols: variable_label, estimate
        df_b05006 = df_b05006.copy()
        df_b05006["estimate"] = pd.to_numeric(df_b05006.get("estimate"), errors="coerce")
        df_b05006 = df_b05006.dropna(subset=["estimate"])

        df_b05006["country_name"] = df_b05006["variable_label"].apply(label_to_country_name)
        df_b05006 = df_b05006.dropna(subset=["country_name"])

        df_b05006["iso3"] = df_b05006["country_name"].apply(country_to_iso3)
        df_map = df_b05006.dropna(subset=["iso3"]).groupby(["iso3"], as_index=False)["estimate"].sum()

        if df_map.empty:
            st.warning("No mappable sovereign countries found for this municipality/year.")
        else:
            fig_world = px.choropleth(
                df_map,
                locations="iso3",
                color="estimate",
                title=f"Foreign-Born Population by Country of Birth (approx.) — {primary_city} ({latest_year})",
                labels={"estimate": "Foreign-born (estimate)"},
            )
            fig_world.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_world, use_container_width=True)

            st.markdown("#### Top Mappable Origins")
            top = df_b05006.dropna(subset=["iso3"]).groupby(["country_name"], as_index=False)["estimate"].sum()
            top = top.sort_values("estimate", ascending=False).head(20)
            AgGrid(top, fit_columns_on_grid_load=True)

# --------------------------------------------------
# SECTION 6: TABLES (primary city)
# --------------------------------------------------
with st.container():
    st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
    st.markdown("### Our Data")

    st.markdown(f"#### Foreign-Born Percent — {primary_city.split(',')[0]}")
    AgGrid(city_data[primary_city]["fb"], fit_columns_on_grid_load=True)

    st.markdown(f"#### Median Income — {primary_city.split(',')[0]}")
    AgGrid(city_data[primary_city]["income"], fit_columns_on_grid_load=True)

    st.markdown(f"#### Poverty Rate — {primary_city.split(',')[0]}")
    AgGrid(city_data[primary_city]["poverty"], fit_columns_on_grid_load=True)

    st.markdown(f"#### Strict Overlap Panel (FB% + Income + Poverty) — {primary_city.split(',')[0]}")
    AgGrid(city_data[primary_city]["struct"], fit_columns_on_grid_load=True)