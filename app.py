from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import streamlit as st

from src.queries import (
    # registry / map
    get_cities,
    get_gateway_fips,
    # raw table (origins)
    get_place_source_table_year,
    # journalist layer
    get_metric_catalog,
    get_latest_year_available,  # (ok if unused)
    get_gateway_metric_snapshot,
    get_gateway_metric_trend,
    get_state_metric_trend,
    get_gateway_ranking,
    get_gateway_scatter,
    get_available_gateway_years,
)
from src.story_angles import STORY_ANGLES

# ==================================================
# CONFIG / CONSTANTS
# ==================================================

GATEWAY_ABBREVIATIONS = {
    "Pittsfield city, Massachusetts": "PIT",
    "Westfield city, Massachusetts": "WES",
    "Holyoke city, Massachusetts": "HLY",
    "Chicopee city, Massachusetts": "CHI",
    "Springfield city, Massachusetts": "SPR",
    "Fitchburg city, Massachusetts": "FIT",
    "Leominster city, Massachusetts": "LEO",
    "Worcester city, Massachusetts": "WOR",
    "Lowell city, Massachusetts": "LOW",
    "Methuen city, Massachusetts": "MET",
    "Lawrence city, Massachusetts": "LAW",
    "Haverhill city, Massachusetts": "HAV",
    "Peabody city, Massachusetts": "PEA",
    "Salem city, Massachusetts": "SAL",
    "Lynn city, Massachusetts": "LYN",
    "Revere city, Massachusetts": "REV",
    "Malden city, Massachusetts": "MAL",
    "Everett city, Massachusetts": "EVE",
    "Chelsea city, Massachusetts": "CHE",
    "Quincy city, Massachusetts": "QUI",
    "Brockton city, Massachusetts": "BRO",
    "Attleboro city, Massachusetts": "ATT",
    "Taunton city, Massachusetts": "TAU",
    "Fall River city, Massachusetts": "FAL",
    "New Bedford city, Massachusetts": "NB",
    "Barnstable Town, Massachusetts": "BAR",
}

COLOR_TARGET = "#4A86C5"
COLOR_BASE = "#F28E8E"
COLOR_BG = "#f4f5f6"
COLOR_TEXT = "#2c2f33"
COLOR_BOSTON = "#5FB3A8"

# Which view modes to support
MODES = ["Public", "Investigative", "Academic", "Executive"]

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# CSS
# ==================================================
def load_css() -> None:
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    st.markdown(
        """
        <style>
        .section-card-marker { display:none; }

        div[data-testid="stVerticalBlock"]:has(.section-card-marker) {
            background: #ffffff;
            padding: 26px;
            border-radius: 6px;
            border: 1px solid #e1e4e8;
            margin-bottom: 22px;
        }

        .hero h1 { margin-bottom: 6px; }
        .accent-line { height: 4px; width: 90px; background: #4A86C5; margin: 8px 0 14px 0; border-radius: 2px; }

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

        .pill {
            display:inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid #e1e4e8;
            background: #f6f8fa;
            font-size: 0.85rem;
            color: #24292f;
            margin-right: 8px;
            margin-top: 6px;
        }

        .subtle { color:#6b7280; font-size:0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()

# ==================================================
# HELPERS
# ==================================================

def normalize_geo_key(name: str) -> str:
    s = str(name)
    s = s.replace(", Massachusetts", "")
    s = re.sub(r"\b(city|town|cdp)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.strip().upper()


def clean_place_label(name: str) -> str:
    s = str(name).replace(", Massachusetts", "").strip()
    s = re.sub(r"\b(city|town)\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


NORMALIZED_ABBR = {
    normalize_geo_key(clean_place_label(k)): v for k, v in GATEWAY_ABBREVIATIONS.items()
}


def fmt_value(v: float, meta: Dict) -> str:
    if pd.isna(v):
        return "—"
    hint = (meta or {}).get("format_hint", "")
    if hint == "percent":
        return f"{float(v):.1f}%"
    if hint == "dollars":
        return f"${float(v):,.0f}"
    try:
        vv = float(v)
        if abs(vv) >= 1000:
            return f"{vv:,.0f}"
        return f"{vv:,.2f}"
    except Exception:
        return str(v)


def fmt_delta(d: float, meta: Dict) -> str:
    if pd.isna(d):
        return ""
    hint = (meta or {}).get("format_hint", "")
    if hint == "percent":
        return f"{float(d):+.1f} pts (5yr)"
    if hint == "dollars":
        return f"{float(d):+,.0f} (5yr)"
    try:
        dd = float(d)
        if abs(dd) >= 1000:
            return f"{dd:+,.0f} (5yr)"
        return f"{dd:+.2f} (5yr)"
    except Exception:
        return ""


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def first_existing(keys: List[str], catalog: Dict[str, Dict]) -> Optional[str]:
    for k in keys:
        if k in catalog:
            return k
    return None


# ==================================================
# ANALYTICS LAYER (in-app)
# ==================================================

@dataclass
class DistributionContext:
    value: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    z: Optional[float]
    percentile: Optional[float]  # 0-100
    rank: Optional[int]
    n: Optional[int]


def compute_distribution_context(rank_df: pd.DataFrame, place_fips: str) -> DistributionContext:
    """
    rank_df expected to include at least: place_fips and a value column.
    We'll try common names: value, x, metric_value.
    """
    if rank_df is None or rank_df.empty:
        return DistributionContext(None, None, None, None, None, None, None, None)

    df = rank_df.copy()
    df["place_fips"] = df["place_fips"].astype(str)

    value_col = None
    for cand in ["value", "metric_value", "x"]:
        if cand in df.columns:
            value_col = cand
            break

    if value_col is None:
        # best-effort: take the first numeric column (excluding ranks)
        numeric_cols = []
        for c in df.columns:
            if c.lower().startswith("rank"):
                continue
            if c in ["place_fips", "place_name"]:
                continue
            if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
                numeric_cols.append(c)
        value_col = numeric_cols[0] if numeric_cols else None

    if value_col is None:
        return DistributionContext(None, None, None, None, None, None, None, None)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return DistributionContext(None, None, None, None, None, None, None, None)

    v = df.loc[df["place_fips"] == str(place_fips), value_col]
    v = safe_float(v.iloc[0]) if not v.empty else None

    mean = safe_float(df[value_col].mean())
    median = safe_float(df[value_col].median())
    std = safe_float(df[value_col].std(ddof=0))

    z = None
    if v is not None and std not in (None, 0.0):
        z = (v - mean) / std

    # percentile (empirical)
    percentile = None
    if v is not None:
        percentile = float((df[value_col] <= v).mean() * 100)

    # rank (if present)
    rank = None
    for rc in ["rank_within_gateway", "rank", "gateway_rank"]:
        if rc in df.columns:
            rr = df.loc[df["place_fips"] == str(place_fips), rc]
            if not rr.empty:
                try:
                    rank = int(rr.iloc[0])
                except Exception:
                    rank = None
            break

    return DistributionContext(
        value=v,
        mean=mean,
        median=median,
        std=std,
        z=z,
        percentile=percentile,
        rank=rank,
        n=int(len(df)),
    )


@dataclass
class TrendDiagnostics:
    slope_5yr: Optional[float]
    slope_10yr: Optional[float]
    delta_5yr: Optional[float]
    last: Optional[float]
    first: Optional[float]


def compute_trend_diagnostics(trend_df: pd.DataFrame) -> TrendDiagnostics:
    """
    trend_df expected: year, value
    """
    if trend_df is None or trend_df.empty:
        return TrendDiagnostics(None, None, None, None, None)

    df = trend_df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).sort_values("year")
    if len(df) < 2:
        return TrendDiagnostics(None, None, None, safe_float(df["value"].iloc[-1]) if len(df) else None, None)

    years = df["year"].to_numpy()
    vals = df["value"].to_numpy()

    def slope_over(window_years: int) -> Optional[float]:
        if len(df) < 2:
            return None
        y_max = years.max()
        mask = years >= (y_max - window_years)
        suby = years[mask]
        subv = vals[mask]
        if len(suby) < 2:
            return None
        # linear slope per year
        try:
            m = np.polyfit(suby, subv, 1)[0]
            return float(m)
        except Exception:
            return None

    slope_5 = slope_over(5)
    slope_10 = slope_over(10)

    # delta_5yr using closest year >= max-5
    delta_5 = None
    try:
        y_max = int(years.max())
        target = y_max - 5
        # choose nearest year to target
        idx0 = int(np.argmin(np.abs(years - target)))
        delta_5 = float(vals[-1] - vals[idx0])
    except Exception:
        delta_5 = None

    return TrendDiagnostics(
        slope_5yr=slope_5,
        slope_10yr=slope_10,
        delta_5yr=delta_5,
        last=safe_float(vals[-1]),
        first=safe_float(vals[0]),
    )


@dataclass
class ScatterStats:
    r: Optional[float]
    r2: Optional[float]
    slope: Optional[float]
    intercept: Optional[float]
    n: Optional[int]


def compute_scatter_stats(sc_df: pd.DataFrame) -> ScatterStats:
    if sc_df is None or sc_df.empty or "x" not in sc_df.columns or "y" not in sc_df.columns:
        return ScatterStats(None, None, None, None, None)

    df = sc_df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if len(df) < 3:
        return ScatterStats(None, None, None, None, int(len(df)))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    try:
        r = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        r = None
    r2 = float(r * r) if r is not None else None

    slope = intercept = None
    try:
        slope, intercept = np.polyfit(x, y, 1)
        slope, intercept = float(slope), float(intercept)
    except Exception:
        pass

    return ScatterStats(r=r, r2=r2, slope=slope, intercept=intercept, n=int(len(df)))


def compute_composite_index(
    place_fips: str,
    year: int,
    catalog: Dict[str, Dict],
    metric_keys: List[str],
    z_lookup: Dict[str, float],
) -> Optional[float]:
    """
    Composite index computed from z-scores for selected metrics.
    metric_keys: list of metric_keys to include.
    z_lookup: metric_key -> z-score for this place/year (precomputed)
    """
    zs = []
    for mk in metric_keys:
        z = z_lookup.get(mk)
        if z is None or pd.isna(z):
            continue
        zs.append(float(z))
    if not zs:
        return None
    return float(np.mean(zs))


def build_narrative_summary(
    city_name: str,
    year: int,
    catalog: Dict[str, Dict],
    place_fips: str,
    focus_metrics: List[str],
) -> List[str]:
    """
    Generate short, newsroom-friendly bullets using snapshot deltas + trend slopes.
    """
    bullets: List[str] = []
    for mk in focus_metrics:
        meta = catalog.get(mk, {"metric_label": mk})
        snap = get_gateway_metric_snapshot(place_fips, mk, year)
        tr = get_gateway_metric_trend(place_fips, mk)

        label = meta.get("metric_label", mk)

        delta = None
        val = None
        if snap is not None and not snap.empty:
            val = safe_float(snap.get("value", pd.Series([None])).iloc[0])
            delta = safe_float(snap.get("delta_5yr", pd.Series([None])).iloc[0])

        diag = compute_trend_diagnostics(tr) if tr is not None else TrendDiagnostics(None, None, None, None, None)

        if val is None:
            continue

        # Compose bullet with available pieces
        if delta is not None:
            bullets.append(f"**{label}:** {fmt_value(val, meta)} ({fmt_delta(delta, meta)}).")
        else:
            bullets.append(f"**{label}:** {fmt_value(val, meta)}.")

        # Add slope note when meaningful
        if diag.slope_10yr is not None and abs(diag.slope_10yr) > 0:
            # Keep it plain-language; no heavy stats in Public/Executive
            bullets.append(f"<span class='subtle'>Trend signal: ~{diag.slope_10yr:+.3g} per year (10yr linear slope).</span>")

    # de-duplicate subtle lines if too many
    return bullets[:10]


# ==================================================
# GEOJSON
# ==================================================
@st.cache_data
def load_ma_map() -> dict:
    with open("data/ma_municipalities.geojson") as f:
        return json.load(f)


ma_geo = load_ma_map()


def get_geo_bounds(geojson: dict) -> Tuple[float, float, float, float]:
    lats: List[float] = []
    lons: List[float] = []

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

# ==================================================
# REGISTRY + CATALOG
# ==================================================
cities_all = get_cities(gateway_only=False).copy()
cities_all["place_fips"] = cities_all["place_fips"].astype(str)

gateway_fips = set(get_gateway_fips()["place_fips"].astype(str))

EXTRA_CITY_NAMES = {"Boston city, Massachusetts", "Cambridge city, Massachusetts"}
extra_cities = cities_all[cities_all["place_name"].isin(list(EXTRA_CITY_NAMES))].copy()
extra_fips = set(extra_cities["place_fips"].astype(str))

gateway_city_options = (
    cities_all[cities_all["place_fips"].isin(gateway_fips)]["place_name"].sort_values().tolist()
)
if not gateway_city_options:
    st.error("No gateway cities found in gateway_cities table. Check is_gateway_city flags.")
    st.stop()

catalog_df = get_metric_catalog()
catalog: Dict[str, Dict] = (
    catalog_df.set_index("metric_key").to_dict(orient="index") if not catalog_df.empty else {}
)
if not catalog:
    st.error("Metric catalog is empty. Verify public.metric_catalog.")
    st.stop()

available_years = get_available_gateway_years()
if not available_years:
    st.error("No data found in public.gateway_metrics.")
    st.stop()

min_year = min(available_years)
max_year = max(available_years)

# Single source of truth for year + city
if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = max_year
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = gateway_city_options[0]
if "mode" not in st.session_state:
    st.session_state["mode"] = "Investigative"

primary_city = st.session_state["selected_city"]
primary_fips = str(cities_all.loc[cities_all["place_name"] == primary_city, "place_fips"].iloc[0])

# ==================================================
# HEADER + MODE
# ==================================================
st.markdown(
    """
    <section class="hero">
        <h1>Gateway Cities Dashboard</h1>
        <div class="accent-line"></div>
        <p>
            Story-first exploration of demographic change, economic stress, and housing pressure across Massachusetts Gateway Cities.
            Source: ACS 5-year estimates (yearly endpoints; latest available in warehouse).
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# Top control strip
cA, cB, cC = st.columns([2.2, 1.2, 1.0])
with cA:
    st.session_state["mode"] = st.radio(
        "View mode",
        MODES,
        horizontal=True,
        index=MODES.index(st.session_state["mode"]) if st.session_state["mode"] in MODES else 1,
        key="mode_selector",
    )
with cB:
    # year selector global
    st.session_state["selected_year"] = st.selectbox(
        "Analysis year",
        options=available_years,
        index=available_years.index(st.session_state["selected_year"]),
        key="global_year",
    )
with cC:
    st.markdown(
        f"<div style='text-align:right;'><span class='pill'>Data range: <b>{min_year}–{max_year}</b></span></div>",
        unsafe_allow_html=True,
    )

MODE = st.session_state["mode"]
selected_year = int(st.session_state["selected_year"])

# Keep city consistent
primary_city = st.session_state["selected_city"]
primary_fips = str(cities_all.loc[cities_all["place_name"] == primary_city, "place_fips"].iloc[0])

# ==================================================
# TABS (Methodology shown for Academic; others always)
# ==================================================
tabs = ["Map", "Investigative Themes", "Compare Metrics", "Origins (B05006)"]
if MODE == "Academic":
    tabs.append("Methodology")

tab_objs = st.tabs(tabs)
tab_map = tab_objs[0]
tab_story = tab_objs[1]
tab_compare = tab_objs[2]
tab_origins = tab_objs[3]
tab_method = tab_objs[4] if MODE == "Academic" else None

# ==================================================
# TAB 1: MAP
# ==================================================
with tab_map:
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Geographic Context")

        town_fips_map = {
            normalize_geo_key(name): fips
            for name, fips in zip(cities_all["place_name"], cities_all["place_fips"])
        }
        allowed_gateway_names = set(
            normalize_geo_key(n)
            for n in cities_all[cities_all["place_fips"].isin(gateway_fips)]["place_name"]
        )

        boston_cambridge_names = set(
            normalize_geo_key(clean_place_label(n))
            for n in extra_cities["place_name"]
        )

        locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]

        # Centroids for labeling
        town_centroids = {}
        for feature in ma_geo["features"]:
            town_raw = feature["properties"]["TOWN"]
            town_key = normalize_geo_key(town_raw)
            coords = feature["geometry"]["coordinates"]

            lats = []
            lons = []

            def extract(c):
                if isinstance(c[0], list):
                    for sub in c:
                        extract(sub)
                else:
                    lons.append(c[0])
                    lats.append(c[1])

            extract(coords)
            if lats and lons:
                town_centroids[town_key] = (sum(lats) / len(lats), sum(lons) / len(lons))

        @st.cache_data
        def build_map(
            geojson: dict,
            locations: List[str],
            allowed_gateway_names: set,
            town_fips_map_local: Dict[str, str],
            selected_fips: str,
            c_lat: float,
            c_lon: float,
            boston_cambridge_names: set,
        ) -> go.Figure:
            z_values = []
            for town_name in locations:
                town_norm = normalize_geo_key(town_name)
                if town_norm in town_fips_map_local and town_fips_map_local[town_norm] == selected_fips:
                    z_values.append(3)
                elif town_norm in boston_cambridge_names:
                    z_values.append(2)
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
                    [0.5, COLOR_BASE],
                    [0.5, COLOR_BOSTON],
                    [0.75, COLOR_BOSTON],
                    [0.75, COLOR_TARGET],
                    [1.0, COLOR_TARGET],
                ],
                zmin=0,
                zmax=3,
                showscale=False,
                marker_line_width=1.0,
                marker_line_color="rgba(60,60,60,0.6)",
                hovertemplate="<b>%{location}</b><extra></extra>",
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=1.0)),
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
            boston_cambridge_names,
        )

        # Gateway abbreviation labels
        LABEL_OFFSETS = {
            "LEOMINSTER": (-0.020, 0.020),
            "METHUEN": (0.015, 0.012),
            "MALDEN": (0.010, -0.010),
            "CHELSEA": (-0.005, 0.02),
            "FITCHBURG": (0.010, -0.010),
            "REVERE": (0.010, 0.010),
            "EVERETT": (-0.005, -0.020),
        }

        label_lats, label_lons, label_text = [], [], []
        for town_name in locations:
            town_key = normalize_geo_key(town_name)
            if town_key in allowed_gateway_names and town_key in town_centroids:
                fips = town_fips_map[town_key]
                full_name = cities_all[cities_all["place_fips"] == fips]["place_name"].iloc[0]
                city_key = normalize_geo_key(clean_place_label(full_name))
                abbr = NORMALIZED_ABBR.get(city_key)
                if not abbr:
                    continue
                lat, lon = town_centroids[town_key]
                off_lat, off_lon = LABEL_OFFSETS.get(town_key, (0.0, 0.0))
                lat += off_lat
                lon += off_lon
                label_lats.append(lat)
                label_lons.append(lon)
                label_text.append(abbr)

        fig_map.add_trace(
            go.Scattermapbox(
                lat=label_lats,
                lon=label_lons,
                mode="text",
                text=label_text,
                textfont=dict(
                    size=11,
                    color="black"
                ),
                textposition="middle center",
                hoverinfo="skip",
                showlegend=False,
                below=""
            )
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True,
            on_select="rerun",
            key="map_select",
        )

        st.markdown(
            f"""
            <div class="map-legend">
                <div class="legend-item"><span class="dot" style="background:{COLOR_TARGET};"></span>Selected Municipality</div>
                <div class="legend-item"><span class="dot" style="background:{COLOR_BOSTON};"></span>Boston & Cambridge</div>
                <div class="legend-item"><span class="dot" style="background:{COLOR_BASE};"></span>Gateway Cities</div>
                <div class="legend-item"><span class="dot" style="background:#E9ECEF; border:1px solid #ccc;"></span>Other Municipalities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Click-to-select (Streamlit plotly selection)
        selection = st.session_state.get("map_select")
        if selection and selection.get("selection"):
            points = selection["selection"].get("points", [])
            loc_points = [p for p in points if p.get("location")]
            if loc_points:
                clicked_town = loc_points[-1]["location"]
                town_key = normalize_geo_key(clicked_town)
                new_fips = town_fips_map.get(town_key)
                if new_fips and str(new_fips) in gateway_fips:
                    new_city = cities_all.loc[cities_all["place_fips"] == str(new_fips), "place_name"].iloc[0]
                    if st.session_state.get("selected_city") != new_city:
                        st.session_state["selected_city"] = new_city
                        st.session_state["map_select"]["selection"]["points"] = []
                        st.rerun()

        # ==================================================
        # CITY INTELLIGENCE PANEL
        # ==================================================
        st.divider()
        st.markdown(f"# {primary_city.split(',')[0]} — City Profile")

        # Executive Summary (Executive + Public + Investigative; Academic can see too)
        if MODE in ("Public", "Executive", "Investigative", "Academic"):
            st.markdown("## Executive Summary")
            focus_for_summary = [
                first_existing(["median_income"], catalog),
                first_existing(["poverty_rate"], catalog),
                first_existing(["rent_burden_30_plus"], catalog),
                first_existing(["median_home_value"], catalog),
                first_existing(["foreign_born_share"], catalog),
                first_existing(["total_population"], catalog),
            ]
            focus_for_summary = [x for x in focus_for_summary if x]

            bullets = build_narrative_summary(primary_city.split(",")[0], selected_year, catalog, primary_fips, focus_for_summary)

            if MODE in ("Public", "Executive"):
                # strip slope lines (subtle) for simpler audiences
                bullets = [b for b in bullets if "Trend signal" not in b]
            if not bullets:
                st.info("No summary available (missing snapshot data for this year).")
            else:
                for b in bullets[:6]:
                    st.markdown(f"- {b}", unsafe_allow_html=True)

        # Snapshot grid (curated by mode)
        st.divider()
        st.markdown(f"## Snapshot {selected_year}")

        CORE_METRICS = [
            "total_population",
            "foreign_born_share",
            "median_income",
            "poverty_rate",
            "rent_burden_30_plus",
            "median_home_value",
            "gini_index",
        ]

        # Public mode: keep it tight
        if MODE == "Public":
            CORE_METRICS = [k for k in CORE_METRICS if k in ("total_population", "median_income", "poverty_rate", "rent_burden_30_plus")]
        # Executive mode: include housing + economy
        if MODE == "Executive":
            CORE_METRICS = [k for k in CORE_METRICS if k in ("median_income", "poverty_rate", "rent_burden_30_plus", "median_home_value", "total_population")]

        cols = st.columns(4)
        for i, mk in enumerate(CORE_METRICS):
            meta = catalog.get(mk, {"metric_label": mk})
            snap = get_gateway_metric_snapshot(primary_fips, mk, selected_year)
            if snap is None or snap.empty:
                cols[i % 4].metric(meta.get("metric_label", mk), "—")
                continue
            value = snap.get("value", pd.Series([None])).iloc[0]
            delta = snap.get("delta_5yr", pd.Series([None])).iloc[0]
            cols[i % 4].metric(meta.get("metric_label", mk), fmt_value(value, meta), fmt_delta(delta, meta))

        # Structural Trend Panel (all but optional simplified in Public)
        st.divider()
        st.markdown("## Structural Trend")

        STRUCTURAL_METRICS = [
            k for k in ["median_income", "poverty_rate", "rent_burden_30_plus", "foreign_born_share", "total_population"]
            if k in catalog
        ]
        if MODE == "Public":
            STRUCTURAL_METRICS = [k for k in STRUCTURAL_METRICS if k in ("median_income", "poverty_rate", "rent_burden_30_plus")]

        trend_metric = st.selectbox(
            "Select trend metric",
            STRUCTURAL_METRICS,
            format_func=lambda k: catalog.get(k, {}).get("metric_label", k),
            key="map_trend_focus",
        )

        city_tr = get_gateway_metric_trend(primary_fips, trend_metric)
        state_tr = get_state_metric_trend(trend_metric)
        meta = catalog.get(trend_metric, {"metric_label": trend_metric})

        if city_tr is not None and not city_tr.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=city_tr["year"], y=city_tr["value"], mode="lines", name=primary_city.split(",")[0]))
            if state_tr is not None and not state_tr.empty:
                fig.add_trace(go.Scatter(x=state_tr["year"], y=state_tr["value"], mode="lines", name="Massachusetts"))
            fig.update_layout(template="plotly_white", height=420, xaxis_title="Year", yaxis_title=meta.get("unit", ""))
            st.plotly_chart(fig, use_container_width=True)

            # Diagnostics (Investigative + Academic)
            if MODE in ("Investigative", "Academic"):
                diag = compute_trend_diagnostics(city_tr)
                c1, c2, c3 = st.columns(3)
                if diag.delta_5yr is not None:
                    c1.metric("5-year change (approx.)", fmt_delta(diag.delta_5yr, meta) or f"{diag.delta_5yr:+.3g}")
                if diag.slope_5yr is not None:
                    c2.metric("Slope (5yr window)", f"{diag.slope_5yr:+.3g} / year")
                if diag.slope_10yr is not None:
                    c3.metric("Slope (10yr window)", f"{diag.slope_10yr:+.3g} / year")
        else:
            st.info("No trend data available.")

        # Gateway distribution / position (Investigative + Academic)
        st.divider()
        st.markdown(f"## Gateway Position — {meta.get('metric_label', trend_metric)} ({selected_year})")

        rank_df = get_gateway_ranking(trend_metric, selected_year)
        if rank_df is None or rank_df.empty:
            st.info("No ranking data available.")
        else:
            ctx = compute_distribution_context(rank_df, primary_fips)
            c1, c2, c3, c4 = st.columns(4)

            if ctx.rank is not None and ctx.n is not None:
                c1.metric("Gateway rank", f"{ctx.rank} / {ctx.n}")
                # "higher is better" is not universally true; show neutral percentile
                if ctx.percentile is not None:
                    c2.metric("Percentile (empirical)", f"{ctx.percentile:.0f}th")
            if MODE in ("Investigative", "Academic") and ctx.z is not None:
                c3.metric("Z-score vs gateway", f"{ctx.z:+.2f}")
            if MODE in ("Investigative", "Academic") and ctx.mean is not None and ctx.value is not None:
                gap = ctx.value - ctx.mean
                c4.metric("Gap vs gateway mean", f"{gap:+.3g}")

            if MODE in ("Investigative", "Academic"):
                st.dataframe(rank_df, use_container_width=True, hide_index=True)

        # Composite Indices (Executive + Investigative + Academic; optional in Public)
        st.divider()
        st.markdown("## Composite Indices")

        # Define index recipes (only include metrics that exist in catalog)
        housing_keys = [k for k in [
            "rent_burden_30_plus",  # +
            "median_home_value",    # +
            "renter_share",         # + (if you have it)
            "vacancy_rate",         # - (we'll invert via sign below if present)
            "price_to_income_ratio" # + (if you have it)
        ] if k in catalog]

        econ_keys = [k for k in [
            "poverty_rate",         # +
            "gini_index",           # +
            "median_income"         # - (we'll invert via sign below)
        ] if k in catalog]

        demo_keys = [k for k in [
            "foreign_born_share",   # +
            "total_population"      # +
        ] if k in catalog]

        # For z-scores we need rank_dfs per metric (gateway distribution per metric)
        def z_for_metric(mk: str) -> Optional[float]:
            rdf = get_gateway_ranking(mk, selected_year)
            if rdf is None or rdf.empty:
                return None
            ctxm = compute_distribution_context(rdf, primary_fips)
            return ctxm.z

        z_lookup = {mk: z_for_metric(mk) for mk in set(housing_keys + econ_keys + demo_keys)}

        # Inversions: for some metrics, "higher" means less stress
        invert = set([k for k in ["vacancy_rate", "median_income"] if k in catalog])

        for mk in invert:
            if z_lookup.get(mk) is not None:
                z_lookup[mk] = -1.0 * z_lookup[mk]

        housing_score = compute_composite_index(primary_fips, selected_year, catalog, housing_keys, z_lookup) if housing_keys else None
        econ_score = compute_composite_index(primary_fips, selected_year, catalog, econ_keys, z_lookup) if econ_keys else None
        demo_score = compute_composite_index(primary_fips, selected_year, catalog, demo_keys, z_lookup) if demo_keys else None

        # Public: show only Housing Stress if available
        show_housing_only = (MODE == "Public")

        c1, c2, c3 = st.columns(3)
        if housing_score is not None:
            c1.metric("Housing Stress (z-avg)", f"{housing_score:+.2f}")
            if MODE in ("Investigative", "Academic"):
                c1.caption("Avg z-score across housing metrics (inverted where appropriate).")

        if not show_housing_only:
            if econ_score is not None:
                c2.metric("Economic Fragility (z-avg)", f"{econ_score:+.2f}")
            if demo_score is not None:
                c3.metric("Demographic Change (z-avg)", f"{demo_score:+.2f}")

        if MODE in ("Investigative", "Academic"):
            st.caption("Interpretation: positive values indicate above-gateway-average ‘stress’ or ‘intensity’ along the index definition.")

# ==================================================
# TAB 2: INVESTIGATIVE THEMES
# ==================================================
with tab_story:
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Investigative Themes")

        cities_df = get_cities(gateway_only=True)
        story_city = st.selectbox(
            "Gateway City",
            cities_df["place_name"].tolist(),
            index=max(0, cities_df["place_name"].tolist().index(primary_city)) if primary_city in cities_df["place_name"].tolist() else 0,
            key="story_city",
        )
        place_fips = str(cities_df.loc[cities_df["place_name"] == story_city, "place_fips"].iloc[0])

        angle_key = st.selectbox(
            "Theme",
            list(STORY_ANGLES.keys()),
            format_func=lambda k: STORY_ANGLES[k]["title"],
            key="angle_key",
        )
        angle = STORY_ANGLES[angle_key]

        investigate = st.toggle("🔍 Explore Relationships", value=(MODE in ("Investigative", "Academic")), key="investigate_toggle")

        st.subheader(angle["title"])

        metrics = angle.get("metrics", [])
        if not metrics:
            st.info("No metrics configured for this theme.")
        else:
            # Public: limit to first 4 to reduce cognitive load
            show_metrics = metrics[:4] if MODE == "Public" else metrics

            cols = st.columns(min(len(show_metrics), 6))
            for i, mk in enumerate(show_metrics):
                meta = catalog.get(mk, {"metric_label": mk, "format_hint": "number"})
                snap = get_gateway_metric_snapshot(place_fips, mk, selected_year)
                if snap is None or snap.empty:
                    cols[i % len(cols)].metric(meta.get("metric_label", mk), "—")
                    continue

                v = snap["value"].iloc[0]
                d5 = snap.get("delta_5yr", pd.Series([None])).iloc[0]
                main = fmt_value(v, meta)
                delta = fmt_delta(d5, meta)
                cols[i % len(cols)].metric(meta.get("metric_label", mk), main, delta)

        st.divider()

        lead_metric = st.selectbox(
            "Trend focus metric",
            metrics if metrics else list(catalog.keys()),
            format_func=lambda k: catalog.get(k, {"metric_label": k}).get("metric_label", k),
            key="lead_metric",
        )

        city_trend = get_gateway_metric_trend(place_fips, lead_metric).rename(columns={"value": "City"})
        ma_trend = get_state_metric_trend(lead_metric).rename(columns={"value": "Massachusetts"})
        trend = city_trend.merge(ma_trend, on="year", how="left")

        meta = catalog.get(lead_metric, {"metric_label": lead_metric})
        st.markdown(f"**Trend:** {meta.get('metric_label', lead_metric)}")

        if trend is None or trend.empty:
            st.info("No trend data available for this metric/city.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["year"], y=trend["City"], mode="lines", name=story_city.split(",")[0]))
            if "Massachusetts" in trend.columns:
                fig.add_trace(go.Scatter(x=trend["year"], y=trend["Massachusetts"], mode="lines", name="Massachusetts"))
            fig.update_layout(template="plotly_white", height=430, xaxis_title="Year", yaxis_title=meta.get("unit", ""), legend=dict(title=""))
            st.plotly_chart(fig, use_container_width=True)

            if MODE in ("Investigative", "Academic"):
                diag = compute_trend_diagnostics(city_trend.rename(columns={"City": "value"})[["year", "City"]].rename(columns={"City": "value"}))
                c1, c2, c3 = st.columns(3)
                if diag.delta_5yr is not None:
                    c1.metric("5-year change (approx.)", fmt_delta(diag.delta_5yr, meta) or f"{diag.delta_5yr:+.3g}")
                if diag.slope_5yr is not None:
                    c2.metric("Slope (5yr window)", f"{diag.slope_5yr:+.3g} / year")
                if diag.slope_10yr is not None:
                    c3.metric("Slope (10yr window)", f"{diag.slope_10yr:+.3g} / year")

        # Ranking table (Investigative + Academic)
        st.divider()
        st.markdown(f"### Gateway Ranking — {meta.get('metric_label')} ({selected_year})")

        rank_df = get_gateway_ranking(lead_metric, selected_year)
        if rank_df is None or rank_df.empty:
            st.info("No ranking data available for this metric/year.")
        else:
            if MODE in ("Investigative", "Academic"):
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
            else:
                # Public/Executive: only show selected city line
                sub = rank_df.copy()
                sub["place_fips"] = sub["place_fips"].astype(str)
                one = sub[sub["place_fips"] == str(place_fips)]
                if not one.empty:
                    st.dataframe(one, use_container_width=True, hide_index=True)
                else:
                    st.info("Selected city not found in ranking output for this year.")

        # Investigative comparisons
        if investigate and MODE in ("Investigative", "Academic"):
            st.divider()
            st.subheader("Investigative comparisons (Gateway Cities only)")

            pairs = angle.get("investigative_pairs", [])
            if not pairs:
                st.info("No investigative pairs configured for this theme.")
            else:
                pair_labels = []
                for xk, yk in pairs:
                    xl = catalog.get(xk, {"metric_label": xk}).get("metric_label", xk)
                    yl = catalog.get(yk, {"metric_label": yk}).get("metric_label", yk)
                    pair_labels.append(f"{xl} vs {yl}")

                idx = st.selectbox("Choose a comparison", range(len(pairs)), format_func=lambda i: pair_labels[i], key="pair_idx")
                xk, yk = pairs[idx]

                sc = get_gateway_scatter(xk, yk, selected_year)
                xl = catalog.get(xk, {"metric_label": xk}).get("metric_label", xk)
                yl = catalog.get(yk, {"metric_label": yk}).get("metric_label", yk)

                if sc is None or sc.empty:
                    st.info("No scatter data available for this comparison.")
                else:
                    sc = sc.copy()
                    sc["is_selected"] = sc["place_fips"].astype(str) == str(place_fips)

                    stats = compute_scatter_stats(sc)

                    fig_sc = px.scatter(
                        sc,
                        x="x",
                        y="y",
                        hover_name="place_name",
                        title=f"{selected_year}: {xl} (x) vs {yl} (y)",
                    )
                    # regression line (Academic only)
                    if MODE == "Academic" and stats.slope is not None and stats.intercept is not None:
                        xx = np.linspace(np.nanmin(sc["x"]), np.nanmax(sc["x"]), 50)
                        yy = stats.slope * xx + stats.intercept
                        fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="OLS fit"))

                    # highlight selected city
                    fig_sc.add_trace(
                        go.Scatter(
                            x=sc.loc[sc["is_selected"], "x"],
                            y=sc.loc[sc["is_selected"], "y"],
                            mode="markers",
                            name=story_city.split(",")[0],
                            marker=dict(size=16, symbol="diamond"),
                            hovertemplate=f"<b>{story_city}</b><br>{xl}: %{{x}}<br>{yl}: %{{y}}<extra></extra>",
                        )
                    )
                    fig_sc.update_layout(template="plotly_white", height=520)
                    st.plotly_chart(fig_sc, use_container_width=True)

                    if MODE in ("Investigative", "Academic"):
                        c1, c2, c3 = st.columns(3)
                        if stats.r is not None:
                            c1.metric("Correlation r", f"{stats.r:+.2f}")
                        if stats.r2 is not None:
                            c2.metric("R²", f"{stats.r2:.2f}")
                        if stats.n is not None:
                            c3.metric("N (cities)", f"{stats.n}")

# ==================================================
# TAB 3: COMPARE METRICS
# ==================================================
with tab_compare:
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Compare Metrics")

        compare_cities = st.multiselect(
            "Select Gateway Cities (Max 3)",
            options=gateway_city_options,
            default=[primary_city],
            max_selections=3,
            key="compare_cities",
        )
        if not compare_cities:
            compare_cities = [primary_city]

        selected_fips = {
            city: str(cities_all.loc[cities_all["place_name"] == city, "place_fips"].iloc[0])
            for city in compare_cities
        }

        metric_keys = sorted(list(catalog.keys()))
        metric_labels = {k: catalog[k].get("metric_label", k) for k in metric_keys}

        col_a, col_b = st.columns([1.2, 2.3])

        with col_a:
            metric_to_compare = st.selectbox(
                "Trend metric",
                metric_keys,
                index=metric_keys.index("median_income") if "median_income" in metric_keys else 0,
                format_func=lambda k: metric_labels.get(k, k),
                key="compare_metric",
            )

        with col_b:
            st.markdown("**Multi-city trend**")
            fig = go.Figure()
            for city_name, fips in selected_fips.items():
                tr = get_gateway_metric_trend(fips, metric_to_compare)
                if tr is None or tr.empty:
                    continue
                fig.add_trace(go.Scatter(x=tr["year"], y=tr["value"], mode="lines", name=city_name.split(",")[0]))
            fig.update_layout(
                template="plotly_white",
                height=450,
                xaxis_title="Year",
                yaxis_title=catalog.get(metric_to_compare, {}).get("unit", ""),
                legend=dict(title=""),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Cross-metric scatter (Gateway Cities)")

        c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
        with c1:
            metric_x = st.selectbox(
                "X metric",
                metric_keys,
                index=metric_keys.index("rent_burden_30_plus") if "rent_burden_30_plus" in metric_keys else 0,
                format_func=lambda k: metric_labels.get(k, k),
                key="scatter_x",
            )
        with c2:
            metric_y = st.selectbox(
                "Y metric",
                metric_keys,
                index=metric_keys.index("poverty_rate") if "poverty_rate" in metric_keys else min(1, len(metric_keys) - 1),
                format_func=lambda k: metric_labels.get(k, k),
                key="scatter_y",
            )
        with c3:
            sc_year = st.selectbox("Year", available_years, index=available_years.index(selected_year), key="scatter_year")

        sc = get_gateway_scatter(metric_x, metric_y, sc_year)
        if sc is None or sc.empty:
            st.info("No data available for that scatter combination.")
        else:
            sc = sc.copy()
            sc["is_selected"] = sc["place_fips"].astype(str).isin(set(selected_fips.values()))
            xl = metric_labels.get(metric_x, metric_x)
            yl = metric_labels.get(metric_y, metric_y)

            stats = compute_scatter_stats(sc)

            fig_sc = px.scatter(sc, x="x", y="y", hover_name="place_name", title=f"{sc_year}: {xl} (x) vs {yl} (y)")

            # highlight selected cities
            sel = sc[sc["is_selected"]]
            if not sel.empty:
                fig_sc.add_trace(
                    go.Scatter(
                        x=sel["x"],
                        y=sel["y"],
                        mode="markers+text",
                        text=sel["place_name"].apply(lambda s: s.split(",")[0]),
                        textposition="top center",
                        name="Selected cities",
                        marker=dict(size=14, symbol="diamond"),
                    )
                )

            # regression line (Academic only)
            if MODE == "Academic" and stats.slope is not None and stats.intercept is not None:
                xx = np.linspace(np.nanmin(sc["x"]), np.nanmax(sc["x"]), 50)
                yy = stats.slope * xx + stats.intercept
                fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="OLS fit"))

            fig_sc.update_layout(template="plotly_white", height=560)
            st.plotly_chart(fig_sc, use_container_width=True)

            if MODE in ("Investigative", "Academic"):
                c1, c2, c3 = st.columns(3)
                if stats.r is not None:
                    c1.metric("Correlation r", f"{stats.r:+.2f}")
                if stats.r2 is not None:
                    c2.metric("R²", f"{stats.r2:.2f}")
                if stats.n is not None:
                    c3.metric("N (cities)", f"{stats.n}")

# ==================================================
# TAB 4: ORIGINS (B05006)
# ==================================================
SOURCE_BIRTH_TABLE = "B05006"

def label_to_country_name(variable_label: str) -> Optional[str]:
    if not isinstance(variable_label, str) or not variable_label.strip():
        return None

    s = variable_label.strip()
    s = re.sub(r"^Total\s*\|\s*", "", s)
    s = re.sub(r"^Foreign-born\s*\|\s*", "", s, flags=re.IGNORECASE)
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return None
    last = parts[-1]

    bad = {
        "Total",
        "Europe",
        "Asia",
        "Africa",
        "Oceania",
        "Latin America",
        "Northern America",
        "Other",
        "Other areas",
        "Other Europe",
        "Other Asia",
        "Other Africa",
        "Other Oceania",
        "Other Latin America",
        "Other Northern America",
        "Other and unspecified",
        "Other and unspecified areas",
    }
    if last in bad:
        return None
    if "total foreign-born" in last.lower():
        return None
    if last.lower().startswith("other"):
        return None

    return last

def country_to_iso3(name: str) -> Optional[str]:
    try:
        return pycountry.countries.lookup(name).alpha_3
    except Exception:
        return None

with tab_origins:
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Global Origins of Foreign-Born Population (Best-Effort)")

        st.caption(
            "This panel uses raw B05006 rows (place of birth). "
            "If your ETL produces a different naming scheme, update label parsing rules here."
        )

        cities_df = get_cities(gateway_only=True)
        origins_city = st.selectbox(
            "City for origins map",
            cities_df["place_name"].tolist(),
            index=max(0, cities_df["place_name"].tolist().index(primary_city)) if primary_city in cities_df["place_name"].tolist() else 0,
            key="origins_city",
        )
        origins_fips = str(cities_df.loc[cities_df["place_name"] == origins_city, "place_fips"].iloc[0])
        year = st.selectbox("Year", available_years, index=available_years.index(selected_year), key="origins_year")

        try:
            df_b05006 = get_place_source_table_year(origins_fips, SOURCE_BIRTH_TABLE, int(year))
        except Exception:
            df_b05006 = pd.DataFrame()

        if df_b05006 is None or df_b05006.empty:
            st.info(
                "No B05006 rows returned. Confirm that acs_place_data includes source_table='B05006' "
                "and that get_place_source_table_year is wired to the correct table/columns."
            )
        else:
            df_b05006 = df_b05006.copy()
            df_b05006["estimate"] = pd.to_numeric(df_b05006.get("estimate"), errors="coerce")
            df_b05006 = df_b05006.dropna(subset=["estimate"])

            df_b05006["country_name"] = df_b05006["variable_label"].apply(label_to_country_name)
            df_b05006 = df_b05006.dropna(subset=["country_name"])

            df_b05006["iso3"] = df_b05006["country_name"].apply(country_to_iso3)

            # Optional: convert to shares (Academic/Investigative)
            total_fb = None
            try:
                # best effort: detect total foreign-born row
                mask_total = df_b05006["variable_label"].str.contains("Total foreign-born", case=False, na=False)
                if mask_total.any():
                    total_fb = float(df_b05006.loc[mask_total, "estimate"].sum())
            except Exception:
                total_fb = None

            df_map = (
                df_b05006.dropna(subset=["iso3"])
                .groupby(["iso3"], as_index=False)["estimate"]
                .sum()
            )

            if df_map.empty:
                st.warning("No mappable sovereign countries found for this municipality/year.")
            else:
                color_col = "estimate"
                labels = {"estimate": "Foreign-born (estimate)"}

                if MODE in ("Academic", "Investigative") and total_fb and total_fb > 0:
                    df_map["share"] = (df_map["estimate"] / total_fb) * 100
                    color_col = "share"
                    labels = {"share": "Share of foreign-born (%)"}

                fig_world = px.choropleth(
                    df_map,
                    locations="iso3",
                    color=color_col,
                    title=f"Foreign-Born Population by Country of Birth — {origins_city} ({year})",
                    labels=labels,
                )
                fig_world.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_world, use_container_width=True)

                st.markdown("#### Top mappable origins")
                top = (
                    df_b05006.dropna(subset=["iso3"])
                    .groupby(["country_name"], as_index=False)["estimate"]
                    .sum()
                    .sort_values("estimate", ascending=False)
                    .head(20)
                )
                if MODE in ("Academic", "Investigative") and total_fb and total_fb > 0:
                    top["share_%"] = (top["estimate"] / total_fb) * 100
                st.dataframe(top, use_container_width=True, hide_index=True)

# ==================================================
# TAB 5: METHODOLOGY (Academic only)
# ==================================================
if tab_method is not None:
    with tab_method:
        with st.container():
            st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
            st.markdown("### Methodology & Notes (Academic Mode)")

            st.markdown(
                """
**Data source**  
- American Community Survey (ACS) **5-year** estimates, using **endpoint years** (e.g., “2022” represents the 2018–2022 pooled estimate).

**Interpretation**  
- 5-year ACS smooths year-to-year volatility; endpoints should be interpreted as rolling windows, not point-in-time measurements.

**Ranking / Distribution context**  
- For a given metric and year, *Gateway* distribution is taken across Gateway cities for that year.
- Z-score shown is: *(city_value − gateway_mean) / gateway_std* (population weighting not applied unless your query layer enforces it).

**Composite indices**  
- “z-avg” indices are simple means of metric z-scores (with sign inversions for metrics where higher implies less stress, e.g., median income).
- These are **diagnostic** indices, not causal measures.

**Scatter statistics**  
- Pearson correlation r and OLS line (if shown) reflect cross-sectional association across Gateway cities for the selected year.
- No causal claims are implied.
                """
            )

            st.markdown("### Warehouse integrity checklist")
            st.markdown(
                """
- Ensure every metric_key in the catalog maps to a populated series in `gateway_metrics`.
- For any metric used in indices, confirm consistent units + formatting in `metric_catalog`.
- For B05006 origins: country parsing is heuristic; verify label conventions in your ETL and adjust parsing rules if needed.
                """
            )