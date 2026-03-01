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
    get_latest_year_available,  # ok if unused
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

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# CSS (YOUR editorial CSS is the source of truth)
# ==================================================
def load_css() -> None:
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    # Minimal additions only (do NOT override your fonts/hero)
    st.markdown(
        """
        <style>
        .section-card-marker { display:none; }

        /* card wrapper (kept light; your CSS controls fonts/colors) */
        div[data-testid="stVerticalBlock"]:has(.section-card-marker) {
            background: #ffffff;
            padding: 26px;
            border-radius: 2px;
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

        /* KPI spacing refinement (keeps your typography) */
        div[data-testid="stMetric"] {
            padding: 8px 10px;
            border-radius: 2px;
        }
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


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


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

    v_ser = df.loc[df["place_fips"] == str(place_fips), value_col]
    v = safe_float(v_ser.iloc[0]) if not v_ser.empty else None

    mean = safe_float(df[value_col].mean())
    median = safe_float(df[value_col].median())
    std = safe_float(df[value_col].std(ddof=0))

    z = None
    if v is not None and std not in (None, 0.0):
        z = (v - mean) / std

    percentile = None
    if v is not None:
        percentile = float((df[value_col] <= v).mean() * 100)

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
    if trend_df is None or trend_df.empty:
        return TrendDiagnostics(None, None, None, None, None)

    df = trend_df.copy()
    if "year" not in df.columns or "value" not in df.columns:
        return TrendDiagnostics(None, None, None, None, None)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).sort_values("year")
    if len(df) < 2:
        last = safe_float(df["value"].iloc[-1]) if len(df) else None
        first = safe_float(df["value"].iloc[0]) if len(df) else None
        return TrendDiagnostics(None, None, None, last, first)

    years = df["year"].to_numpy()
    vals = df["value"].to_numpy()

    def slope_over(window_years: int) -> Optional[float]:
        y_max = years.max()
        mask = years >= (y_max - window_years)
        suby = years[mask]
        subv = vals[mask]
        if len(suby) < 2:
            return None
        try:
            return float(np.polyfit(suby, subv, 1)[0])
        except Exception:
            return None

    slope_5 = slope_over(5)
    slope_10 = slope_over(10)

    delta_5 = None
    try:
        y_max = float(years.max())
        target = y_max - 5
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


def build_narrative_summary(
    city_name: str,
    year: int,
    catalog: Dict[str, Dict],
    place_fips: str,
    focus_metrics: List[str],
    advanced: bool,
) -> List[str]:
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

        if val is None:
            continue

        if delta is not None:
            bullets.append(f"**{label}:** {fmt_value(val, meta)} ({fmt_delta(delta, meta)}).")
        else:
            bullets.append(f"**{label}:** {fmt_value(val, meta)}.")

        if advanced and tr is not None and not tr.empty:
            diag = compute_trend_diagnostics(tr)
            if diag.slope_10yr is not None and abs(diag.slope_10yr) > 0:
                bullets.append(
                    f"<span class='subtle'>Trend signal: ~{diag.slope_10yr:+.3g} per year (10yr linear slope).</span>"
                )

    return bullets[:10]


# ==================================================
# GEOJSON (MAP)
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
    st.error("No gateway cities found. Check is_gateway_city flags / gateway_cities table.")
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

# ==================================================
# SESSION STATE (single source of truth)
# ==================================================
if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = max_year
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = gateway_city_options[0]
if "advanced" not in st.session_state:
    st.session_state["advanced"] = False

# ==================================================
# HERO + TOP CONTROL STRIP (keep your design)
# ==================================================
st.markdown(
    """
    <section class="hero">
        <h1>Gateway Cities Dashboard</h1>
        <div class="accent-line"></div>
        <p>
            Story-first exploration of demographic change, economic stress, and housing pressure across Massachusetts Gateway Cities.
            Source: ACS 5-year estimates (endpoint years; latest available in warehouse).
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

cA, cB, cC, cD = st.columns([2.2, 1.2, 1.2, 1.0])

with cB:
    st.session_state["selected_city"] = st.selectbox(
        "Gateway city",
        options=gateway_city_options,
        index=gateway_city_options.index(st.session_state["selected_city"])
        if st.session_state["selected_city"] in gateway_city_options
        else 0,
        key="global_city",
    )

with cC:
    st.session_state["selected_year"] = st.selectbox(
        "Analysis year",
        options=available_years,
        index=available_years.index(st.session_state["selected_year"]),
        key="global_year",
    )

with cD:
    st.markdown(
        f"<div style='text-align:right;'><span class='pill'>Data range: <b>{min_year}–{max_year}</b></span></div>",
        unsafe_allow_html=True,
    )

selected_year = int(st.session_state["selected_year"])
primary_city = st.session_state["selected_city"]
primary_fips = str(cities_all.loc[cities_all["place_name"] == primary_city, "place_fips"].iloc[0])

# One clean toggle: Advanced (only meaningful in Investigative/Academic; keep UI uncluttered)
adv_allowed = True
st.session_state["advanced"] = st.toggle(
    "Advanced analysis",
    value=(st.session_state["advanced"] if adv_allowed else False),
    disabled=not adv_allowed,
    help="Shows diagnostics, full ranking tables, and relationships. Hidden in Public/Executive.",
    key="advanced_toggle",
)
ADV = bool(st.session_state["advanced"] and adv_allowed)

# ==================================================
# TABS (consistent; no extra page systems)
# ==================================================
tabs = ["Map", "Investigative Themes", "Compare Metrics", "Origins (B05006)", "Methodology"]

tab_objs = st.tabs(tabs)
tab_map = tab_objs[0]
tab_story = tab_objs[1]
tab_compare = tab_objs[2]
tab_origins = tab_objs[3]
tab_method = tab_objs[4]

# ==================================================
# TAB 1: MAP (choropleth + click select + city profile)
# ==================================================
with tab_map:
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Geographic Context")

        # We key the map to TOWN (as in your prior working version)
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

        # Centroids for labels
        town_centroids: Dict[str, Tuple[float, float]] = {}
        for feature in ma_geo["features"]:
            town_raw = feature["properties"]["TOWN"]
            town_key = normalize_geo_key(town_raw)
            coords = feature["geometry"]["coordinates"]
            lats: List[float] = []
            lons: List[float] = []

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
            locations_in: List[str],
            allowed_gateway_names_in: set,
            town_fips_map_local: Dict[str, str],
            selected_fips: str,
            c_lat: float,
            c_lon: float,
            boston_cambridge_names_in: set,
        ) -> go.Figure:
            z_values: List[int] = []
            for town_name in locations_in:
                town_norm = normalize_geo_key(town_name)
                if town_norm in town_fips_map_local and town_fips_map_local[town_norm] == selected_fips:
                    z_values.append(3)
                elif town_norm in boston_cambridge_names_in:
                    z_values.append(2)
                elif town_norm in allowed_gateway_names_in:
                    z_values.append(1)
                else:
                    z_values.append(0)

            trace = go.Choroplethmapbox(
                geojson=geojson,
                locations=locations_in,
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

        # Abbreviation labels (offsets for dense Boston-area)
        LABEL_OFFSETS = {
            "LEOMINSTER": (-0.020, 0.020),
            "METHUEN": (0.015, 0.012),
            "MALDEN": (0.010, -0.010),
            "CHELSEA": (-0.005, 0.020),
            "FITCHBURG": (0.010, -0.010),
            "REVERE": (0.010, 0.010),
            "EVERETT": (-0.005, -0.020),
        }

        label_lats: List[float] = []
        label_lons: List[float] = []
        label_text: List[str] = []
        for town_name in locations:
            town_key = normalize_geo_key(town_name)
            if town_key in allowed_gateway_names and town_key in town_centroids:
                fips = town_fips_map.get(town_key)
                if not fips:
                    continue
                full_name = cities_all[cities_all["place_fips"] == str(fips)]["place_name"].iloc[0]
                city_key = normalize_geo_key(clean_place_label(full_name))
                abbr = NORMALIZED_ABBR.get(city_key)
                if not abbr:
                    continue
                lat, lon = town_centroids[town_key]
                off_lat, off_lon = LABEL_OFFSETS.get(town_key, (0.0, 0.0))
                label_lats.append(lat + off_lat)
                label_lons.append(lon + off_lon)
                label_text.append(abbr)

        fig_map.add_trace(
            go.Scattermapbox(
                lat=label_lats,
                lon=label_lons,
                mode="text",
                text=label_text,
                textfont=dict(size=11, color="#111111"),
                textposition="middle center",
                hoverinfo="skip",
                showlegend=False,
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

        # Click-to-select (robust; no double-click logic)
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
                        # hard reset selection to prevent "lag click"
                        st.session_state["map_select"] = {"selection": {"points": []}}
                        st.rerun()

        # =========================
        # CITY PROFILE (keeps your design)
        # =========================
        st.divider()
        st.markdown(f"# {primary_city.split(',')[0]} — City Profile")

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

        bullets = build_narrative_summary(
            primary_city.split(",")[0],
            selected_year,
            catalog,
            primary_fips,
            focus_for_summary,
            advanced=ADV,
        )
        if not ADV:
            bullets = [b for b in bullets if "Trend signal" not in b]
        if not bullets:
            st.info("No summary available (missing snapshot data for this year).")
        else:
            for b in bullets[:6]:
                st.markdown(f"- {b}", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"## Snapshot {selected_year}")

        core_metrics = [
            "total_population",
            "foreign_born_share",
            "median_income",
            "poverty_rate",
            "rent_burden_30_plus",
            "median_home_value",
            "gini_index",
        ]

        cols = st.columns(4)
        for i, mk in enumerate(core_metrics):
            meta = catalog.get(mk, {"metric_label": mk})
            snap = get_gateway_metric_snapshot(primary_fips, mk, selected_year)
            if snap is None or snap.empty:
                cols[i % 4].metric(meta.get("metric_label", mk), "—")
                continue
            value = snap.get("value", pd.Series([None])).iloc[0]
            delta = snap.get("delta_5yr", pd.Series([None])).iloc[0]
            cols[i % 4].metric(meta.get("metric_label", mk), fmt_value(value, meta), fmt_delta(delta, meta))

        st.divider()
        st.markdown("## Structural Trend")

        structural_metrics = [k for k in ["median_income", "poverty_rate", "rent_burden_30_plus", "foreign_born_share", "total_population"] if k in catalog]

        trend_metric = st.selectbox(
            "Select trend metric",
            structural_metrics,
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
            fig.update_layout(template="plotly_white", height=420, xaxis_title="Year", yaxis_title=meta.get("unit", ""), legend=dict(title=""))
            st.plotly_chart(fig, use_container_width=True)

            if ADV:
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
            if ctx.percentile is not None:
                c2.metric("Percentile (empirical)", f"{ctx.percentile:.0f}th")

            if ADV and ctx.z is not None:
                c3.metric("Z-score vs gateway", f"{ctx.z:+.2f}")
            if ADV and ctx.mean is not None and ctx.value is not None:
                gap = ctx.value - ctx.mean
                c4.metric("Gap vs gateway mean", f"{gap:+.3g}")

            if ADV:
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
            else:
                sub = rank_df.copy()
                sub["place_fips"] = sub["place_fips"].astype(str)
                one = sub[sub["place_fips"] == str(primary_fips)]
                if not one.empty:
                    st.dataframe(one, use_container_width=True, hide_index=True)

# ==================================================
# TAB 2: INVESTIGATIVE THEMES (no extra toggles; Advanced controls depth)
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

        st.subheader(angle["title"])

        metrics = angle.get("metrics", [])
        if not metrics:
            st.info("No metrics configured for this theme.")
        else:
            show_metrics = metrics[:4]
            cols = st.columns(min(len(show_metrics), 6))
            for i, mk in enumerate(show_metrics):
                meta = catalog.get(mk, {"metric_label": mk, "format_hint": "number"})
                snap = get_gateway_metric_snapshot(place_fips, mk, selected_year)
                if snap is None or snap.empty:
                    cols[i % len(cols)].metric(meta.get("metric_label", mk), "—")
                    continue
                v = snap["value"].iloc[0]
                d5 = snap.get("delta_5yr", pd.Series([None])).iloc[0]
                cols[i % len(cols)].metric(meta.get("metric_label", mk), fmt_value(v, meta), fmt_delta(d5, meta))

        st.divider()

        lead_metric = st.selectbox(
            "Trend focus metric",
            metrics if metrics else list(catalog.keys()),
            format_func=lambda k: catalog.get(k, {"metric_label": k}).get("metric_label", k),
            key="lead_metric",
        )

        city_trend = get_gateway_metric_trend(place_fips, lead_metric)
        ma_trend = get_state_metric_trend(lead_metric)

        meta = catalog.get(lead_metric, {"metric_label": lead_metric})
        st.markdown(f"**Trend:** {meta.get('metric_label', lead_metric)}")

        if city_trend is None or city_trend.empty:
            st.info("No trend data available for this metric/city.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=city_trend["year"], y=city_trend["value"], mode="lines", name=story_city.split(",")[0]))
            if ma_trend is not None and not ma_trend.empty:
                fig.add_trace(go.Scatter(x=ma_trend["year"], y=ma_trend["value"], mode="lines", name="Massachusetts"))
            fig.update_layout(template="plotly_white", height=430, xaxis_title="Year", yaxis_title=meta.get("unit", ""), legend=dict(title=""))
            st.plotly_chart(fig, use_container_width=True)

            if ADV:
                diag = compute_trend_diagnostics(city_trend)
                c1, c2, c3 = st.columns(3)
                if diag.delta_5yr is not None:
                    c1.metric("5-year change (approx.)", fmt_delta(diag.delta_5yr, meta) or f"{diag.delta_5yr:+.3g}")
                if diag.slope_5yr is not None:
                    c2.metric("Slope (5yr window)", f"{diag.slope_5yr:+.3g} / year")
                if diag.slope_10yr is not None:
                    c3.metric("Slope (10yr window)", f"{diag.slope_10yr:+.3g} / year")

        st.divider()
        st.markdown(f"### Gateway Ranking — {meta.get('metric_label')} ({selected_year})")

        rank_df = get_gateway_ranking(lead_metric, selected_year)
        if rank_df is None or rank_df.empty:
            st.info("No ranking data available for this metric/year.")
        else:
            if ADV:
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
            else:
                sub = rank_df.copy()
                sub["place_fips"] = sub["place_fips"].astype(str)
                one = sub[sub["place_fips"] == str(place_fips)]
                if not one.empty:
                    st.dataframe(one, use_container_width=True, hide_index=True)
                else:
                    st.info("Selected city not found in ranking output for this year.")

        if ADV:
            st.divider()
            st.subheader("Relationships (Gateway Cities only)")

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
                    sc["x"] = pd.to_numeric(sc["x"], errors="coerce")
                    sc["y"] = pd.to_numeric(sc["y"], errors="coerce")
                    sc = sc.dropna(subset=["x", "y"])
                    if sc.empty:
                        st.info("No valid numeric scatter data available for this comparison.")
                    else:
                        sc["is_selected"] = sc["place_fips"].astype(str) == str(place_fips)
                        stats = compute_scatter_stats(sc)

                        fig_sc = px.scatter(
                            sc,
                            x="x",
                            y="y",
                            hover_name="place_name",
                            title=f"{selected_year}: {xl} (x) vs {yl} (y)",
                        )

                        if ADV and stats.slope is not None and stats.intercept is not None:
                            xx = np.linspace(float(sc["x"].min()), float(sc["x"].max()), 50)
                            yy = stats.slope * xx + stats.intercept
                            fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="OLS fit"))

                        sel = sc[sc["is_selected"]]
                        if not sel.empty:
                            fig_sc.add_trace(
                                go.Scatter(
                                    x=sel["x"],
                                    y=sel["y"],
                                    mode="markers",
                                    name=story_city.split(",")[0],
                                    marker=dict(size=16, symbol="diamond"),
                                    hovertemplate=f"<b>{story_city}</b><br>{xl}: %{{x}}<br>{yl}: %{{y}}<extra></extra>",
                                )
                            )

                        fig_sc.update_layout(template="plotly_white", height=520)
                        st.plotly_chart(fig_sc, use_container_width=True)

                        c1, c2, c3 = st.columns(3)
                        if stats.r is not None:
                            c1.metric("Correlation r", f"{stats.r:+.2f}")
                        if stats.r2 is not None:
                            c2.metric("R²", f"{stats.r2:.2f}")
                        if stats.n is not None:
                            c3.metric("N (cities)", f"{stats.n}")

# ==================================================
# TAB 3: COMPARE METRICS (advanced = extra stats/regression)
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
            sc["x"] = pd.to_numeric(sc["x"], errors="coerce")
            sc["y"] = pd.to_numeric(sc["y"], errors="coerce")
            sc = sc.dropna(subset=["x", "y"])
            if sc.empty:
                st.info("No valid numeric scatter data available.")
            else:
                sc["is_selected"] = sc["place_fips"].astype(str).isin(set(selected_fips.values()))
                xl = metric_labels.get(metric_x, metric_x)
                yl = metric_labels.get(metric_y, metric_y)

                stats = compute_scatter_stats(sc)

                fig_sc = px.scatter(sc, x="x", y="y", hover_name="place_name", title=f"{sc_year}: {xl} (x) vs {yl} (y)")

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

                if ADV and stats.slope is not None and stats.intercept is not None:
                    x_min = float(sc["x"].min())
                    x_max = float(sc["x"].max())
                    if x_min != x_max:
                        xx = np.linspace(x_min, x_max, 50)
                        yy = stats.slope * xx + stats.intercept
                        fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="OLS fit"))

                fig_sc.update_layout(template="plotly_white", height=560)
                st.plotly_chart(fig_sc, use_container_width=True)

                if ADV:
                    c1, c2, c3 = st.columns(3)
                    if stats.r is not None:
                        c1.metric("Correlation r", f"{stats.r:+.2f}")
                    if stats.r2 is not None:
                        c2.metric("R²", f"{stats.r2:.2f}")
                    if stats.n is not None:
                        c3.metric("N (cities)", f"{stats.n}")

# ==================================================
# TAB 4: ORIGINS (B05006) - hardened
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
            "This panel uses raw B05006 rows (place of birth). Country parsing is heuristic; adjust rules if ETL naming differs."
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
            st.info("No B05006 rows returned for this city/year.")
        else:
            df = df_b05006.copy()
            if "estimate" not in df.columns or "variable_label" not in df.columns:
                st.warning("B05006 output missing required columns: estimate / variable_label.")
            else:
                df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
                df = df.dropna(subset=["estimate"])
                if df.empty:
                    st.info("No numeric B05006 values available.")
                else:
                    df["country_name"] = df["variable_label"].apply(label_to_country_name)
                    df = df.dropna(subset=["country_name"])
                    if df.empty:
                        st.info("No country-level rows detected in B05006.")
                    else:
                        df["iso3"] = df["country_name"].apply(country_to_iso3)

                        total_fb = None
                        try:
                            mask_total = df["variable_label"].str.contains("Total foreign-born", case=False, na=False)
                            if mask_total.any():
                                total_fb = float(df.loc[mask_total, "estimate"].sum())
                        except Exception:
                            total_fb = None

                        df_map = df.dropna(subset=["iso3"]).groupby(["iso3"], as_index=False)["estimate"].sum()
                        if df_map.empty:
                            st.warning("No mappable sovereign countries found for this municipality/year.")
                        else:
                            color_col = "estimate"
                            labels = {"estimate": "Foreign-born (estimate)"}
                            if ADV and total_fb and total_fb > 0:
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
                                df.dropna(subset=["iso3"])
                                .groupby(["country_name"], as_index=False)["estimate"]
                                .sum()
                                .sort_values("estimate", ascending=False)
                                .head(20)
                            )
                            if ADV and total_fb and total_fb > 0:
                                top["share_%"] = (top["estimate"] / total_fb) * 100
                            st.dataframe(top, use_container_width=True, hide_index=True)

# ==================================================
# TAB 5: METHODOLOGY (Academic only; tied to mode, not extra toggles)
# ==================================================
with tab_method:
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
- Z-score shown (when Advanced is enabled) is: *(city_value − gateway_mean) / gateway_std* (population weighting not applied unless your query layer enforces it).

**Scatter statistics**  
- Pearson correlation r and OLS line (if shown) reflect cross-sectional association across Gateway cities for the selected year.
- No causal claims are implied.

**B05006 origins**  
- Country parsing is heuristic; validate label conventions in your ETL and adjust parsing rules if needed.
                """
            )