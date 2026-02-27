from __future__ import annotations

import json
import re
import textwrap
from typing import Dict, List, Optional, Tuple

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
    get_latest_year_available,
    get_gateway_metric_snapshot,
    get_gateway_metric_trend,
    get_state_metric_trend,
    get_gateway_ranking,
    get_gateway_scatter,
)
from src.story_angles import STORY_ANGLES

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

# --------------------------------------------------
# Design System Colors
# --------------------------------------------------
COLOR_TARGET = "#4A86C5"   # softened blue (selected)
COLOR_BASE = "#F28E8E"
COLOR_BG = "#f4f5f6"
COLOR_TEXT = "#2c2f33"
COLOR_BOSTON = "#5FB3A8"   # softened teal (Boston/Cambridge)

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CSS
# --------------------------------------------------
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
        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()

# --------------------------------------------------
# GeoJSON
# --------------------------------------------------
@st.cache_data
def load_ma_map() -> dict:
    with open("data/ma_municipalities.geojson") as f:
        return json.load(f)


ma_geo = load_ma_map()

def normalize_geo_key(name: str) -> str:
    s = str(name)

    # remove state
    s = s.replace(", Massachusetts", "")

    # remove city/town anywhere
    s = re.sub(r"\b(city|town|cdp)\b", "", s, flags=re.IGNORECASE)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)

    return s.strip().upper()


def clean_place_label(name: str) -> str:
    # "Boston city, Massachusetts" -> "Boston"
    s = str(name).replace(", Massachusetts", "").strip()
    s = re.sub(r"\b(city|town)\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s

NORMALIZED_ABBR = {
    normalize_geo_key(clean_place_label(k)): v
    for k, v in GATEWAY_ABBREVIATIONS.items()
}

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

# --------------------------------------------------
# Registry + Catalog
# --------------------------------------------------
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

# Metric catalog (single source of truth for labels, units, formatting)
catalog_df = get_metric_catalog()
catalog: Dict[str, Dict] = (
    catalog_df.set_index("metric_key").to_dict(orient="index") if not catalog_df.empty else {}
)

latest_year = get_latest_year_available()
if not latest_year:
    st.error("No data found in public.gateway_metrics. Verify your ETL/materialized view refresh.")
    st.stop()


def fmt_value(v: float, meta: Dict) -> str:
    if pd.isna(v):
        return "—"
    hint = (meta or {}).get("format_hint", "")
    if hint == "percent":
        return f"{float(v):.1f}%"
    if hint == "dollars":
        return f"${float(v):,.0f}"
    # fallback
    try:
        # treat as numeric
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


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <section class="hero">
        <h1>Gateway Cities Investigative Dashboard</h1>
        <div class="accent-line"></div>
        <p>
            Story-first exploration of demographic change, economic stress, and housing pressure across Massachusetts Gateway Cities.
            Source: ACS 5-year estimates (yearly endpoints; latest available in warehouse).
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Selected City State (single source of truth)
# --------------------------------------------------
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = gateway_city_options[0]

primary_city = st.session_state["selected_city"]
primary_fips = str(
    cities_all.loc[cities_all["place_name"] == primary_city, "place_fips"].iloc[0]
)

# Minimal metadata strip (cleaner than old top panel)
st.markdown(
    f"""
    <div style="display:flex; justify-content:flex-end; margin-bottom:10px;">
        <span class='pill'>Latest year: <b>{latest_year}</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_map, tab_story, tab_compare, tab_origins = st.tabs(
    ["Map", "Investigative Themes", "Compare Metrics", "Origins (B05006)"]
)

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
            for n in cities_all[
                cities_all["place_fips"].isin(gateway_fips)
            ]["place_name"]
        )

        boston_cambridge_names = set(
            normalize_geo_key(clean_place_label(n))
            for n in extra_cities["place_name"]
        )

        locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]

        # Compute centroids for labeling (keyed by normalized town name)
        town_centroids = {}

        for feature in ma_geo["features"]:
            town_raw = feature["properties"]["TOWN"]
            town_key = normalize_geo_key(town_raw)  # ✅ normalize here
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
            selected_index = None

            for i, town_name in enumerate(locations):
                town_norm = normalize_geo_key(town_name)

                if town_norm in town_fips_map_local and town_fips_map_local[town_norm] == selected_fips:
                    z_values.append(3)
                    selected_index = i
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

        # --------------------------------------------
        # Add Gateway Abbreviation Labels
        # --------------------------------------------
        label_lats = []
        label_lons = []
        label_text = []
        label_sizes = []
        label_colors = []

        # --------------------------------------------------
        # Manual label offsets (lat, lon)
        # --------------------------------------------------
        LABEL_OFFSETS = {
            "LEOMINSTER": (-0.020, 0.020),
            "METHUEN": (0.015, 0.012),
            "MALDEN": (0.010, -0.010),
            "CHELSEA": (-0.005, 0.02),
            "FITCHBURG": (0.010, -0.010),
            "REVERE": (0.010, 0.010),
            "EVERETT": (-0.005, -0.020),
        }

        for town_name in locations:
            town_key = normalize_geo_key(town_name)  # ✅ use same normalization

            if town_key in allowed_gateway_names and town_key in town_centroids:
                fips = town_fips_map[town_key]

                full_name = cities_all[cities_all["place_fips"] == fips]["place_name"].iloc[0]

                city_key = normalize_geo_key(clean_place_label(full_name))
                abbr = NORMALIZED_ABBR.get(city_key)

                if not abbr:
                    continue

                lat, lon = town_centroids[town_key]

                # Apply manual offset if defined
                offset_lat, offset_lon = LABEL_OFFSETS.get(town_key, (0, 0))
                lat += offset_lat
                lon += offset_lon

                label_lats.append(lat)
                label_lons.append(lon)
                label_text.append(abbr)

                # Emphasize selected city
                if fips == primary_fips:
                    label_sizes.append(15)
                    label_colors.append(COLOR_TARGET)
                else:
                    label_sizes.append(11)
                    label_colors.append("#111827")

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
                below=""  # <-- critical: draw above choropleth
            )
        )

        map_event = st.plotly_chart(
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

        st.markdown("### Gateway City Abbreviations")

        legend_cols = st.columns(4)

        for i, (full, abbr) in enumerate(GATEWAY_ABBREVIATIONS.items()):
            legend_cols[i % 4].markdown(
                f"**{abbr}** — {full.split(',')[0]}"
            )

        # --------------------------------------------------
        # Click-to-select (correct Streamlit pattern)
        # --------------------------------------------------
        selection = st.session_state.get("map_select")

        if selection and selection.get("selection"):
            points = selection["selection"].get("points", [])

            # Only keep choropleth clicks (must have 'location')
            loc_points = [p for p in points if p.get("location")]

            if loc_points:
                clicked_town = loc_points[-1]["location"]
                town_key = normalize_geo_key(clicked_town)

                new_fips = town_fips_map.get(town_key)

                if new_fips and new_fips in gateway_fips:
                    new_city = cities_all.loc[
                        cities_all["place_fips"] == str(new_fips),
                        "place_name"
                    ].iloc[0]

                    if st.session_state.get("selected_city") != new_city:
                        st.session_state["selected_city"] = new_city

                        # Clear selection to prevent replay
                        st.session_state["map_select"]["selection"]["points"] = []

                        st.rerun()

        # ==================================================
        # CITY INTELLIGENCE PANEL
        # ==================================================

        st.divider()
        st.markdown(f"# {primary_city.split(',')[0]} — City Profile")

        # --------------------------------------------------
        # 1️⃣ Snapshot Grid
        # --------------------------------------------------
        st.markdown("## Snapshot (Latest Year)")

        CORE_METRICS = [
            "total_population",
            "foreign_born_share",
            "median_income",
            "poverty_rate",
            "rent_burden_30_plus",
            "median_home_value",
            "gini_index",
        ]

        cols = st.columns(4)

        for i, mk in enumerate(CORE_METRICS):
            meta = catalog.get(mk, {"metric_label": mk})
            snap = get_gateway_metric_snapshot(primary_fips, mk)

            if snap is None or snap.empty:
                cols[i % 4].metric(meta.get("metric_label", mk), "—")
                continue

            value = snap["value"].iloc[0]
            delta = snap.get("delta_5yr", pd.Series([None])).iloc[0]

            cols[i % 4].metric(
                meta.get("metric_label", mk),
                fmt_value(value, meta),
                fmt_delta(delta, meta),
            )

        # --------------------------------------------------
        # 2️⃣ Structural Trend Panel
        # --------------------------------------------------
        st.divider()
        st.markdown("## Structural Trend")

        STRUCTURAL_METRICS = [
            "median_income",
            "poverty_rate",
            "rent_burden_30_plus",
            "foreign_born_share",
        ]

        trend_metric = st.selectbox(
            "Select trend metric",
            STRUCTURAL_METRICS,
            format_func=lambda k: catalog.get(k, {}).get("metric_label", k),
            key="map_trend_focus",
        )

        city_tr = get_gateway_metric_trend(primary_fips, trend_metric)
        state_tr = get_state_metric_trend(trend_metric)

        if city_tr is not None and not city_tr.empty:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=city_tr["year"],
                y=city_tr["value"],
                mode="lines",
                name=primary_city
            ))

            if state_tr is not None and not state_tr.empty:
                fig.add_trace(go.Scatter(
                    x=state_tr["year"],
                    y=state_tr["value"],
                    mode="lines",
                    name="Massachusetts"
                ))

            fig.update_layout(
                template="plotly_white",
                height=420,
                xaxis_title="Year",
                yaxis_title=catalog.get(trend_metric, {}).get("unit", ""),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available.")

        # --------------------------------------------------
        # 3️⃣ Gateway Ranking Panel
        # --------------------------------------------------
        st.divider()
        st.markdown("## Gateway Position")

        rank_df = get_gateway_ranking("poverty_rate", latest_year)
        
        st.write("Expected Gateway Count:", len(gateway_fips))
        st.write("Returned Ranking Rows:", len(rank_df))

        returned_fips = set(rank_df["place_fips"].astype(str))
        expected_fips = set(gateway_fips)

        missing_fips = expected_fips - returned_fips

        st.write("Missing FIPS:", missing_fips)

        if rank_df is not None and not rank_df.empty:

            city_rank = rank_df[
                rank_df["place_fips"].astype(str) == str(primary_fips)
            ]

            if not city_rank.empty:

                r = int(city_rank["rank_within_gateway"].iloc[0])
                total = len(rank_df)

                col1, col2 = st.columns(2)

                col1.metric(
                    "Poverty Rank (Gateway)",
                    f"{r} / {total}"
                )

                percentile = 100 - int((r / total) * 100)

                col2.metric(
                    "Relative Position",
                    f"{percentile}th percentile"
                )


# ==================================================
# TAB 2: INVESTIGATIVE THEMES (Story Angles)
# ==================================================
with tab_story:
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Investigative Themes")

        # City selector (gateway-only)
        cities_df = get_cities(gateway_only=True)
        story_city = st.selectbox(
            "Gateway City",
            cities_df["place_name"].tolist(),
            index=max(0, cities_df["place_name"].tolist().index(primary_city)) if primary_city in cities_df["place_name"].tolist() else 0,
        )
        place_fips = str(cities_df.loc[cities_df["place_name"] == story_city, "place_fips"].iloc[0])

        angle_key = st.selectbox(
            "Theme",
            list(STORY_ANGLES.keys()),
            format_func=lambda k: STORY_ANGLES[k]["title"],
        )
        angle = STORY_ANGLES[angle_key]

        investigate = st.toggle("🔍 Explore Relationships", value=False)

        st.subheader(angle["title"])

        # Snapshot cards
        metrics = angle.get("metrics", [])
        if not metrics:
            st.info("No metrics configured for this theme.")
        else:
            cols = st.columns(min(len(metrics), 6))
            for i, mk in enumerate(metrics):
                meta = catalog.get(mk, {"metric_label": mk, "format_hint": "number"})
                snap = get_gateway_metric_snapshot(place_fips, mk)
                if snap is None or snap.empty:
                    cols[i % len(cols)].metric(meta.get("metric_label", mk), "—")
                    continue

                v = snap["value"].iloc[0]
                yr = int(snap["year"].iloc[0]) if pd.notna(snap["year"].iloc[0]) else latest_year
                rnk = snap.get("rank_within_gateway", pd.Series([None])).iloc[0]
                d5 = snap.get("delta_5yr", pd.Series([None])).iloc[0]

                main = fmt_value(v, meta)
                delta = fmt_delta(d5, meta)

                c = cols[i % len(cols)]
                c.metric(meta.get("metric_label", mk), main, delta)
                if pd.notna(rnk):
                    c.caption(f"Rank: {int(rnk)} (Gateway Cities) • {yr}")

        st.divider()

        # Lead metric selection
        lead_metric = st.selectbox(
            "Trend focus metric",
            metrics if metrics else list(catalog.keys()),
            format_func=lambda k: catalog.get(k, {"metric_label": k}).get("metric_label", k),
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
            fig.add_trace(go.Scatter(x=trend["year"], y=trend["City"], mode="lines", name=story_city))
            if "Massachusetts" in trend.columns:
                fig.add_trace(go.Scatter(x=trend["year"], y=trend["Massachusetts"], mode="lines", name="Massachusetts"))
            fig.update_layout(
                template="plotly_white",
                height=430,
                xaxis_title="Year",
                yaxis_title=meta.get("unit", ""),
                legend=dict(title=""),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Ranking table for latest year
        st.markdown(
            f"**Gateway ranking ({latest_year}):** {meta.get('metric_label', lead_metric)}"
        )
        rank_df = get_gateway_ranking(lead_metric, latest_year)
        st.dataframe(rank_df, use_container_width=True, hide_index=True)

        # Investigative mode
        if investigate:
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

                idx = st.selectbox("Choose a comparison", range(len(pairs)), format_func=lambda i: pair_labels[i])
                xk, yk = pairs[idx]

                sc = get_gateway_scatter(xk, yk, latest_year)
                xl = catalog.get(xk, {"metric_label": xk}).get("metric_label", xk)
                yl = catalog.get(yk, {"metric_label": yk}).get("metric_label", yk)

                if sc is None or sc.empty:
                    st.info("No scatter data available for this comparison.")
                else:
                    sc = sc.copy()
                    sc["is_selected"] = sc["place_fips"].astype(str) == str(place_fips)

                    fig_sc = px.scatter(
                        sc,
                        x="x",
                        y="y",
                        hover_name="place_name",
                        title=f"{latest_year}: {xl} (x) vs {yl} (y)",
                    )
                    # highlight selected city
                    fig_sc.update_traces(
                        marker=dict(size=10),
                        selector=dict(mode="markers"),
                    )
                    fig_sc.add_trace(
                        go.Scatter(
                            x=sc.loc[sc["is_selected"], "x"],
                            y=sc.loc[sc["is_selected"], "y"],
                            mode="markers",
                            name=story_city,
                            marker=dict(size=16, symbol="diamond"),
                            hovertemplate=f"<b>{story_city}</b><br>{xl}: %{{x}}<br>{yl}: %{{y}}<extra></extra>",
                        )
                    )
                    fig_sc.update_layout(template="plotly_white", height=520)
                    st.plotly_chart(fig_sc, use_container_width=True)

                    city_row = sc[sc["is_selected"]]
                    if not city_row.empty:
                        st.caption(
                            f"{story_city}: {xl}={city_row['x'].iloc[0]:.2f}, {yl}={city_row['y'].iloc[0]:.2f}"
                        )


# ==================================================
# TAB 3: COMPARE METRICS (cross-city + cross-metric)
# ==================================================
with tab_compare:
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Compare Metrics")

        # --------------------------------------------
        # Local multi-city selector (independent)
        # --------------------------------------------
        compare_cities = st.multiselect(
            "Select Gateway Cities (Max 3)",
            options=gateway_city_options,
            default=[primary_city],
            max_selections=3,
        )

        if not compare_cities:
            compare_cities = [primary_city]

        selected_fips = {
            city: str(
                cities_all.loc[
                    cities_all["place_name"] == city,
                    "place_fips"
                ].iloc[0]
            )
            for city in compare_cities
        }

        if not catalog:
            st.info("Metric catalog is empty. Verify public.metric_catalog.")
            st.stop()

        metric_keys = sorted(list(catalog.keys()))
        metric_labels = {k: catalog[k].get("metric_label", k) for k in metric_keys}

        col_a, col_b = st.columns([1.2, 2.3])

        # --------------------------------------------
        # Trend selector
        # --------------------------------------------
        with col_a:
            metric_to_compare = st.selectbox(
                "Trend metric",
                metric_keys,
                index=metric_keys.index("median_income")
                if "median_income" in metric_keys else 0,
                format_func=lambda k: metric_labels.get(k, k),
            )

        # --------------------------------------------
        # Multi-city trend
        # --------------------------------------------
        with col_b:
            st.markdown("**Multi-city trend**")

            fig = go.Figure()

            for city_name, fips in selected_fips.items():
                tr = get_gateway_metric_trend(fips, metric_to_compare)
                if tr is None or tr.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=tr["year"],
                        y=tr["value"],
                        mode="lines",
                        name=city_name,
                    )
                )

            fig.update_layout(
                template="plotly_white",
                height=450,
                xaxis_title="Year",
                yaxis_title=catalog.get(metric_to_compare, {}).get("unit", ""),
                legend=dict(title=""),
            )

        # ==================================================
        # Cross-metric scatter
        # ==================================================
        st.divider()
        st.markdown("### Cross-metric scatter (Gateway Cities)")

        c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

        with c1:
            metric_x = st.selectbox(
                "X metric",
                metric_keys,
                index=metric_keys.index("rent_burden_30_plus")
                if "rent_burden_30_plus" in metric_keys else 0,
                format_func=lambda k: metric_labels.get(k, k),
                key="scatter_x",
            )

        with c2:
            metric_y = st.selectbox(
                "Y metric",
                metric_keys,
                index=metric_keys.index("poverty_rate")
                if "poverty_rate" in metric_keys else min(1, len(metric_keys) - 1),
                format_func=lambda k: metric_labels.get(k, k),
                key="scatter_y",
            )

        with c3:
            sc_year = st.selectbox(
                "Year",
                [latest_year],
                index=0,
                key="scatter_year",
            )

        sc = get_gateway_scatter(metric_x, metric_y, sc_year)

        if sc is None or sc.empty:
            st.info("No data available for that scatter combination.")
        else:
            sc = sc.copy()
            sc["is_selected"] = sc["place_fips"].astype(str).isin(
                set(selected_fips.values())
            )

            xl = catalog.get(metric_x, {"metric_label": metric_x}).get("metric_label", metric_x)
            yl = catalog.get(metric_y, {"metric_label": metric_y}).get("metric_label", metric_y)

            fig_sc = px.scatter(
                sc,
                x="x",
                y="y",
                hover_name="place_name",
                title=f"{sc_year}: {xl} (x) vs {yl} (y)",
            )

            # Highlight selected cities
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

            fig_sc.update_layout(
                template="plotly_white",
                height=560,
            )

            st.plotly_chart(fig_sc, use_container_width=True)


# ==================================================
# TAB 4: ORIGINS (B05006)
# ==================================================
SOURCE_BIRTH_TABLE = "B05006"


def label_to_country_name(variable_label: str) -> Optional[str]:
    """
    Extract a country-ish token from the B05006 variable_label.
    Drop broad regions and obvious aggregates.
    """
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

        # Choose a city for origins (default primary)
        cities_df = get_cities(gateway_only=True)
        origins_city = st.selectbox(
            "City for origins map",
            cities_df["place_name"].tolist(),
            index=max(0, cities_df["place_name"].tolist().index(primary_city)) if primary_city in cities_df["place_name"].tolist() else 0,
            key="origins_city",
        )
        origins_fips = str(cities_df.loc[cities_df["place_name"] == origins_city, "place_fips"].iloc[0])

        # Use latest year from gateway_metrics as default
        year = st.selectbox("Year", [latest_year], index=0, key="origins_year")

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
            df_map = (
                df_b05006.dropna(subset=["iso3"])
                .groupby(["iso3"], as_index=False)["estimate"]
                .sum()
            )

            if df_map.empty:
                st.warning("No mappable sovereign countries found for this municipality/year.")
            else:
                fig_world = px.choropleth(
                    df_map,
                    locations="iso3",
                    color="estimate",
                    title=f"Foreign-Born Population by Country of Birth (approx.) — {origins_city} ({year})",
                    labels={"estimate": "Foreign-born (estimate)"},
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
                st.dataframe(top, use_container_width=True, hide_index=True)