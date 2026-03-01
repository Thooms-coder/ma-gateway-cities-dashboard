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
    get_cities,
    get_gateway_fips,
    get_place_source_table_year,
    get_metric_catalog,
    get_latest_year_available,
    get_gateway_metric_snapshot,
    get_gateway_metric_trend,
    get_state_metric_trend,
    get_gateway_ranking,
    get_gateway_scatter,
    get_available_gateway_years,
)
from src.story_angles import STORY_ANGLES


# ==================================================
# CONFIG
# ==================================================

COLOR_TARGET = "#4A86C5"
COLOR_BASE = "#F28E8E"
COLOR_BG = "#f4f5f6"
COLOR_TEXT = "#2c2f33"
COLOR_BOSTON = "#5FB3A8"

MODES = ["Public", "Investigative", "Academic", "Executive"]

st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# HELPERS
# ==================================================

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except:
        return None


def fmt_value(v: float, meta: Dict) -> str:
    if pd.isna(v):
        return "—"
    hint = meta.get("format_hint", "")
    if hint == "percent":
        return f"{float(v):.1f}%"
    if hint == "dollars":
        return f"${float(v):,.0f}"
    return f"{float(v):,.2f}"


def fmt_delta(d: float, meta: Dict) -> str:
    if pd.isna(d):
        return ""
    hint = meta.get("format_hint", "")
    if hint == "percent":
        return f"{float(d):+.1f} pts"
    if hint == "dollars":
        return f"{float(d):+,.0f}"
    return f"{float(d):+.2f}"


# ==================================================
# ANALYTICS (Schema-Locked)
# ==================================================

@dataclass
class DistributionContext:
    value: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    z: Optional[float]
    rank: Optional[int]
    n: Optional[int]

def compute_distribution_context(rank_df: pd.DataFrame, place_fips: str):
    if rank_df is None or rank_df.empty:
        return DistributionContext(None, None, None, None, None, None)

    df = rank_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    v_row = df[df["place_fips"].astype(str) == str(place_fips)]
    if v_row.empty:
        return DistributionContext(None, None, None, None, None, len(df))

    v = float(v_row["value"].iloc[0])
    mean = float(df["value"].mean())
    std = float(df["value"].std(ddof=0))
    z = (v - mean) / std if std != 0 else None

    rank = None
    if "rank_within_gateway" in v_row.columns:
        try:
            rank = int(v_row["rank_within_gateway"].iloc[0])
        except:
            rank = None

    return DistributionContext(v, mean, std, z, rank, len(df))

@dataclass
class ScatterStats:
    slope: Optional[float]
    intercept: Optional[float]
    r: Optional[float]
    r2: Optional[float]
    n: Optional[int]


def compute_scatter_stats(df: pd.DataFrame) -> ScatterStats:
    if df is None or df.empty:
        return ScatterStats(None, None, None, None, None)

    df = df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    if len(df) < 2:
        return ScatterStats(None, None, None, None, len(df))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    r2 = r ** 2

    return ScatterStats(float(slope), float(intercept), float(r), float(r2), len(df))

@dataclass
class TrendDiagnostics:
    slope_5yr: Optional[float]
    slope_10yr: Optional[float]
    delta_5yr: Optional[float]


def compute_trend_diagnostics(trend_df: pd.DataFrame):
    if trend_df is None or trend_df.empty:
        return TrendDiagnostics(None, None, None)

    df = trend_df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).sort_values("year")

    if len(df) < 2:
        return TrendDiagnostics(None, None, None)

    years = df["year"].to_numpy()
    vals = df["value"].to_numpy()

    def slope(window):
        max_year = years.max()
        mask = years >= (max_year - window)
        if mask.sum() < 2:
            return None
        return float(np.polyfit(years[mask], vals[mask], 1)[0])

    slope5 = slope(5)
    slope10 = slope(10)

    delta5 = None
    if "delta_5yr" in df.columns:
        delta5 = safe_float(df["delta_5yr"].iloc[-1])

    return TrendDiagnostics(slope5, slope10, delta5)


# ==================================================
# DATA LOAD
# ==================================================

cities_all = get_cities(gateway_only=False)
cities_all["place_fips"] = cities_all["place_fips"].astype(str)

gateway_fips = set(get_gateway_fips()["place_fips"].astype(str))

gateway_city_options = (
    cities_all[cities_all["place_fips"].isin(gateway_fips)]
    ["place_name"]
    .tolist()
)

catalog_df = get_metric_catalog()
catalog = catalog_df.set_index("metric_key").to_dict(orient="index")

available_years = get_available_gateway_years()

if available_years:
    min_year = min(available_years)
    max_year = max(available_years)
else:
    st.error("No available years returned from warehouse.")
    st.stop()

if "selected_city" not in st.session_state:
    gateway_cities_df = cities_all[cities_all["place_fips"].isin(gateway_fips)]

    if gateway_cities_df.empty:
        st.error("No gateway cities returned from warehouse.")
        st.stop()

    if "selected_city" not in st.session_state:
        st.session_state["selected_city"] = gateway_cities_df["place_name"].iloc[0]

if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = max_year

if "mode" not in st.session_state:
    st.session_state["mode"] = "Investigative"

primary_city = st.session_state["selected_city"]
primary_fips = cities_all.loc[cities_all["place_name"] == primary_city, "place_fips"].iloc[0]
selected_year = st.session_state["selected_year"]

# ==================================================
# HEADER
# ==================================================

st.title("Gateway Cities Dashboard")

c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    st.session_state["mode"] = st.radio("Mode", MODES, horizontal=True)

with c2:
    st.session_state["selected_year"] = st.selectbox("Year", available_years, index=available_years.index(selected_year))

with c3:
    st.write(f"Data range: {min_year}-{max_year}")

MODE = st.session_state["mode"]
selected_year = st.session_state["selected_year"]

# ==================================================
# TABS
# ==================================================

tabs = ["Map", "Investigative Themes", "Compare Metrics", "Origins"]
if MODE == "Academic":
    tabs.append("Methodology")

tab_objs = st.tabs(tabs)
tab_map, tab_story, tab_compare, tab_origins = tab_objs[:4]
tab_method = tab_objs[4] if MODE == "Academic" else None


# ==================================================
# TAB: MAP
# ==================================================

with tab_map:
    st.header(primary_city.split(",")[0])

    # Snapshot
    st.subheader(f"Snapshot {selected_year}")

    core = ["median_income", "poverty_rate", "rent_burden_30_plus", "median_home_value"]

    cols = st.columns(4)
    for i, mk in enumerate(core):
        if mk not in catalog:
            continue
        snap = get_gateway_metric_snapshot(primary_fips, mk, selected_year)
        if snap is None or snap.empty:
            cols[i].metric(catalog[mk]["metric_label"], "—")
        else:
            v = snap["value"].iloc[0]
            d = snap.get("delta_5yr", pd.Series([None])).iloc[0]
            cols[i].metric(
                catalog[mk]["metric_label"],
                fmt_value(v, catalog[mk]),
                fmt_delta(d, catalog[mk])
            )

    # Trend
    st.subheader("Trend")

    trend_metric = st.selectbox("Metric", list(catalog.keys()), format_func=lambda k: catalog[k]["metric_label"])

    city_tr = get_gateway_metric_trend(primary_fips, trend_metric)
    state_tr = get_state_metric_trend(trend_metric)

    if city_tr is not None and not city_tr.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=city_tr["year"], y=city_tr["value"], mode="lines", name="City"))
        if state_tr is not None and not state_tr.empty:
            fig.add_trace(go.Scatter(x=state_tr["year"], y=state_tr["value"], mode="lines", name="MA"))
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

        if MODE in ("Investigative", "Academic"):
            diag = compute_trend_diagnostics(city_tr)
            c1, c2, c3 = st.columns(3)
            if diag.delta_5yr is not None:
                fmt_delta(diag.delta_5yr, catalog[trend_metric])
            if diag.slope_5yr is not None:
                c2.metric("Slope 5yr", f"{diag.slope_5yr:+.3f}")
            if diag.slope_10yr is not None:
                c3.metric("Slope 10yr", f"{diag.slope_10yr:+.3f}")

    # Ranking
    st.subheader("Gateway Position")

    rank_df = get_gateway_ranking(trend_metric, selected_year)
    if rank_df is not None and not rank_df.empty:
        ctx = compute_distribution_context(rank_df, primary_fips)
        c1, c2, c3 = st.columns(3)
        if ctx.rank is not None:
            c1.metric("Rank", f"{ctx.rank}/{ctx.n}")
        if ctx.z is not None:
            c2.metric("Z-score", f"{ctx.z:+.2f}")
        if ctx.mean is not None and ctx.value is not None:
            c3.metric("Gap vs Mean", f"{ctx.value - ctx.mean:+.2f}")

        if MODE in ("Investigative", "Academic"):
            st.dataframe(rank_df, use_container_width=True)

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
            index=max(0, cities_df["place_name"].tolist().index(primary_city))
            if primary_city in cities_df["place_name"].tolist() else 0,
            key="story_city",
        )

        place_fips = str(
            cities_df.loc[cities_df["place_name"] == story_city, "place_fips"].iloc[0]
        )

        angle_key = st.selectbox(
            "Theme",
            list(STORY_ANGLES.keys()),
            format_func=lambda k: STORY_ANGLES[k]["title"],
            key="angle_key",
        )

        angle = STORY_ANGLES[angle_key]

        investigate = st.toggle(
            "🔍 Explore Relationships",
            value=(MODE in ("Investigative", "Academic")),
            key="investigate_toggle",
        )

        st.subheader(angle["title"])

        metrics = angle.get("metrics", [])

        # =========================
        # SNAPSHOT GRID
        # =========================
        if metrics:
            show_metrics = metrics[:4] if MODE == "Public" else metrics
            cols = st.columns(min(len(show_metrics), 6))

            for i, mk in enumerate(show_metrics):
                meta = catalog.get(mk, {"metric_label": mk})
                snap = get_gateway_metric_snapshot(place_fips, mk, selected_year)

                if snap is None or snap.empty:
                    cols[i % len(cols)].metric(meta["metric_label"], "—")
                    continue

                v = snap["value"].iloc[0]
                d5 = snap.get("delta_5yr", pd.Series([None])).iloc[0]

                cols[i % len(cols)].metric(
                    meta["metric_label"],
                    fmt_value(v, meta),
                    fmt_delta(d5, meta),
                )
        else:
            st.info("No metrics configured for this theme.")

        st.divider()

        # =========================
        # TREND SECTION
        # =========================
        lead_metric = st.selectbox(
            "Trend focus metric",
            metrics if metrics else list(catalog.keys()),
            format_func=lambda k: catalog.get(k, {}).get("metric_label", k),
            key="lead_metric",
        )

        city_trend = get_gateway_metric_trend(place_fips, lead_metric)
        ma_trend = get_state_metric_trend(lead_metric)

        meta = catalog.get(lead_metric, {"metric_label": lead_metric})
        st.markdown(f"**Trend:** {meta['metric_label']}")

        if city_trend is None or city_trend.empty:
            st.info("No trend data available.")
        else:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=city_trend["year"],
                    y=city_trend["value"],
                    mode="lines",
                    name=story_city.split(",")[0],
                )
            )

            if ma_trend is not None and not ma_trend.empty:
                fig.add_trace(
                    go.Scatter(
                        x=ma_trend["year"],
                        y=ma_trend["value"],
                        mode="lines",
                        name="Massachusetts",
                    )
                )

            fig.update_layout(
                template="plotly_white",
                height=430,
                xaxis_title="Year",
                yaxis_title=meta.get("unit", ""),
            )

            st.plotly_chart(fig, use_container_width=True)

            # ---- Diagnostics (schema-locked) ----
            if MODE in ("Investigative", "Academic"):
                diag = compute_trend_diagnostics(city_trend)

                c1, c2, c3 = st.columns(3)

                if diag.delta_5yr is not None:
                    c1.metric("5-year change", fmt_delta(diag.delta_5yr, meta))

                if diag.slope_5yr is not None:
                    c2.metric("Slope (5yr window)", f"{diag.slope_5yr:+.3g} / year")

                if diag.slope_10yr is not None:
                    c3.metric("Slope (10yr window)", f"{diag.slope_10yr:+.3g} / year")

        # =========================
        # RANKING SECTION
        # =========================
        st.divider()
        st.markdown(f"### Gateway Ranking — {meta['metric_label']} ({selected_year})")

        rank_df = get_gateway_ranking(lead_metric, selected_year)

        if rank_df is None or rank_df.empty:
            st.info("No ranking data available.")
        else:
            if MODE in ("Investigative", "Academic"):
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
            else:
                one = rank_df[rank_df["place_fips"].astype(str) == str(place_fips)]
                if not one.empty:
                    st.dataframe(one, use_container_width=True, hide_index=True)
                else:
                    st.info("Selected city not found in ranking output.")

        # =========================
        # SCATTER COMPARISONS
        # =========================
        if investigate and MODE in ("Investigative", "Academic"):

            st.divider()
            st.subheader("Investigative comparisons (Gateway Cities only)")

            pairs = angle.get("investigative_pairs", [])

            if not pairs:
                st.info("No investigative pairs configured.")
            else:
                pair_labels = []
                for xk, yk in pairs:
                    xl = catalog.get(xk, {}).get("metric_label", xk)
                    yl = catalog.get(yk, {}).get("metric_label", yk)
                    pair_labels.append(f"{xl} vs {yl}")

                idx = st.selectbox(
                    "Choose a comparison",
                    range(len(pairs)),
                    format_func=lambda i: pair_labels[i],
                    key="pair_idx",
                )

                xk, yk = pairs[idx]

                sc = get_gateway_scatter(xk, yk, selected_year)

                if sc is None or sc.empty:
                    st.info("No scatter data available.")
                else:
                    sc = sc.copy()
                    sc["is_selected"] = (
                        sc["place_fips"].astype(str) == str(place_fips)
                    )

                    stats = compute_scatter_stats(sc)

                    xl = catalog.get(xk, {}).get("metric_label", xk)
                    yl = catalog.get(yk, {}).get("metric_label", yk)

                    fig_sc = px.scatter(
                        sc,
                        x="x",
                        y="y",
                        hover_name="place_name",
                        title=f"{selected_year}: {xl} (x) vs {yl} (y)",
                    )

                    if MODE == "Academic" and stats.slope is not None:
                        xx = np.linspace(np.nanmin(sc["x"]), np.nanmax(sc["x"]), 50)
                        yy = stats.slope * xx + stats.intercept
                        fig_sc.add_trace(
                            go.Scatter(x=xx, y=yy, mode="lines", name="OLS fit")
                        )

                    sel = sc[sc["is_selected"]]
                    if not sel.empty:
                        fig_sc.add_trace(
                            go.Scatter(
                                x=sel["x"],
                                y=sel["y"],
                                mode="markers",
                                marker=dict(size=16, symbol="diamond"),
                                name=story_city.split(",")[0],
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
            city: str(
                cities_all.loc[
                    cities_all["place_name"] == city,
                    "place_fips"
                ].iloc[0]
            )
            for city in compare_cities
        }

        metric_keys = sorted(list(catalog.keys()))
        metric_labels = {
            k: catalog[k].get("metric_label", k)
            for k in metric_keys
        }

        # =========================
        # MULTI-CITY TREND
        # =========================
        col_a, col_b = st.columns([1.2, 2.3])

        with col_a:
            metric_to_compare = st.selectbox(
                "Trend metric",
                metric_keys,
                index=metric_keys.index("median_income")
                if "median_income" in metric_keys else 0,
                format_func=lambda k: metric_labels[k],
                key="compare_metric",
            )

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
                        name=city_name.split(",")[0],
                    )
                )

            fig.update_layout(
                template="plotly_white",
                height=450,
                xaxis_title="Year",
                yaxis_title=catalog.get(metric_to_compare, {}).get("unit", ""),
                legend=dict(title=""),
            )

            st.plotly_chart(fig, use_container_width=True)

        # =========================
        # CROSS-METRIC SCATTER
        # =========================
        st.divider()
        st.markdown("### Cross-metric scatter (Gateway Cities)")

        c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

        with c1:
            metric_x = st.selectbox(
                "X metric",
                metric_keys,
                index=metric_keys.index("rent_burden_30_plus")
                if "rent_burden_30_plus" in metric_keys else 0,
                format_func=lambda k: metric_labels[k],
                key="scatter_x",
            )

        with c2:
            metric_y = st.selectbox(
                "Y metric",
                metric_keys,
                index=metric_keys.index("poverty_rate")
                if "poverty_rate" in metric_keys else min(1, len(metric_keys) - 1),
                format_func=lambda k: metric_labels[k],
                key="scatter_y",
            )

        with c3:
            sc_year = st.selectbox(
                "Year",
                available_years,
                index=available_years.index(selected_year),
                key="scatter_year",
            )

        sc = get_gateway_scatter(metric_x, metric_y, sc_year)

        if sc is None or sc.empty:
            st.info("No data available for that scatter combination.")
        else:
            sc = sc.copy()

            # ---- STRICT NUMERIC CAST ----
            sc["x"] = pd.to_numeric(sc["x"], errors="coerce")
            sc["y"] = pd.to_numeric(sc["y"], errors="coerce")
            sc = sc.dropna(subset=["x", "y"])

            if sc.empty:
                st.info("No valid numeric scatter data available.")
            else:
                sc["is_selected"] = sc["place_fips"].astype(str).isin(
                    set(selected_fips.values())
                )

                stats = compute_scatter_stats(sc)

                xl = metric_labels[metric_x]
                yl = metric_labels[metric_y]

                fig_sc = px.scatter(
                    sc,
                    x="x",
                    y="y",
                    hover_name="place_name",
                    title=f"{sc_year}: {xl} (x) vs {yl} (y)",
                )

                # ---- Highlight selected cities ----
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

                # ---- Regression (Academic only) ----
                if (
                    MODE == "Academic"
                    and stats.slope is not None
                    and stats.intercept is not None
                ):
                    x_min = float(sc["x"].min())
                    x_max = float(sc["x"].max())

                    if x_min != x_max:
                        xx = np.linspace(x_min, x_max, 50)
                        yy = stats.slope * xx + stats.intercept

                        fig_sc.add_trace(
                            go.Scatter(
                                x=xx,
                                y=yy,
                                mode="lines",
                                name="OLS fit",
                            )
                        )

                fig_sc.update_layout(
                    template="plotly_white",
                    height=560,
                )

                st.plotly_chart(fig_sc, use_container_width=True)

                # ---- Stats panel ----
                if MODE in ("Investigative", "Academic"):
                    s1, s2, s3 = st.columns(3)

                    if stats.r is not None:
                        s1.metric("Correlation r", f"{stats.r:+.2f}")

                    if stats.r2 is not None:
                        s2.metric("R²", f"{stats.r2:.2f}")

                    if stats.n is not None:
                        s3.metric("N (cities)", f"{stats.n}")

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

    # Remove regional aggregates
    bad_terms = {
        "Total",
        "Europe", "Asia", "Africa", "Oceania",
        "Latin America", "Northern America",
        "Other", "Other areas",
        "Other Europe", "Other Asia", "Other Africa",
        "Other Oceania", "Other Latin America",
        "Other Northern America",
        "Other and unspecified",
        "Other and unspecified areas",
    }

    if last in bad_terms:
        return None
    if last.lower().startswith("other"):
        return None
    if "total foreign-born" in last.lower():
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
            "Uses raw B05006 (place of birth). "
            "Country parsing is heuristic; adjust rules if ETL naming differs."
        )

        cities_df = get_cities(gateway_only=True)

        origins_city = st.selectbox(
            "City for origins map",
            cities_df["place_name"].tolist(),
            index=max(0, cities_df["place_name"].tolist().index(primary_city))
            if primary_city in cities_df["place_name"].tolist() else 0,
            key="origins_city",
        )

        origins_fips = str(
            cities_df.loc[
                cities_df["place_name"] == origins_city,
                "place_fips"
            ].iloc[0]
        )

        year = st.selectbox(
            "Year",
            available_years,
            index=available_years.index(selected_year),
            key="origins_year",
        )

        try:
            df_b05006 = get_place_source_table_year(
                origins_fips,
                SOURCE_BIRTH_TABLE,
                int(year),
            )
        except Exception:
            df_b05006 = None

        if df_b05006 is None or df_b05006.empty:
            st.info("No B05006 rows returned for this city/year.")
        else:
            df = df_b05006.copy()

            # ---- Ensure required columns exist ----
            if "estimate" not in df.columns or "variable_label" not in df.columns:
                st.warning("B05006 table missing required columns (estimate / variable_label).")
            else:
                # ---- Clean numeric ----
                df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
                df = df.dropna(subset=["estimate"])

                if df.empty:
                    st.info("No numeric B05006 values available.")
                else:
                    # ---- Extract country ----
                    df["country_name"] = df["variable_label"].apply(label_to_country_name)
                    df = df.dropna(subset=["country_name"])

                    if df.empty:
                        st.info("No country-level rows detected in B05006.")
                    else:
                        df["iso3"] = df["country_name"].apply(country_to_iso3)

                        # ---- Detect total foreign-born safely ----
                        total_fb = None
                        try:
                            mask_total = df["variable_label"].str.contains(
                                "Total foreign-born",
                                case=False,
                                na=False,
                            )
                            if mask_total.any():
                                total_fb = float(
                                    df.loc[mask_total, "estimate"].sum()
                                )
                        except Exception:
                            total_fb = None

                        # ---- Aggregate for map ----
                        df_map = (
                            df.dropna(subset=["iso3"])
                            .groupby("iso3", as_index=False)["estimate"]
                            .sum()
                        )

                        if df_map.empty:
                            st.warning("No mappable sovereign countries found.")
                        else:
                            color_col = "estimate"
                            labels = {"estimate": "Foreign-born (estimate)"}

                            if (
                                MODE in ("Academic", "Investigative")
                                and total_fb is not None
                                and total_fb > 0
                            ):
                                df_map["share"] = (
                                    df_map["estimate"] / total_fb
                                ) * 100
                                color_col = "share"
                                labels = {"share": "Share of foreign-born (%)"}

                            fig_world = px.choropleth(
                                df_map,
                                locations="iso3",
                                color=color_col,
                                title=f"Foreign-Born Population by Country — {origins_city} ({year})",
                                labels=labels,
                            )

                            fig_world.update_layout(
                                template="plotly_white",
                                height=560,
                                margin=dict(l=10, r=10, t=60, b=10),
                            )

                            st.plotly_chart(fig_world, use_container_width=True)

                            # ---- Top origins table ----
                            st.markdown("#### Top mappable origins")

                            top = (
                                df.dropna(subset=["iso3"])
                                .groupby("country_name", as_index=False)["estimate"]
                                .sum()
                                .sort_values("estimate", ascending=False)
                                .head(20)
                            )

                            if (
                                MODE in ("Academic", "Investigative")
                                and total_fb is not None
                                and total_fb > 0
                            ):
                                top["share_%"] = (
                                    top["estimate"] / total_fb
                                ) * 100

                            st.dataframe(
                                top,
                                use_container_width=True,
                                hide_index=True,
                            )
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
### Data Source
- American Community Survey (ACS) **5-year estimates**
- Endpoint year convention: e.g., “2022” represents the 2018–2022 pooled estimate.
- Data accessed via warehouse queries:
  - `get_gateway_metric_snapshot`
  - `get_gateway_metric_trend`
  - `get_gateway_ranking`
  - `get_gateway_scatter`

### Interpretation of Time Series
- 5-year ACS reduces sampling volatility.
- Values should be interpreted as rolling-window aggregates, not point-in-time observations.
- Trend slopes are computed via simple OLS on `(year, value)` pairs.

### Ranking & Distribution Context
- Rankings use `rank_within_gateway` returned from the query layer.
- Distribution metrics are computed across Gateway cities for a given year.
- Z-score formula:
  \[
  z = \frac{(value - \mu_{gateway})}{\sigma_{gateway}}
  \]
- No population weighting applied unless enforced at the query level.

### Scatter & Regression
- Cross-sectional comparisons use Gateway cities only.
- Pearson correlation coefficient (r) is computed on numeric `(x, y)` pairs.
- OLS regression line (if shown) is unweighted:
  \[
  y = \beta_0 + \beta_1 x
  \]
- These represent **associations**, not causal inference.

### Warehouse Integrity Checklist
- Each `metric_key` in `metric_catalog` must map to a populated series in `gateway_metrics`.
- Confirm:
  - `year`
  - `value`
  - `rank_within_gateway`
  - `delta_5yr`
- Ensure units and formatting hints in `metric_catalog` are consistent with warehouse data.
- For B05006 origins:
  - Verify `variable_label` naming structure
  - Confirm `estimate` column integrity
  - Validate country parsing logic if ETL format changes
                """
            )