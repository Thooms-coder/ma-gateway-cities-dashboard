from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

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
    get_gateway_metric_snapshot,
    get_gateway_metric_trend,
    get_state_metric_trend,
    get_gateway_ranking,
    get_gateway_scatter,
    get_available_gateway_years,
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Gateway Cities Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# LOAD CSS
# ==================================================

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
<div class="hero">
    <h1>Gateway Cities Dashboard</h1>
    <div class="accent-line"></div>
    <p>
        A data-driven examination of economic pressure, housing burden,
        and structural inequality across Massachusetts Gateway Cities.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# HELPERS
# ==================================================

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
# ANALYTICS
# ==================================================

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

if not available_years:
    st.error("No available years returned from warehouse.")
    st.stop()

selected_year = st.selectbox("Year", available_years)
advanced = st.toggle("Advanced Analysis", value=False)

primary_city = st.selectbox("City", gateway_city_options)
primary_fips = cities_all.loc[cities_all["place_name"] == primary_city, "place_fips"].iloc[0]

# ==================================================
# TABS
# ==================================================

tab_map, tab_compare, tab_origins, tab_method = st.tabs(
    ["Overview", "Compare", "Origins", "Methodology"]
)

# ==================================================
# TAB 1 — OVERVIEW
# ==================================================

with tab_map:
    st.subheader("Snapshot")

    core = ["median_income", "poverty_rate", "rent_burden_30_plus", "median_home_value"]
    cols = st.columns(4)

    for i, mk in enumerate(core):
        if mk not in catalog:
            continue

        snap = get_gateway_metric_snapshot(primary_fips, mk, selected_year)

        if snap is None or snap.empty:
            continue

        v = snap["value"].iloc[0]
        label = catalog[mk]["metric_label"]
        value = fmt_value(v, catalog[mk])

        cols[i].markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Trend")

    metric = st.selectbox(
        "Metric",
        list(catalog.keys()),
        format_func=lambda k: catalog[k]["metric_label"],
    )

    city_tr = get_gateway_metric_trend(primary_fips, metric)
    state_tr = get_state_metric_trend(metric)

    if city_tr is not None and not city_tr.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=city_tr["year"], y=city_tr["value"], mode="lines", name="City"))
        if state_tr is not None and not state_tr.empty:
            fig.add_trace(go.Scatter(x=state_tr["year"], y=state_tr["value"], mode="lines", name="MA"))
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 2 — COMPARE
# ==================================================

with tab_compare:
    st.subheader("Cross-metric scatter")

    metric_x = st.selectbox("X metric", list(catalog.keys()), format_func=lambda k: catalog[k]["metric_label"])
    metric_y = st.selectbox("Y metric", list(catalog.keys()), format_func=lambda k: catalog[k]["metric_label"])

    sc = get_gateway_scatter(metric_x, metric_y, selected_year)

    if sc is not None and not sc.empty:
        sc["x"] = pd.to_numeric(sc["x"], errors="coerce")
        sc["y"] = pd.to_numeric(sc["y"], errors="coerce")
        sc = sc.dropna(subset=["x", "y"])

        stats = compute_scatter_stats(sc)

        fig_sc = px.scatter(sc, x="x", y="y", hover_name="place_name")
        if advanced and stats.slope is not None:
            xx = np.linspace(sc["x"].min(), sc["x"].max(), 50)
            yy = stats.slope * xx + stats.intercept
            fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="OLS"))

        st.plotly_chart(fig_sc, use_container_width=True)

# ==================================================
# TAB 3 — ORIGINS
# ==================================================

with tab_origins:
    st.subheader("Foreign-born origins")

    df = get_place_source_table_year(primary_fips, "B05006", int(selected_year))

    if df is not None and not df.empty and "estimate" in df.columns:
        df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
        df = df.dropna(subset=["estimate"])

        st.dataframe(df.head(20), use_container_width=True)

# ==================================================
# TAB 4 — METHODOLOGY
# ==================================================

with tab_method:
    st.markdown("""
**Data Source**  
ACS 5-year estimates.

**Interpretation**  
Rolling window data, not point-in-time.

**Statistics**  
OLS and Pearson correlation are descriptive, not causal.
""")