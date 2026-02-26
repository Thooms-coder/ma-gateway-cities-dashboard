import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import AgGrid
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd
import re
import textwrap

from src.queries import (
    get_cities,
    get_gateway_fips,
    get_place_variable_trend
)

# --------------------------------------------------
# ACS Variable IDs (Warehouse-Controlled)
# --------------------------------------------------

VAR_FOREIGN_BORN_TOTAL = "B05002_013E"
VAR_TOTAL_POP = "B01003_001E"
VAR_MEDIAN_INCOME = "S1901_C01_012E"
VAR_POVERTY_RATE = "S1701_C03_001E"

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Load Cities
# --------------------------------------------------

cities = get_cities(gateway_only=False)
gateway_fips = set(get_gateway_fips()["place_fips"])

cities["place_fips"] = cities["place_fips"].astype(str)

# --------------------------------------------------
# City Selection
# --------------------------------------------------

city_options = (
    cities[cities["place_fips"].isin(gateway_fips)]["place_name"]
    .sort_values()
    .tolist()
)

if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = [city_options[0]]

selected_cities = st.multiselect(
    "Compare Municipalities (Max 3)",
    options=city_options,
    default=st.session_state.selected_cities,
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

# --------------------------------------------------
# Data Loading
# --------------------------------------------------

city_data = {}

for city, fips in selected_fips.items():

    # --------------------------------
    # Foreign Born %
    # --------------------------------
    df_fb_total = get_place_variable_trend(fips, VAR_FOREIGN_BORN_TOTAL)
    df_total_pop = get_place_variable_trend(fips, VAR_TOTAL_POP)

    if not df_fb_total.empty and not df_total_pop.empty:
        df_fb = (
            df_fb_total
            .merge(df_total_pop, on="acs_end_year", suffixes=("_fb", "_pop"))
        )

        df_fb["year"] = df_fb["acs_end_year"]
        df_fb["foreign_born_percent"] = (
            df_fb["estimate_fb"] / df_fb["estimate_pop"]
        ) * 100

        df_fb = df_fb[["year", "foreign_born_percent"]].sort_values("year")
    else:
        df_fb = pd.DataFrame(columns=["year", "foreign_born_percent"])


    # --------------------------------
    # Income
    # --------------------------------
    df_income_raw = get_place_variable_trend(fips, VAR_MEDIAN_INCOME)

    if not df_income_raw.empty:
        df_income = df_income_raw.rename(columns={
            "acs_end_year": "year",
            "estimate": "median_income"
        })[["year", "median_income"]]
    else:
        df_income = pd.DataFrame(columns=["year", "median_income"])


    # --------------------------------
    # Poverty
    # --------------------------------
    df_poverty_raw = get_place_variable_trend(fips, VAR_POVERTY_RATE)

    if not df_poverty_raw.empty:
        df_poverty = df_poverty_raw.rename(columns={
            "acs_end_year": "year",
            "estimate": "poverty_rate"
        })[["year", "poverty_rate"]]
    else:
        df_poverty = pd.DataFrame(columns=["year", "poverty_rate"])


    # --------------------------------
    # Strict Overlap (Safe Merge)
    # --------------------------------
    if not df_fb.empty and not df_income.empty and not df_poverty.empty:
        df_struct = (
            df_fb
            .merge(df_income, on="year", how="inner")
            .merge(df_poverty, on="year", how="inner")
            .sort_values("year")
            .reset_index(drop=True)
        )
    else:
        df_struct = pd.DataFrame(
            columns=["year", "foreign_born_percent", "median_income", "poverty_rate"]
        )

    city_data[city] = {
        "fb": df_fb,
        "income": df_income,
        "poverty": df_poverty,
        "struct": df_struct
    }

# --------------------------------------------------
# SECTION: KPI + Narrative
# --------------------------------------------------

df_primary_fb = city_data.get(primary_city, {}).get("fb", pd.DataFrame())

if not df_primary_fb.empty:

    latest_percent = df_primary_fb["foreign_born_percent"].iloc[-1]
    start_val = df_primary_fb["foreign_born_percent"].iloc[0]
    growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else 0

    col_kpi, col_text = st.columns([1, 2])

    with col_kpi:
        st.metric("Foreign-Born Population", f"{latest_percent:.1f}%")
        st.metric("Period Growth", f"{growth:.1f}%")

    with col_text:
        trend_word = "surged" if growth > 10 else "grown" if growth > 0 else "declined"
        st.markdown(f"""
        Over the observed period, the foreign-born population in **{primary_city}**
        has {trend_word} by **{abs(growth):.1f}%**, now representing
        **{latest_percent:.1f}%** of the community.
        """)

# --------------------------------------------------
# SECTION: Income & Poverty
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    fig_inc = go.Figure()
    for city, data in city_data.items():
        df = data["income"]
        if not df.empty:
            fig_inc.add_trace(go.Scatter(
                x=df["year"],
                y=df["median_income"],
                mode="lines",
                name=city
            ))
    fig_inc.update_layout(
        title="Median Household Income",
        template="plotly_white",
        yaxis_title="Income ($)"
    )
    st.plotly_chart(fig_inc, use_container_width=True)

with col2:
    fig_pov = go.Figure()
    for city, data in city_data.items():
        df = data["poverty"]
        if not df.empty:
            fig_pov.add_trace(go.Scatter(
                x=df["year"],
                y=df["poverty_rate"],
                mode="lines",
                name=city
            ))
    fig_pov.update_layout(
        title="Poverty Rate",
        template="plotly_white",
        yaxis_title="Poverty Rate (%)"
    )
    st.plotly_chart(fig_pov, use_container_width=True)

# --------------------------------------------------
# SECTION: Trajectory
# --------------------------------------------------

fig_traj = go.Figure()

for city, data in city_data.items():
    df = data["struct"]
    if not df.empty and len(df) > 1:
        fig_traj.add_trace(go.Scatter(
            x=df["foreign_born_percent"],
            y=df["median_income"],
            mode="markers",
            name=city,
            text=df["year"]
        ))

fig_traj.update_layout(
    title="Income vs Foreign-Born Population",
    xaxis_title="Foreign-Born (%)",
    yaxis_title="Median Income ($)",
    template="plotly_white"
)

st.plotly_chart(fig_traj, use_container_width=True)

# --------------------------------------------------
# SECTION: Tables
# --------------------------------------------------

st.markdown(f"### Data for {primary_city}")

AgGrid(city_data[primary_city]["fb"])
AgGrid(city_data[primary_city]["poverty"])
AgGrid(city_data[primary_city]["struct"])