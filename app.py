import streamlit as st
from st_aggrid import AgGrid
import plotly.graph_objects as go
import json
import pandas as pd
import re
import textwrap

from src.queries import (
    get_cities,
    get_gateway_fips,
    get_foreign_born_percent,
    get_income_trend,
    get_poverty_trend
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LOAD GEOJSON
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

# --------------------------------------------------
# LOAD CITY DATA
# --------------------------------------------------

cities = get_cities(gateway_only=False)
cities["place_fips"] = cities["place_fips"].astype(str)

gateway_fips = set(get_gateway_fips()["place_fips"].astype(str))

# Dropdown = gateway only
city_options = (
    cities[cities["place_fips"].isin(gateway_fips)]
    ["place_name"]
    .sort_values()
    .tolist()
)

if not city_options:
    st.error("No gateway cities found.")
    st.stop()

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
    city: str(
        cities[cities["place_name"] == city]["place_fips"].values[0]
    )
    for city in selected_cities
}

primary_city = selected_cities[0]
primary_fips = selected_fips[primary_city]

# --------------------------------------------------
# LOAD TIME SERIES DATA (SAFE)
# --------------------------------------------------

city_data = {}

for city, fips in selected_fips.items():

    # -----------------------
    # Foreign Born %
    # -----------------------
    df_fb = get_foreign_born_percent(fips)
    if not df_fb.empty:
        df_fb["year"] = pd.to_numeric(df_fb["year"], errors="coerce")
        df_fb.dropna(subset=["year"], inplace=True)
        df_fb.sort_values("year", inplace=True)
    else:
        df_fb = pd.DataFrame(columns=["year", "foreign_born_percent"])

    # -----------------------
    # Income
    # -----------------------
    df_income = get_income_trend(fips)
    if not df_income.empty:
        df_income["year"] = pd.to_numeric(df_income["year"], errors="coerce")
        df_income.dropna(subset=["year"], inplace=True)
        df_income.sort_values("year", inplace=True)
    else:
        df_income = pd.DataFrame(columns=["year", "median_income"])

    # -----------------------
    # Poverty
    # -----------------------
    df_poverty = get_poverty_trend(fips)
    if not df_poverty.empty:
        df_poverty["year"] = pd.to_numeric(df_poverty["year"], errors="coerce")
        df_poverty.dropna(subset=["year"], inplace=True)
        df_poverty.sort_values("year", inplace=True)
    else:
        df_poverty = pd.DataFrame(columns=["year", "poverty_rate"])

    # -----------------------
    # Strict Overlap Dataset
    # -----------------------
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
# MAP SECTION
# --------------------------------------------------

town_fips_map = {
    normalize(clean_place_label(name)): fips
    for name, fips in zip(cities["place_name"], cities["place_fips"])
}

locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]

def build_map():
    z_values = []
    selected_index = None

    for i, town_name in enumerate(locations):
        town_norm = normalize(town_name)

        if (
            town_norm in town_fips_map
            and town_fips_map[town_norm] == primary_fips
        ):
            z_values.append(2)
            selected_index = i
        elif town_norm in town_fips_map and town_fips_map[town_norm] in gateway_fips:
            z_values.append(1)
        else:
            z_values.append(0)

    fig = go.Figure(go.Choroplethmapbox(
        geojson=ma_geo,
        locations=locations,
        z=z_values,
        featureidkey="properties.TOWN",
        colorscale=[
            [0, "#E9ECEF"],
            [0.5, "#dc3220"],
            [1, "#005ab5"]
        ],
        zmin=0,
        zmax=2,
        showscale=False,
        hovertemplate="<b>%{location}</b><extra></extra>"
    ))

    fig.update_layout(
        mapbox=dict(
            style="white-bg",
            center=dict(lat=42.3, lon=-71.8),
            zoom=7.8
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )

    return fig

st.plotly_chart(build_map(), use_container_width=True)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

df_primary_fb = city_data[primary_city]["fb"]

if not df_primary_fb.empty and len(df_primary_fb) > 1:

    latest_percent = df_primary_fb["foreign_born_percent"].iloc[-1]
    start_val = df_primary_fb["foreign_born_percent"].iloc[0]
    growth = ((latest_percent - start_val) / start_val) * 100 if start_val != 0 else 0

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Foreign-Born Population", f"{latest_percent:.1f}%")
        st.metric("Period Growth", f"{growth:.1f}%")

    with col2:
        st.markdown(
            f"""
            Over the observed period, the foreign-born population in **{primary_city}**
            has changed by **{growth:.1f}%**, now representing
            **{latest_percent:.1f}%** of the community.
            """
        )

# --------------------------------------------------
# INCOME & POVERTY CHARTS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    fig_inc = go.Figure()
    for city, data in city_data.items():
        if not data["income"].empty:
            fig_inc.add_trace(go.Scatter(
                x=data["income"]["year"],
                y=data["income"]["median_income"],
                mode="lines",
                name=city
            ))
    fig_inc.update_layout(
        title="Median Household Income",
        template="plotly_white"
    )
    st.plotly_chart(fig_inc, use_container_width=True)

with col2:
    fig_pov = go.Figure()
    for city, data in city_data.items():
        if not data["poverty"].empty:
            fig_pov.add_trace(go.Scatter(
                x=data["poverty"]["year"],
                y=data["poverty"]["poverty_rate"],
                mode="lines",
                name=city
            ))
    fig_pov.update_layout(
        title="Poverty Rate",
        template="plotly_white"
    )
    st.plotly_chart(fig_pov, use_container_width=True)

# --------------------------------------------------
# TRAJECTORY
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
# TABLES
# --------------------------------------------------

st.markdown(f"### Data for {primary_city}")

AgGrid(city_data[primary_city]["fb"])
AgGrid(city_data[primary_city]["poverty"])
AgGrid(city_data[primary_city]["struct"])