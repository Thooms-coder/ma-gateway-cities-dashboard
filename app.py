import streamlit as st
import plotly.express as px
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
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Load Custom CSS
# --------------------------------------------------

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# Sidebar Customization Panel
# --------------------------------------------------

with st.sidebar:
    st.header("Customize View")

    show_income = st.toggle("Show Income Trend", value=False)
    show_poverty = st.toggle("Show Poverty Trend", value=False)
    show_markers = st.toggle("Show Markers", value=True)
    smooth_lines = st.toggle("Smooth Lines", value=False)

# --------------------------------------------------
# Hero Landing Section
# --------------------------------------------------

st.markdown("""
<div class="hero fade-in">
    <h1>Gateway Cities</h1>
    <p>
    A longitudinal investigation of immigration patterns,
    economic transformation, and structural inequality across Massachusetts.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section fade-in">', unsafe_allow_html=True)

# --------------------------------------------------
# City Selector
# --------------------------------------------------

cities = get_cities(gateway_only=False)

selected_city = st.selectbox(
    "Select a City",
    cities["place_name"]
)

place_fips = cities[cities["place_name"] == selected_city]["place_fips"].values[0]

# --------------------------------------------------
# Foreign-Born Trend
# --------------------------------------------------

df_fb = get_foreign_born_percent(place_fips)

fig_fb = px.line(
    df_fb,
    x="year",
    y="foreign_born_percent",
    markers=show_markers,
)

if smooth_lines:
    fig_fb.update_traces(line_shape="spline")

fig_fb.update_layout(
    template="plotly_dark",
    title=f"Foreign-Born Population (%) — {selected_city}",
    font=dict(family="Inter"),
    title_font=dict(size=24),
    hoverlabel=dict(bgcolor="#111111"),
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig_fb, use_container_width=True)

# --------------------------------------------------
# Optional Economic Context
# --------------------------------------------------

if show_income:
    df_income = get_income_trend(place_fips)

    fig_income = px.line(
        df_income,
        x="year",
        y="median_income",
        markers=show_markers,
    )

    if smooth_lines:
        fig_income.update_traces(line_shape="spline")

    fig_income.update_layout(
        template="plotly_dark",
        title=f"Median Household Income — {selected_city}",
        font=dict(family="Inter"),
        title_font=dict(size=22),
    )

    st.plotly_chart(fig_income, use_container_width=True)

if show_poverty:
    df_poverty = get_poverty_trend(place_fips)

    fig_poverty = px.line(
        df_poverty,
        x="year",
        y="poverty_rate",
        markers=show_markers,
    )

    if smooth_lines:
        fig_poverty.update_traces(line_shape="spline")

    fig_poverty.update_layout(
        template="plotly_dark",
        title=f"Poverty Rate — {selected_city}",
        font=dict(family="Inter"),
        title_font=dict(size=22),
    )

    st.plotly_chart(fig_poverty, use_container_width=True)

# --------------------------------------------------
# Narrative Insight Block
# --------------------------------------------------

st.markdown("""
### Investigative Insight

Use this interface to explore how foreign-born population growth aligns
with structural economic indicators. Toggle contextual layers in the sidebar
to surface correlations and divergences across time.
""")

st.markdown('</div>', unsafe_allow_html=True)