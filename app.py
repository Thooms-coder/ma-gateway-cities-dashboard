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
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Display Options")

    show_income = st.toggle("Income Trend", value=False)
    show_poverty = st.toggle("Poverty Trend", value=False)
    show_markers = st.toggle("Markers", value=True)
    smooth_lines = st.toggle("Smooth Lines", value=False)

# --------------------------------------------------
# Hero Section (Fade-In)
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>Gateway Cities</h1>
    <div class="accent-line"></div>
    <p>
    A longitudinal investigation of immigration patterns,
    economic transformation, and structural inequality across Massachusetts.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Animated Content Section
# --------------------------------------------------

st.markdown('<div class="section">', unsafe_allow_html=True)

# --------------------------------------------------
# City Selector
# --------------------------------------------------

cities = get_cities(gateway_only=False)

col1, col2 = st.columns([2, 1])

with col1:
    selected_city = st.selectbox(
        "Select City",
        cities["place_name"],
        label_visibility="collapsed"
    )

place_fips = cities[cities["place_name"] == selected_city]["place_fips"].values[0]

# --------------------------------------------------
# Foreign-Born Trend
# --------------------------------------------------

df_fb = get_foreign_born_percent(place_fips)

latest_percent = df_fb["foreign_born_percent"].iloc[-1]
growth = (
    (df_fb["foreign_born_percent"].iloc[-1] - df_fb["foreign_born_percent"].iloc[0])
    / df_fb["foreign_born_percent"].iloc[0]
) * 100

# Metric Row
m1, m2 = st.columns(2)

m1.metric("Current Foreign-Born %", f"{latest_percent:.1f}%")
m2.metric("Growth Since Start", f"{growth:.1f}%")

# Chart
fig_fb = px.line(
    df_fb,
    x="year",
    y="foreign_born_percent",
    markers=show_markers,
)

if smooth_lines:
    fig_fb.update_traces(line_shape="spline")

fig_fb.update_layout(
    template="plotly_white",
    title=f"Foreign-Born Population (%) — {selected_city}",
    font=dict(family="Inter"),
    title_font=dict(size=20),
    margin=dict(l=20, r=20, t=60, b=20),
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig_fb, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

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
        template="plotly_white",
        title=f"Median Household Income — {selected_city}",
        font=dict(family="Inter"),
        title_font=dict(size=18),
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_income, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
        template="plotly_white",
        title=f"Poverty Rate — {selected_city}",
        font=dict(family="Inter"),
        title_font=dict(size=18),
    )

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_poverty, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Editorial Insight Block
# --------------------------------------------------

st.markdown("""
### Investigative Insight

Use this interface to explore how foreign-born population growth aligns
with economic conditions across time. Toggle contextual layers to identify
divergence, acceleration, or structural shifts.
""")

st.markdown('</div>', unsafe_allow_html=True)