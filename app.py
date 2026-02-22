import streamlit as st
import plotly.express as px
from src.queries import get_foreign_born_trend

st.set_page_config(
    page_title="Gateway Cities Dashboard",
    layout="wide"
)

st.title("Gateway Cities – Foreign-Born Trends (2010–2024)")

city_fips = st.text_input("Enter City FIPS Code")

if city_fips:
    df = get_foreign_born_trend(city_fips)

    fig = px.line(
        df,
        x="acs_end_year",
        y="foreign_born_percent",
        markers=True,
        title="Foreign-Born % Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)
    
import streamlit as st
from sqlalchemy import text
from src.db import engine

st.title("Database Test")

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        st.success(f"Connected: {result.scalar()}")
except Exception as e:
    st.error(e)