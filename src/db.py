import streamlit as st
from sqlalchemy import create_engine

# Use Streamlit secrets (Cloud) or fallback to local secrets
DATABASE_URL = st.secrets["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)