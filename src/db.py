import os
import streamlit as st
from sqlalchemy import create_engine

@st.cache_resource
def get_engine():
    DATABASE_URL = os.getenv("DATABASE_URL")

    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,      # verifies connection before use
        pool_recycle=300,        # recycle every 5 min
        pool_size=5,             # small stable pool
        max_overflow=2,          # small burst buffer
    )

engine = get_engine()