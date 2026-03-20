from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import streamlit as st
from openai import OpenAI
from src.queries import (
    # registry / map
    get_cities,
    get_gateway_fips,
    # raw table (origins)
    get_place_source_table_year,
    # journalist layer
    get_metric_catalog,
    get_latest_year_available,  # ok if unused
    get_gateway_metric_snapshot,
    get_gateway_metric_trend,
    get_state_metric_trend,
    get_gateway_ranking,
    get_gateway_scatter,
    get_available_gateway_years,
)
from src.story_angles import STORY_ANGLES

# ==================================================
# CACHED QUERY WRAPPERS (performance)
# ==================================================

@st.cache_data(ttl=600)
def cached_gateway_metric_snapshot(place_fips, metric_key, year):
    return get_gateway_metric_snapshot(place_fips, metric_key, year)

@st.cache_data(ttl=600)
def cached_gateway_metric_trend(place_fips, metric_key):
    return get_gateway_metric_trend(place_fips, metric_key)

@st.cache_data(ttl=600)
def cached_state_metric_trend(metric_key):
    return get_state_metric_trend(metric_key)

@st.cache_data(ttl=600)
def cached_gateway_ranking(metric_key, year):
    return get_gateway_ranking(metric_key, year)

@st.cache_data(ttl=600)
def cached_gateway_scatter(metric_x, metric_y, year):
    return get_gateway_scatter(metric_x, metric_y, year)

# ==================================================
# CONFIG / CONSTANTS
# ==================================================

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

COLOR_TARGET = "#4A86C5"
COLOR_BASE = "#F28E8E"
COLOR_BG = "#f4f5f6"
COLOR_TEXT = "#2c2f33"
COLOR_BOSTON = "#5FB3A8"

# Curated, journalist-friendly list categorized by reporting beat
CENSUS_VARIABLES = """
### DEMOGRAPHICS & POPULATION
B01003_001E: Total population (count)
B01002_001E: Median age (years)
S0501_C02_001E: Foreign-born share (%)
B03002_003E: Non-Hispanic White population (count)
B03002_004E: Black or African American population (count)
B03002_006E: Asian population (count)
B03002_012E: Hispanic or Latino population (count)

### INCOME & INEQUALITY
B19013_001E: Median household income ($)
B19301_001E: Per capita income ($)
B19083_001E: Gini Index of Income Inequality (score 0 to 1)
S1701_C03_001E: Poverty rate (%)
S1701_C03_002E: Child poverty rate (%)
S1701_C03_010E: Senior (65+) poverty rate (%)

### HOUSING & LIVING ARRANGEMENTS
B25077_001E: Median home value ($)
B25064_001E: Median gross rent ($)
S2502_C01_013E: Renter share (%)
DP04_0142PE: Gross rent as 35% or more of household income (severe rent burden %)
B25002_003E: Vacant housing units (count)

### EDUCATION & WORK
S1501_C02_014E: High school graduate or higher share (%)
S1501_C02_015E: Bachelor's degree or higher share (%)
S2301_C04_001E: Unemployment rate (%)
S0801_C01_009E: Workers commuting by public transit (%)
S0801_C01_013E: Workers who worked from home (%)
""".strip()

# Expanded aliases mapping natural language to codes
VARIABLE_ALIASES = """
Aliases (use these in reasoning; output must still use codes):
- "population" / "total people" -> B01003_001E
- "median age" -> B01002_001E
- "foreign-born share" / "immigrant share" -> S0501_C02_001E
- "white population" -> B03002_003E
- "black population" -> B03002_004E
- "asian population" -> B03002_006E
- "hispanic population" / "latino population" -> B03002_012E
- "median income" -> B19013_001E
- "income per person" / "per capita income" -> B19301_001E
- "inequality" / "gini index" / "wealth gap" -> B19083_001E
- "poverty" / "poverty rate" -> S1701_C03_001E
- "child poverty" -> S1701_C03_002E
- "elderly poverty" / "senior poverty" -> S1701_C03_010E
- "median home value" / "house price" -> B25077_001E
- "median rent" -> B25064_001E
- "renter share" -> S2502_C01_013E
- "rent burden" / "housing cost burden" -> DP04_0142PE
- "vacant homes" / "vacancy" -> B25002_003E
- "high school diploma" -> S1501_C02_014E
- "college degree" / "bachelor's" / "highly educated" -> S1501_C02_015E
- "unemployment" / "jobless rate" -> S2301_C04_001E
- "public transit" / "commute by bus/train" -> S0801_C01_009E
- "work from home" / "remote workers" -> S0801_C01_013E
""".strip()
        
AGENT_SYSTEM_PROMPT = f"""
You are an AI data journalist assistant helping users explore a dashboard about Massachusetts Gateway Cities.
Use your tools to fetch real data or navigate the user to different tabs. 

When you answer, be concise, insightful, and cite the data you pulled. Do not overwhelm the user with raw Census codes in your final response—translate them back into plain English.

### DATA DICTIONARY
When a user asks about a topic, look up the correct Census code below and use it as the `metric_key` or `metric_x`/`metric_y` in your tool calls.

{CENSUS_VARIABLES}

{VARIABLE_ALIASES}
"""

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_dashboard",
            "description": "Changes the dashboard UI state to focus on a specific tab, city, or year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string", "enum": ["Map", "Investigative Themes", "Compare Metrics", "Origins (B05006)", "Ask the Data", "Methodology"]},
                    "city": {"type": "string", "description": "Full name of the Gateway city (e.g., 'Chelsea city, Massachusetts')"},
                    "year": {"type": "integer"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_data",
            "description": "Fetches the current value for a specific metric in a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_fips": {"type": "string", "description": "The 5-digit FIPS code of the city. See your Data Dictionary to translate the city name to the correct FIPS code."},
                    "metric_key": {"type": "string", "description": "The internal Census code (e.g., 'B19013_001E'). See your Data Dictionary for the correct code."}
                },
                "required": ["city_fips", "metric_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cities",
            "description": "Compares a specific metric between two different Gateway cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_a_fips": {"type": "string", "description": "The 5-digit FIPS code of the first city. See your Data Dictionary."},
                    "city_b_fips": {"type": "string", "description": "The 5-digit FIPS code of the second city. See your Data Dictionary."},
                    "metric_key": {"type": "string", "description": "The internal Census code (e.g., 'B19013_001E'). See your Data Dictionary for the correct code."}
                },
                "required": ["city_a_fips", "city_b_fips", "metric_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_correlation",
            "description": "Calculates the statistical correlation (r-value) between two different metrics across all Gateway cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_x": {"type": "string", "description": "First internal Census code (e.g., 'B19013_001E'). See your Data Dictionary."},
                    "metric_y": {"type": "string", "description": "Second internal Census code. See your Data Dictionary."}
                },
                "required": ["metric_x", "metric_y"]
            }
        }
    }
]

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Gateway Cities | Investigative Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# CSS (YOUR editorial CSS is the source of truth)
# ==================================================
def load_css() -> None:
    try:
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    # Minimal additions only (do NOT override your fonts/hero)
    st.markdown(
        """
        <style>
        .section-card-marker { display:none; }

        /* card wrapper (kept light; your CSS controls fonts/colors) */
        div[data-testid="stVerticalBlock"]:has(.section-card-marker) {
            background: #ffffff;
            padding: 26px;
            border-radius: 2px;
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

        .subtle { color:#6b7280; font-size:0.9rem; }

        /* KPI spacing refinement (keeps your typography) */
        div[data-testid="stMetric"] {
            padding: 8px 10px;
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_css()

# ==================================================
# HELPERS
# ==================================================

def apply_chart_style(fig, height: int = 520):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(title=""),
        title=dict(x=0.01),
        font=dict(size=13),
        title_font=dict(size=18),
    )
    return fig


def toast_select(city_full: str, year: int | None = None):
    city = city_full.split(",")[0]
    msg = f"Selected: {city}"
    if year is not None:
        msg += f" • {year}"
    st.toast(msg)
    
def normalize_geo_key(name: str) -> str:
    s = str(name)
    s = s.replace(", Massachusetts", "")
    s = re.sub(r"\b(city|town|cdp)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.strip().upper()


def clean_place_label(name: str) -> str:
    s = str(name).replace(", Massachusetts", "").strip()
    s = re.sub(r"\b(city|town)\b", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


NORMALIZED_ABBR = {
    normalize_geo_key(clean_place_label(k)): v for k, v in GATEWAY_ABBREVIATIONS.items()
}


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_value(v: float, meta: Dict) -> str:
    if pd.isna(v):
        return "—"
    hint = (meta or {}).get("format_hint", "")
    if hint == "percent":
        return f"{float(v):.1f}%"
    if hint == "dollars":
        return f"${float(v):,.0f}"
    try:
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


def first_existing(keys: List[str], catalog: Dict[str, Dict]) -> Optional[str]:
    for k in keys:
        if k in catalog:
            return k
    return None

def run_dashboard_agent(user_message: str):
    client = OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    # We pass the running chat history to the model so it remembers context
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    
    # Format session state messages for the API
    for msg in st.session_state["agent_messages"]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the newest prompt
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=AGENT_TOOLS,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message) 
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            if function_name == "navigate_dashboard":
                execute_agent_action(args) # Uses your existing function!
                tool_result = "Dashboard updated successfully. Tell the user what you did."
                
            elif function_name == "get_metric_data":
                # Uses your existing cache wrappers!
                snap = cached_gateway_metric_snapshot(args["city_fips"], args["metric_key"], st.session_state["selected_year"])
                val = safe_float(snap.get("value", pd.Series([None])).iloc[0]) if snap is not None else "Unknown"
                tool_result = f"The value for {args['metric_key']} is {val}."

            elif function_name == "compare_cities":
                year = st.session_state["selected_year"]
                
                # Fetch City A
                snap_a = cached_gateway_metric_snapshot(args["city_a_fips"], args["metric_key"], year)
                val_a = safe_float(snap_a.get("value", pd.Series([None])).iloc[0]) if snap_a is not None and not snap_a.empty else "Unknown"
                
                # Fetch City B
                snap_b = cached_gateway_metric_snapshot(args["city_b_fips"], args["metric_key"], year)
                val_b = safe_float(snap_b.get("value", pd.Series([None])).iloc[0]) if snap_b is not None and not snap_b.empty else "Unknown"
                
                tool_result = f"For {args['metric_key']} in {year}: City A ({args['city_a_fips']}) is {val_a}, City B ({args['city_b_fips']}) is {val_b}."

            elif function_name == "get_metric_correlation":
                year = st.session_state["selected_year"]
                
                # Use your existing scatter cache and stats function!
                df_sc = cached_gateway_scatter(args["metric_x"], args["metric_y"], year)
                stats = compute_scatter_stats(df_sc)
                
                if stats and stats.r is not None:
                    tool_result = f"The correlation (r-value) between {args['metric_x']} and {args['metric_y']} is {stats.r:.2f} (n={stats.n} cities)."
                else:
                    tool_result = "Could not calculate correlation. Data might be missing."
                    
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(tool_result),
            })

        # Second API Call to synthesize the final text
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        return final_response.choices[0].message.content

    return response_message.content

def execute_agent_action(action: dict):

    # -------------------------------------------------
    # New schema (tab / city / year)
    # -------------------------------------------------
    if "tab" in action and action["tab"]:
        st.session_state["active_tab"] = action["tab"]

    if "city" in action and action["city"]:
        st.session_state["selected_city"] = action["city"]

    if "year" in action and action["year"]:
        st.session_state["selected_year"] = int(action["year"])

    # -------------------------------------------------
    # Backwards compatibility (old action-style schema)
    # -------------------------------------------------
    if "action" in action:

        if action["action"] == "open_tab":
            if "tab" in action:
                st.session_state["active_tab"] = action["tab"]

            if "city" in action:
                st.session_state["selected_city"] = action["city"]

        elif action["action"] == "set_city":
            st.session_state["selected_city"] = action["city"]

        elif action["action"] == "set_year":
            st.session_state["selected_year"] = int(action["year"])

        elif action["action"] == "run_investigation":
            st.session_state["active_tab"] = "Investigative Themes"

        elif action["action"] == "explain_chart":
            st.session_state["agent_explain"] = True

def render_section_card_start():
    """Drops the hidden marker that triggers your custom CSS card wrapper."""
    st.markdown("<div class='section-card-marker'></div>", unsafe_allow_html=True)

def render_pill(text: str):
    """Renders a small editorial pill/tag."""
    st.markdown(f"<span class='pill'>{text}</span>", unsafe_allow_html=True)

def render_map_legend(items: list[tuple[str, str]]):
    """
    Renders a custom legend. 
    Expects a list of tuples: (color_hex, label)
    """
    legend_html = "<div class='map-legend'>"
    for color, label in items:
        legend_html += f"""
        <div class='legend-item'>
            <span class='dot' style='background-color: {color};'></span>
            <span>{label}</span>
        </div>
        """
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)
           
# ==================================================
# ANALYTICS LAYER (in-app)
# ==================================================
@dataclass
class DistributionContext:
    value: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    z: Optional[float]
    percentile: Optional[float]  # 0-100
    rank: Optional[int]
    n: Optional[int]


def compute_distribution_context(rank_df: pd.DataFrame, place_fips: str) -> DistributionContext:
    if rank_df is None or rank_df.empty:
        return DistributionContext(None, None, None, None, None, None, None, None)

    df = rank_df.copy()
    df["place_fips"] = df["place_fips"].astype(str)

    value_col = None
    for cand in ["value", "metric_value", "x"]:
        if cand in df.columns:
            value_col = cand
            break

    if value_col is None:
        numeric_cols = []
        for c in df.columns:
            if c.lower().startswith("rank"):
                continue
            if c in ["place_fips", "place_name"]:
                continue
            if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
                numeric_cols.append(c)
        value_col = numeric_cols[0] if numeric_cols else None

    if value_col is None:
        return DistributionContext(None, None, None, None, None, None, None, None)

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return DistributionContext(None, None, None, None, None, None, None, None)

    v_ser = df.loc[df["place_fips"] == str(place_fips), value_col]
    v = safe_float(v_ser.iloc[0]) if not v_ser.empty else None

    mean = safe_float(df[value_col].mean())
    median = safe_float(df[value_col].median())
    std = safe_float(df[value_col].std(ddof=0))

    z = None
    if v is not None and std not in (None, 0.0):
        z = (v - mean) / std

    percentile = None
    if v is not None:
        percentile = float((df[value_col] <= v).mean() * 100)

    rank = None
    for rc in ["rank_within_gateway", "rank", "gateway_rank"]:
        if rc in df.columns:
            rr = df.loc[df["place_fips"] == str(place_fips), rc]
            if not rr.empty:
                try:
                    rank = int(rr.iloc[0])
                except Exception:
                    rank = None
            break

    return DistributionContext(
        value=v,
        mean=mean,
        median=median,
        std=std,
        z=z,
        percentile=percentile,
        rank=rank,
        n=int(len(df)),
    )


@dataclass
class TrendDiagnostics:
    slope_5yr: Optional[float]
    slope_10yr: Optional[float]
    delta_5yr: Optional[float]
    last: Optional[float]
    first: Optional[float]


def compute_trend_diagnostics(trend_df: pd.DataFrame) -> TrendDiagnostics:
    if trend_df is None or trend_df.empty:
        return TrendDiagnostics(None, None, None, None, None)

    df = trend_df.copy()
    if "year" not in df.columns or "value" not in df.columns:
        return TrendDiagnostics(None, None, None, None, None)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"]).sort_values("year")
    if len(df) < 2:
        last = safe_float(df["value"].iloc[-1]) if len(df) else None
        first = safe_float(df["value"].iloc[0]) if len(df) else None
        return TrendDiagnostics(None, None, None, last, first)

    years = df["year"].to_numpy()
    vals = df["value"].to_numpy()

    def slope_over(window_years: int) -> Optional[float]:
        y_max = years.max()
        mask = years >= (y_max - window_years)
        suby = years[mask]
        subv = vals[mask]
        if len(suby) < 2:
            return None
        try:
            return float(np.polyfit(suby, subv, 1)[0])
        except Exception:
            return None

    slope_5 = slope_over(5)
    slope_10 = slope_over(10)

    delta_5 = None
    try:
        y_max = float(years.max())
        target = y_max - 5
        idx0 = int(np.argmin(np.abs(years - target)))
        delta_5 = float(vals[-1] - vals[idx0])
    except Exception:
        delta_5 = None

    return TrendDiagnostics(
        slope_5yr=slope_5,
        slope_10yr=slope_10,
        delta_5yr=delta_5,
        last=safe_float(vals[-1]),
        first=safe_float(vals[0]),
    )


@dataclass
class ScatterStats:
    r: Optional[float]
    r2: Optional[float]
    slope: Optional[float]
    intercept: Optional[float]
    n: Optional[int]


def compute_scatter_stats(sc_df: pd.DataFrame) -> ScatterStats:
    if sc_df is None or sc_df.empty or "x" not in sc_df.columns or "y" not in sc_df.columns:
        return ScatterStats(None, None, None, None, None)

    df = sc_df.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if len(df) < 3:
        return ScatterStats(None, None, None, None, int(len(df)))

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    try:
        r = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        r = None
    r2 = float(r * r) if r is not None else None

    slope = intercept = None
    try:
        slope, intercept = np.polyfit(x, y, 1)
        slope, intercept = float(slope), float(intercept)
    except Exception:
        pass

    return ScatterStats(r=r, r2=r2, slope=slope, intercept=intercept, n=int(len(df)))


def build_narrative_summary(
    city_name: str,
    year: int,
    catalog: Dict[str, Dict],
    place_fips: str,
    focus_metrics: List[str],
    advanced: bool,
) -> List[str]:
    bullets: List[str] = []
    for mk in focus_metrics:
        meta = catalog.get(mk, {"metric_label": mk})
        snap = cached_gateway_metric_snapshot(place_fips, mk, year)
        tr = cached_gateway_metric_trend(place_fips, mk)

        label = meta.get("metric_label", mk)

        delta = None
        val = None
        if snap is not None and not snap.empty:
            val = safe_float(snap.get("value", pd.Series([None])).iloc[0])
            delta = safe_float(snap.get("delta_5yr", pd.Series([None])).iloc[0])

        if val is None:
            continue

        if delta is not None:
            bullets.append(f"**{label}:** {fmt_value(val, meta)} ({fmt_delta(delta, meta)}).")
        else:
            bullets.append(f"**{label}:** {fmt_value(val, meta)}.")

        if advanced and tr is not None and not tr.empty:
            diag = compute_trend_diagnostics(tr)
            if diag.slope_10yr is not None and abs(diag.slope_10yr) > 0:
                bullets.append(
                    f"<span class='subtle'>Trend signal: ~{diag.slope_10yr:+.3g} per year (10yr linear slope).</span>"
                )

    return bullets[:10]


# ==================================================
# GEOJSON (MAP)
# ==================================================
@st.cache_data
def load_ma_map() -> dict:
    with open("data/ma_municipalities.geojson") as f:
        return json.load(f)


ma_geo = load_ma_map()


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

# ==================================================
# REGISTRY + CATALOG
# ==================================================
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
    st.error("No gateway cities found. Check is_gateway_city flags / gateway_cities table.")
    st.stop()

catalog_df = get_metric_catalog()
catalog: Dict[str, Dict] = (
    catalog_df.set_index("metric_key").to_dict(orient="index") if not catalog_df.empty else {}
)
if not catalog:
    st.error("Metric catalog is empty. Verify public.metric_catalog.")
    st.stop()

available_years = get_available_gateway_years()
if not available_years:
    st.error("No data found in public.gateway_metrics.")
    st.stop()

min_year = min(available_years)
max_year = max(available_years)

# ==================================================
# SESSION STATE (single source of truth)
# ==================================================
if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = max_year
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = gateway_city_options[0]

ADV = True

# ==================================================
# AGENT STATE
# ==================================================

if "agent_messages" not in st.session_state:
    st.session_state["agent_messages"] = []

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Map"
    
# ==================================================
# HERO + TOP CONTROL STRIP (keep your design)
# ==================================================
st.markdown(
    """
    <section class="hero">
        <h1>Gateway Cities Dashboard</h1>
        <div class="accent-line"></div>
        <p>
            Story-first exploration of demographic change, economic stress, and housing pressure across Massachusetts Gateway Cities.
            Source: ACS 5-year estimates (endpoint years; latest available in warehouse).
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

selected_year = int(st.session_state["selected_year"])
primary_city = st.session_state["selected_city"]
primary_fips = str(cities_all.loc[cities_all["place_name"] == primary_city, "place_fips"].iloc[0])

# ==================================================
# DASHBOARD AGENT CHAT (Homepage Copilot)
# ==================================================

for msg in st.session_state["agent_messages"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("E.g., What is the poverty trend in Lynn?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Crunching numbers..."):
            answer = run_dashboard_agent(prompt)
            st.markdown(answer)

    st.session_state["agent_messages"].append({"role": "user", "content": prompt})
    st.session_state["agent_messages"].append({"role": "assistant", "content": answer})

    st.rerun()
        
# ==================================================
# STORY LEADS (AI INVESTIGATIVE SIGNALS)
# ==================================================

core = [
    k for k in [
        "median_income",
        "poverty_rate",
        "rent_burden_30_plus",
        "foreign_born_share",
        "total_population"
    ] if k in catalog
]

# ----------------------------
# Extreme outlier (selected city vs distribution)
# ----------------------------

z_records = []

for k in core:
    df_rank = cached_gateway_ranking(k, selected_year)

    if df_rank is None or df_rank.empty:
        continue

    ctx = compute_distribution_context(df_rank, primary_fips)

    if ctx and ctx.z is not None:
        z_records.append((k, ctx.z))

valid_z = [r for r in z_records if r[1] is not None]

zmax = max(valid_z, key=lambda t: abs(t[1]), default=(None, None))


# ----------------------------
# Strongest cross-city relationship
# ----------------------------

pairs = [
    (x, y)
    for a in STORY_ANGLES.values()
    for (x, y) in a.get("investigative_pairs", [])
    if x in catalog and y in catalog
]

records = []

for x, y in pairs:

    df_sc = cached_gateway_scatter(x, y, selected_year)

    if df_sc is None or df_sc.empty:
        continue

    stats = compute_scatter_stats(df_sc)

    if stats and stats.r is not None:
        records.append((x, y, stats.r))

valid_records = [r for r in records if r[2] is not None]

best = max(valid_records, key=lambda t: abs(t[2]), default=(None, None, None))

# ----------------------------
# Fastest changing trend
# ----------------------------

trend_records = []

for k in core:

    df_tr = cached_gateway_metric_trend(primary_fips, k)

    if df_tr is None or df_tr.empty:
        continue

    diag = compute_trend_diagnostics(df_tr)

    if diag and diag.slope_10yr is not None:
        trend_records.append((k, diag.slope_10yr))

valid_trends = [r for r in trend_records if r[1] is not None]

fast = max(valid_trends, key=lambda t: abs(t[1]), default=(None, None))


# ==================================================
# Render section
# ==================================================

st.markdown("## Story Leads for Journalists")

c1, c2, c3 = st.columns(3)

with c1:
    if zmax[0] and zmax[1] is not None:
        st.metric("Extreme outlier", catalog[zmax[0]]["metric_label"], f"z = {zmax[1]:+.2f}")
    else:
        st.metric("Extreme outlier", "—")

with c2:
    if best[0] and best[1] and best[2] is not None:
        st.metric(
            "Strongest relationship",
            f"{catalog[best[0]]['metric_label']} vs {catalog[best[1]]['metric_label']}",
            f"r = {best[2]:+.2f}"
        )
    else:
        st.metric("Strongest relationship", "—")

with c3:
    if fast[0] and fast[1] is not None:
        st.metric("Fastest trend", catalog[fast[0]]["metric_label"], f"{fast[1]:+.3g}/yr")
    else:
        st.metric("Fastest trend", "—")

st.caption(
    "Automated investigative signals generated from cross-city comparisons and time trends. "
    "Leads for reporting, not causal conclusions."
)
        
# ==================================================
# TABS / NAVIGATION
# ==================================================

tabs = [
    "Map",
    "Investigative Themes",
    "Compare Metrics",
    "Origins (B05006)",
    "Ask the Data",
    "Methodology",
]

# ensure active_tab exists
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Map"

# Radio navigation (styled like tabs)
with st.container():
    selected_tab = st.radio(
        "",
        tabs,
        index=tabs.index(st.session_state["active_tab"]),
        horizontal=True,
        label_visibility="collapsed"
    )

# keep session state synced
st.session_state["active_tab"] = selected_tab

# ==================================================
# TAB 1: MAP (choropleth + click select + city profile)
# ==================================================
if st.session_state["active_tab"] == "Map":
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Geographic Context")

        cA, cB, cC, cD = st.columns([2.2, 1.2, 1.2, 1.0])

        with cB:
            st.session_state["selected_city"] = st.selectbox(
                "Gateway city",
                options=gateway_city_options,
                index=gateway_city_options.index(st.session_state["selected_city"])
                if st.session_state["selected_city"] in gateway_city_options
                else 0,
                key="map_city",
            )

        with cC:
            st.session_state["selected_year"] = st.selectbox(
                "Analysis year",
                options=available_years,
                index=available_years.index(st.session_state["selected_year"]),
                key="map_year",
            )

        with cD:
            st.markdown(
                f"<div style='text-align:right;'><span class='pill'>Data range: <b>{min_year}–{max_year}</b></span></div>",
                unsafe_allow_html=True,
            )

        selected_year = int(st.session_state["selected_year"])
        primary_city = st.session_state["selected_city"]
        primary_fips = str(
            cities_all.loc[cities_all["place_name"] == primary_city, "place_fips"].iloc[0]
        )

        # We key the map to TOWN (as in your prior working version)
        town_fips_map = {
            normalize_geo_key(name): fips
            for name, fips in zip(cities_all["place_name"], cities_all["place_fips"])
        }
        allowed_gateway_names = set(
            normalize_geo_key(n)
            for n in cities_all[cities_all["place_fips"].isin(gateway_fips)]["place_name"]
        )
        boston_cambridge_names = set(
            normalize_geo_key(clean_place_label(n))
            for n in extra_cities["place_name"]
        )

        locations = [f["properties"]["TOWN"] for f in ma_geo["features"]]

        # Centroids for labels
        town_centroids: Dict[str, Tuple[float, float]] = {}
        for feature in ma_geo["features"]:
            town_raw = feature["properties"]["TOWN"]
            town_key = normalize_geo_key(town_raw)
            coords = feature["geometry"]["coordinates"]
            lats: List[float] = []
            lons: List[float] = []

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

        @st.cache_data(ttl=3600)
        def build_map(
            geojson: dict,
            locations_in: List[str],
            allowed_gateway_names_in: set,
            town_fips_map_local: Dict[str, str],
            selected_fips: str,
            c_lat: float,
            c_lon: float,
            boston_cambridge_names_in: set,
        ) -> go.Figure:
            z_values: List[int] = []
            for town_name in locations_in:
                town_norm = normalize_geo_key(town_name)
                if town_norm in town_fips_map_local and town_fips_map_local[town_norm] == selected_fips:
                    z_values.append(3)
                elif town_norm in boston_cambridge_names_in:
                    z_values.append(2)
                elif town_norm in allowed_gateway_names_in:
                    z_values.append(1)
                else:
                    z_values.append(0)

            trace = go.Choroplethmapbox(
                geojson=geojson,
                locations=locations_in,
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

        # Abbreviation labels (offsets for dense Boston-area)
        LABEL_OFFSETS = {
            "LEOMINSTER": (-0.020, 0.020),
            "METHUEN": (0.015, 0.012),
            "MALDEN": (0.010, -0.010),
            "CHELSEA": (-0.005, 0.020),
            "FITCHBURG": (0.010, -0.010),
            "REVERE": (0.010, 0.010),
            "EVERETT": (-0.005, -0.020),
        }

        label_lats: List[float] = []
        label_lons: List[float] = []
        label_text: List[str] = []
        for town_name in locations:
            town_key = normalize_geo_key(town_name)
            if town_key in allowed_gateway_names and town_key in town_centroids:
                fips = town_fips_map.get(town_key)
                if not fips:
                    continue
                full_name = cities_all[cities_all["place_fips"] == str(fips)]["place_name"].iloc[0]
                city_key = normalize_geo_key(clean_place_label(full_name))
                abbr = NORMALIZED_ABBR.get(city_key)
                if not abbr:
                    continue
                lat, lon = town_centroids[town_key]
                off_lat, off_lon = LABEL_OFFSETS.get(town_key, (0.0, 0.0))
                label_lats.append(lat + off_lat)
                label_lons.append(lon + off_lon)
                label_text.append(abbr)

        fig_map.add_trace(
            go.Scattermapbox(
                lat=label_lats,
                lon=label_lons,
                mode="text",
                text=label_text,
                textfont=dict(size=11, color="#111111"),
                textposition="middle center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True,
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

        # Click-to-select (robust; no double-click logic)
        selection = st.session_state.get("map_select")

        if selection and selection.get("selection"):
            points = selection["selection"].get("points", [])
            loc_points = [p for p in points if p.get("location")]

            if loc_points:
                clicked_town = loc_points[-1]["location"]
                town_key = normalize_geo_key(clicked_town)
                new_fips = town_fips_map.get(town_key)

                if new_fips and str(new_fips) in gateway_fips:
                    new_city = cities_all.loc[
                        cities_all["place_fips"] == str(new_fips),
                        "place_name"
                    ].iloc[0]

                    if st.session_state["selected_city"] != new_city:
                        st.session_state["selected_city"] = new_city
                        toast_select(new_city, selected_year)
                        st.rerun()

        # =========================
        # CITY PROFILE (keeps your design)
        # =========================
        st.divider()
        st.markdown(f"# {primary_city.split(',')[0]} — City Profile")

        st.markdown("## Executive Summary")
        focus_for_summary = [
            first_existing(["median_income"], catalog),
            first_existing(["poverty_rate"], catalog),
            first_existing(["rent_burden_30_plus"], catalog),
            first_existing(["median_home_value"], catalog),
            first_existing(["foreign_born_share"], catalog),
            first_existing(["total_population"], catalog),
        ]
        focus_for_summary = [x for x in focus_for_summary if x]

        bullets = build_narrative_summary(
            primary_city.split(",")[0],
            selected_year,
            catalog,
            primary_fips,
            focus_for_summary,
            advanced=ADV,
        )
        if not ADV:
            bullets = [b for b in bullets if "Trend signal" not in b]
        if not bullets:
            st.info("No summary available (missing snapshot data for this year).")
        else:
            for b in bullets[:6]:
                st.markdown(f"- {b}", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"## Snapshot {selected_year}")

        core_metrics = [
            "total_population",
            "foreign_born_share",
            "median_income",
            "poverty_rate",
            "rent_burden_30_plus",
            "median_home_value",
            "gini_index",
        ]

        cols = st.columns(4)
        for i, mk in enumerate(core_metrics):
            meta = catalog.get(mk, {"metric_label": mk})
            snap = cached_gateway_metric_snapshot(primary_fips, mk, selected_year)
            if snap is None or snap.empty:
                cols[i % 4].metric(meta.get("metric_label", mk), "—")
                continue
            value = snap.get("value", pd.Series([None])).iloc[0]
            delta = snap.get("delta_5yr", pd.Series([None])).iloc[0]
            cols[i % 4].metric(meta.get("metric_label", mk), fmt_value(value, meta), fmt_delta(delta, meta))

        st.divider()
        st.markdown("## Structural Trend")

        structural_metrics = [k for k in ["median_income", "poverty_rate", "rent_burden_30_plus", "foreign_born_share", "total_population"] if k in catalog]

        trend_metric = st.selectbox(
            "Select trend metric",
            structural_metrics,
            format_func=lambda k: catalog.get(k, {}).get("metric_label", k),
            key="map_trend_focus",
        )

        city_tr = cached_gateway_metric_trend(primary_fips, trend_metric)
        state_tr = cached_state_metric_trend(trend_metric)
        meta = catalog.get(trend_metric, {"metric_label": trend_metric})

        if city_tr is not None and not city_tr.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=city_tr["year"], y=city_tr["value"], mode="lines", name=primary_city.split(",")[0]))
            if state_tr is not None and not state_tr.empty:
                fig.add_trace(go.Scatter(x=state_tr["year"], y=state_tr["value"], mode="lines", name="Massachusetts"))
            fig.update_layout(template="plotly_white", height=420, xaxis_title="Year", yaxis_title=meta.get("unit", ""), legend=dict(title=""))
            st.plotly_chart(fig, use_container_width=True)

            if ADV:
                diag = compute_trend_diagnostics(city_tr)
                c1, c2, c3 = st.columns(3)
                if diag.delta_5yr is not None:
                    c1.metric("5-year change (approx.)", fmt_delta(diag.delta_5yr, meta) or f"{diag.delta_5yr:+.3g}")
                if diag.slope_5yr is not None:
                    c2.metric("Slope (5yr window)", f"{diag.slope_5yr:+.3g} / year")
                if diag.slope_10yr is not None:
                    c3.metric("Slope (10yr window)", f"{diag.slope_10yr:+.3g} / year")
        else:
            st.info("No trend data available.")

        st.divider()
        st.markdown(f"## Gateway Position — {meta.get('metric_label', trend_metric)} ({selected_year})")

        rank_df = cached_gateway_ranking(trend_metric, selected_year)
        if rank_df is None or rank_df.empty:
            st.info("No ranking data available.")
        else:
            ctx = compute_distribution_context(rank_df, primary_fips)
            c1, c2, c3, c4 = st.columns(4)

            if ctx.rank is not None and ctx.n is not None:
                c1.metric("Gateway rank", f"{ctx.rank} / {ctx.n}")
            if ctx.percentile is not None:
                c2.metric("Percentile (empirical)", f"{ctx.percentile:.0f}th")
            if ADV and ctx.z is not None:
                c3.metric("Z-score vs gateway", f"{ctx.z:+.2f}", help="How many standard deviations this city is from the Gateway City mean.")
            if ADV and ctx.mean is not None and ctx.value is not None:
                gap = ctx.value - ctx.mean
                c4.metric("Gap vs gateway mean", f"{gap:+.3g}")

            if ADV:
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
            else:
                sub = rank_df.copy()
                sub["place_fips"] = sub["place_fips"].astype(str)
                one = sub[sub["place_fips"] == str(primary_fips)]
                if not one.empty:
                    st.dataframe(one, use_container_width=True, hide_index=True)

# ==================================================
# TAB 2: INVESTIGATIVE THEMES (no extra toggles; Advanced controls depth)
# ==================================================
if st.session_state["active_tab"] == "Investigative Themes":
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Investigative Themes")

        cities_df = get_cities(gateway_only=True)
        story_city = st.selectbox(
            "Gateway City",
            cities_df["place_name"].tolist(),
            index=max(0, cities_df["place_name"].tolist().index(primary_city)) if primary_city in cities_df["place_name"].tolist() else 0,
            key="story_city",
        )
        place_fips = str(cities_df.loc[cities_df["place_name"] == story_city, "place_fips"].iloc[0])

        angle_key = st.selectbox(
            "Theme",
            list(STORY_ANGLES.keys()),
            format_func=lambda k: STORY_ANGLES[k]["title"],
            key="angle_key",
        )
        angle = STORY_ANGLES[angle_key]

        st.subheader(angle["title"])

        metrics = angle.get("metrics", [])
        if not metrics:
            st.info("No metrics configured for this theme.")
        else:
            show_metrics = metrics[:4]
            cols = st.columns(min(len(show_metrics), 6))
            for i, mk in enumerate(show_metrics):
                meta = catalog.get(mk, {"metric_label": mk, "format_hint": "number"})
                snap = cached_gateway_metric_snapshot(place_fips, mk, selected_year)
                if snap is None or snap.empty:
                    cols[i % len(cols)].metric(meta.get("metric_label", mk), "—")
                    continue
                v = snap["value"].iloc[0]
                d5 = snap.get("delta_5yr", pd.Series([None])).iloc[0]
                cols[i % len(cols)].metric(meta.get("metric_label", mk), fmt_value(v, meta), fmt_delta(d5, meta))

        st.divider()

        lead_metric = st.selectbox(
            "Trend focus metric",
            metrics if metrics else list(catalog.keys()),
            format_func=lambda k: catalog.get(k, {"metric_label": k}).get("metric_label", k),
            key="lead_metric",
        )

        city_trend = cached_gateway_metric_trend(place_fips, lead_metric)
        ma_trend = cached_state_metric_trend(lead_metric)

        meta = catalog.get(lead_metric, {"metric_label": lead_metric})
        st.markdown(f"**Trend:** {meta.get('metric_label', lead_metric)}")

        if city_trend is None or city_trend.empty:
            st.info("No trend data available for this metric/city.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=city_trend["year"], y=city_trend["value"], mode="lines", name=story_city.split(",")[0]))
            if ma_trend is not None and not ma_trend.empty:
                fig.add_trace(go.Scatter(x=ma_trend["year"], y=ma_trend["value"], mode="lines", name="Massachusetts"))
            fig.update_layout(template="plotly_white", height=430, xaxis_title="Year", yaxis_title=meta.get("unit", ""), legend=dict(title=""))
            st.plotly_chart(fig, use_container_width=True)

            if ADV:
                diag = compute_trend_diagnostics(city_trend)
                c1, c2, c3 = st.columns(3)
                if diag.delta_5yr is not None:
                    c1.metric("5-year change (approx.)", fmt_delta(diag.delta_5yr, meta) or f"{diag.delta_5yr:+.3g}")
                if diag.slope_5yr is not None:
                    c2.metric("Slope (5yr window)", f"{diag.slope_5yr:+.3g} / year")
                if diag.slope_10yr is not None:
                    c3.metric("Slope (10yr window)", f"{diag.slope_10yr:+.3g} / year")

        st.divider()
        st.markdown(f"### Gateway Ranking — {meta.get('metric_label')} ({selected_year})")

        rank_df = cached_gateway_ranking(lead_metric, selected_year)
        if rank_df is None or rank_df.empty:
            st.info("No ranking data available for this metric/year.")
        else:
            if ADV:
                st.dataframe(rank_df, use_container_width=True, hide_index=True)
            else:
                sub = rank_df.copy()
                sub["place_fips"] = sub["place_fips"].astype(str)
                one = sub[sub["place_fips"] == str(place_fips)]
                if not one.empty:
                    st.dataframe(one, use_container_width=True, hide_index=True)
                else:
                    st.info("Selected city not found in ranking output for this year.")

        if ADV:
            st.divider()
            st.subheader("Relationships (Gateway Cities only)")

            pairs = angle.get("investigative_pairs", [])
            if not pairs:
                st.info("No investigative pairs configured for this theme.")
            else:
                pair_labels = []
                for xk, yk in pairs:
                    xl = catalog.get(xk, {"metric_label": xk}).get("metric_label", xk)
                    yl = catalog.get(yk, {"metric_label": yk}).get("metric_label", yk)
                    pair_labels.append(f"{xl} vs {yl}")

                idx = st.selectbox("Choose a comparison", range(len(pairs)), format_func=lambda i: pair_labels[i], key="pair_idx")
                xk, yk = pairs[idx]

                sc = cached_gateway_scatter(xk, yk, selected_year)
                xl = catalog.get(xk, {"metric_label": xk}).get("metric_label", xk)
                yl = catalog.get(yk, {"metric_label": yk}).get("metric_label", yk)

                if sc is None or sc.empty:
                    st.info("No scatter data available for this comparison.")
                else:
                    sc = sc.copy()
                    sc["x"] = pd.to_numeric(sc["x"], errors="coerce")
                    sc["y"] = pd.to_numeric(sc["y"], errors="coerce")
                    sc = sc.dropna(subset=["x", "y"])
                    if sc.empty:
                        st.info("No valid numeric scatter data available for this comparison.")
                    else:
                        sc["is_selected"] = sc["place_fips"].astype(str) == str(place_fips)
                        stats = compute_scatter_stats(sc)

                        # --------------------------------
                        # Detect statistical outliers
                        # --------------------------------
                        if stats.slope is not None and stats.intercept is not None:
                            sc["predicted"] = stats.slope * sc["x"] + stats.intercept
                            sc["residual"] = sc["y"] - sc["predicted"]
                            sc["outlier"] = np.abs(sc["residual"]) > (2 * sc["residual"].std())
                        else:
                            sc["outlier"] = False
    
                        fig_sc = px.scatter(
                            sc,
                            x="x",
                            y="y",
                            hover_name="place_name",
                            title=f"{selected_year}: {xl} (x) vs {yl} (y)",
                        )

                        if ADV and stats.slope is not None and stats.intercept is not None:
                            xx = np.linspace(float(sc["x"].min()), float(sc["x"].max()), 50)
                            yy = stats.slope * xx + stats.intercept
                            fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="OLS fit"))

                        sel = sc[sc["is_selected"]]
                        if not sel.empty:
                            fig_sc.add_trace(
                                go.Scatter(
                                    x=sel["x"],
                                    y=sel["y"],
                                    mode="markers",
                                    name=story_city.split(",")[0],
                                    marker=dict(size=16, symbol="diamond"),
                                    hovertemplate=f"<b>{story_city}</b><br>{xl}: %{{x}}<br>{yl}: %{{y}}<extra></extra>",
                                )
                            )

                            # Highlight statistical outliers
                            outliers = sc[sc["outlier"]]

                            if not outliers.empty:
                                fig_sc.add_trace(
                                    go.Scatter(
                                        x=outliers["x"],
                                        y=outliers["y"],
                                        mode="markers+text",
                                        text=outliers["place_name"].str.split(",").str[0],
                                        textposition="top center",
                                        marker=dict(size=18, color="black", symbol="circle-open"),
                                        name="Statistical outliers",
                                    )
                                )
    
                        fig_sc.update_layout(template="plotly_white", height=520)
                        st.plotly_chart(fig_sc, use_container_width=True)

                        # Agent explanation
                        if st.session_state.get("agent_explain"):
                            st.info(
                                """
                        This chart compares the selected metrics across Massachusetts Gateway Cities.

                        Each point represents one city. The regression line shows the overall cross-city
                        relationship between the two indicators.

                        Cities above the line have **higher-than-expected values** given the relationship,
                        while cities below the line perform **lower than expected**.

                        Large deviations from the line are potential **investigative outliers** and may
                        reflect local housing markets, demographic composition, or economic policy differences.
                        """
                            )

                            st.session_state["agent_explain"] = False
    
                        # -------------------------------
                        # Dynamic caption + interpretation
                        # -------------------------------
                        if stats.r is not None:
                            abs_r = abs(stats.r)

                            if abs_r > 0.7:
                                strength = "strong"
                            elif abs_r > 0.4:
                                strength = "moderate"
                            elif abs_r > 0.2:
                                strength = "weak"
                            else:
                                strength = "very weak"

                            direction = "positive" if stats.r > 0 else "negative"

                            st.caption(
                                f"""
                        Each point represents a Massachusetts Gateway City. The horizontal axis shows **{xl}**
                        while the vertical axis shows **{yl}** in **{selected_year}**. The fitted line represents
                        the ordinary least squares (OLS) trend across cities.

                        Interpretation: The relationship between **{xl}** and **{yl}** is **{strength} and {direction}**
                        (r = {stats.r:.2f}). Differences across cities suggest local housing, economic, or demographic
                        conditions shape outcomes beyond this single relationship.
                        """
                            )
    
                        c1, c2, c3 = st.columns(3)
                        if stats.r is not None:
                            c1.metric("Correlation r", f"{stats.r:+.2f}")
                        if stats.r2 is not None:
                            c2.metric("R²", f"{stats.r2:.2f}")
                        if stats.n is not None:
                            c3.metric("N (cities)", f"{stats.n}")

# ==================================================
# TAB 3: COMPARE METRICS (advanced = extra stats/regression)
# ==================================================
if st.session_state["active_tab"] == "Compare Metrics":
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Compare Metrics")

        # Initialize session state once
        if "compare_cities" not in st.session_state:
            st.session_state["compare_cities"] = [primary_city]
            
        # --- Select all helper buttons ---
        btn_col1, btn_col2 = st.columns([1,1])

        with btn_col1:
            if st.button("Select All Gateway Cities"):
                st.session_state["compare_cities"] = gateway_city_options

        with btn_col2:
            if st.button("Clear Selection"):
                st.session_state["compare_cities"] = []

        # --- City selector ---
        compare_cities = st.multiselect(
            "Select Gateway Cities",
            options=gateway_city_options,
            key="compare_cities",
        )
        if not compare_cities:
            compare_cities = [primary_city]

        selected_fips = {
            city: str(cities_all.loc[cities_all["place_name"] == city, "place_fips"].iloc[0])
            for city in compare_cities
        }

        metric_keys = sorted(list(catalog.keys()))
        metric_labels = {k: catalog[k].get("metric_label", k) for k in metric_keys}

        col_a, col_b = st.columns([1.2, 2.3])

        with col_a:
            metric_to_compare = st.selectbox(
                "Trend metric",
                metric_keys,
                index=metric_keys.index("median_income") if "median_income" in metric_keys else 0,
                format_func=lambda k: metric_labels.get(k, k),
                key="compare_metric",
            )

        with col_b:
            st.markdown("**Multi-city trend**")
            fig = go.Figure()
            for city_name, fips in selected_fips.items():
                tr = cached_gateway_metric_trend(fips, metric_to_compare)
                if tr is None or tr.empty:
                    continue
                fig.add_trace(go.Scatter(x=tr["year"], y=tr["value"], mode="lines", name=city_name.split(",")[0]))
            fig.update_layout(
                template="plotly_white",
                height=450,
                xaxis_title="Year",
                yaxis_title=catalog.get(metric_to_compare, {}).get("unit", ""),
                legend=dict(title=""),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Cross-metric scatter (Gateway Cities)")

        c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
        with c1:
            metric_x = st.selectbox(
                "X metric",
                metric_keys,
                index=metric_keys.index("rent_burden_30_plus") if "rent_burden_30_plus" in metric_keys else 0,
                format_func=lambda k: metric_labels.get(k, k),
                key="scatter_x",
            )
        with c2:
            metric_y = st.selectbox(
                "Y metric",
                metric_keys,
                index=metric_keys.index("poverty_rate") if "poverty_rate" in metric_keys else min(1, len(metric_keys) - 1),
                format_func=lambda k: metric_labels.get(k, k),
                key="scatter_y",
            )
        with c3:
            sc_year = st.selectbox("Year", available_years, index=available_years.index(selected_year), key="scatter_year")

        sc = cached_gateway_scatter(metric_x, metric_y, sc_year)
        if sc is None or sc.empty:
            st.info("No data available for that scatter combination.")
        else:
            sc = sc.copy()
            sc["x"] = pd.to_numeric(sc["x"], errors="coerce")
            sc["y"] = pd.to_numeric(sc["y"], errors="coerce")
            sc = sc.dropna(subset=["x", "y"])
            if sc.empty:
                st.info("No valid numeric scatter data available.")
            else:
                sc["is_selected"] = sc["place_fips"].astype(str).isin(set(selected_fips.values()))
                xl = metric_labels.get(metric_x, metric_x)
                yl = metric_labels.get(metric_y, metric_y)

                stats = compute_scatter_stats(sc)

                # --------------------------------
                # Detect statistical outliers
                # --------------------------------
                if stats.slope is not None and stats.intercept is not None:
                    sc["predicted"] = stats.slope * sc["x"] + stats.intercept
                    sc["residual"] = sc["y"] - sc["predicted"]
                    sc["outlier"] = np.abs(sc["residual"]) > (2 * sc["residual"].std())
                else:
                    sc["outlier"] = False
    
                fig_sc = px.scatter(sc, x="x", y="y", hover_name="place_name", title=f"{sc_year}: {xl} (x) vs {yl} (y)")

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

                if ADV and stats.slope is not None and stats.intercept is not None:
                    x_min = float(sc["x"].min())
                    x_max = float(sc["x"].max())
                    if x_min != x_max:
                        xx = np.linspace(x_min, x_max, 50)
                        yy = stats.slope * xx + stats.intercept
                        fig_sc.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="OLS fit"))

                    # Highlight statistical outliers
                    outliers = sc[sc["outlier"]]

                    if not outliers.empty:
                        fig_sc.add_trace(
                            go.Scatter(
                                x=outliers["x"],
                                y=outliers["y"],
                                mode="markers+text",
                                text=outliers["place_name"].str.split(",").str[0],
                                textposition="top center",
                                marker=dict(size=18, color="black", symbol="circle-open"),
                                name="Statistical outliers",
                            )
                        )
                fig_sc.update_layout(template="plotly_white", height=560)
                st.plotly_chart(fig_sc, use_container_width=True)

                # Agent explanation
                if st.session_state.get("agent_explain"):
                    st.info(
                        """
                This scatter plot compares two indicators across Gateway Cities.

                Each point represents a municipality in the selected year.

                The regression line shows the average relationship across cities.
                Cities that sit far above or below the line are **statistical outliers**.

                These outliers often indicate **local structural differences** such as
                housing supply, migration patterns, labor markets, or demographic change.
                """
                    )

                    st.session_state["agent_explain"] = False
    
                # -------------------------------
                # Dynamic caption + interpretation
                # -------------------------------
                if stats.r is not None:
                    abs_r = abs(stats.r)

                    if abs_r > 0.7:
                        strength = "strong"
                    elif abs_r > 0.4:
                        strength = "moderate"
                    elif abs_r > 0.2:
                        strength = "weak"
                    else:
                        strength = "very weak"

                    direction = "positive" if stats.r > 0 else "negative"

                    st.caption(
                        f"""
                Each point represents a Gateway City. The horizontal axis shows **{xl}**
                and the vertical axis shows **{yl}** for **{sc_year}**.

                The correlation across cities is **{stats.r:.2f}**, indicating a **{strength} {direction} relationship**.
                The regression line represents the overall cross-city trend, though individual cities may deviate
                due to local economic or housing conditions.
                """
                    )
    
                if ADV:
                    c1, c2, c3 = st.columns(3)
                    if stats.r is not None:
                        c1.metric("Correlation r", f"{stats.r:+.2f}")
                    if stats.r2 is not None:
                        c2.metric("R²", f"{stats.r2:.2f}")
                    if stats.n is not None:
                        c3.metric("N (cities)", f"{stats.n}")

# ==================================================
# TAB 4: ORIGINS (B05006) - hardened + improved map
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

    bad = {
        "Total","Europe","Asia","Africa","Oceania","Latin America","Northern America",
        "Other","Other areas","Other Europe","Other Asia","Other Africa","Other Oceania",
        "Other Latin America","Other Northern America","Other and unspecified",
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


if st.session_state["active_tab"] == "Origins (B05006)":
    with st.container():

        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Global Origins of Foreign-Born Population")
        st.caption(
            "Based on ACS B05006 place-of-birth data. Country parsing uses heuristic rules and may require adjustments if ETL labels change."
        )

        cities_df = get_cities(gateway_only=True)

        origins_city = st.selectbox(
            "City for origins map",
            cities_df["place_name"].tolist(),
            index=max(
                0,
                cities_df["place_name"].tolist().index(primary_city)
            ) if primary_city in cities_df["place_name"].tolist() else 0,
            key="origins_city",
        )

        origins_fips = str(
            cities_df.loc[cities_df["place_name"] == origins_city, "place_fips"].iloc[0]
        )

        year = st.selectbox(
            "Year",
            available_years,
            index=available_years.index(selected_year),
            key="origins_year",
        )

        try:
            df_b05006 = get_place_source_table_year(origins_fips, SOURCE_BIRTH_TABLE, int(year))
        except Exception:
            df_b05006 = pd.DataFrame()

        if df_b05006 is None or df_b05006.empty:
            st.info("No B05006 rows returned for this city/year.")

        else:

            df = df_b05006.copy()

            if "estimate" not in df.columns or "variable_label" not in df.columns:
                st.warning("B05006 output missing required columns: estimate / variable_label.")

            else:

                df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
                df = df.dropna(subset=["estimate"])

                if df.empty:
                    st.info("No numeric B05006 values available.")

                else:

                    df["country_name"] = df["variable_label"].apply(label_to_country_name)
                    df = df.dropna(subset=["country_name"])

                    if df.empty:
                        st.info("No country-level rows detected in B05006.")

                    else:

                        df["iso3"] = df["country_name"].apply(country_to_iso3)

                        # -------------------------------------------------
                        # total foreign born
                        # -------------------------------------------------
                        total_fb = None

                        try:
                            mask_total = df["variable_label"].str.contains(
                                "Total foreign-born", case=False, na=False
                            )
                            if mask_total.any():
                                total_fb = float(df.loc[mask_total, "estimate"].sum())
                        except Exception:
                            total_fb = None

                        # -------------------------------------------------
                        # aggregated country dataset
                        # -------------------------------------------------
                        df_map = (
                            df.dropna(subset=["iso3"])
                            .groupby(["iso3", "country_name"], as_index=False)["estimate"]
                            .sum()
                        )

                        if df_map.empty:
                            st.warning("No mappable sovereign countries found.")

                        else:

                            # calculate share
                            color_col = "estimate"
                            labels = {"estimate": "Foreign-born population"}

                            if ADV and total_fb and total_fb > 0:
                                df_map["share"] = (df_map["estimate"] / total_fb) * 100
                                color_col = "share"
                                labels = {"share": "Share of foreign-born (%)"}

                            # -------------------------------------------------
                            # insight summary
                            # -------------------------------------------------
                            top_country = df_map.sort_values("estimate", ascending=False).iloc[0]

                            if "share" in df_map.columns:
                                st.info(
                                    f"Largest origin group: **{top_country['country_name']}**, "
                                    f"representing **{top_country['share']:.1f}%** of the foreign-born population."
                                )
                            else:
                                st.info(
                                    f"Largest origin group: **{top_country['country_name']}**, "
                                    f"with **{int(top_country['estimate']):,}** residents."
                                )

                            # -------------------------------------------------
                            # world map
                            # -------------------------------------------------
                            fig_world = px.choropleth(
                                df_map,
                                locations="iso3",
                                color=color_col,
                                hover_name="country_name",
                                hover_data={
                                    "estimate": True,
                                    "share": True if "share" in df_map.columns else False,
                                },
                                color_continuous_scale="Blues",
                                title=f"Foreign-Born Population by Country of Birth — {origins_city} ({year})",
                                labels=labels,
                            )

                            fig_world.update_geos(
                                showcoastlines=True,
                                coastlinecolor="lightgray",
                                showland=True,
                                landcolor="#f7f7f7",
                                showocean=True,
                                oceancolor="#eef3f7",
                            )

                            fig_world.update_layout(
                                template="plotly_white",
                                height=560,
                                margin=dict(l=10, r=10, t=60, b=10),
                            )

                            st.plotly_chart(
                                fig_world,
                                use_container_width=True,
                                key=f"origins_map_{origins_fips}_{year}",
                            )

                            # -------------------------------------------------
                            # top origins table
                            # -------------------------------------------------
                            st.markdown("#### Top origin countries")

                            top = (
                                df_map.sort_values("estimate", ascending=False)
                                .head(20)
                                .copy()
                            )

                            if ADV and total_fb and total_fb > 0:
                                top["share_%"] = (top["estimate"] / total_fb) * 100

                            st.dataframe(
                                top[["country_name", "estimate"] + (["share_%"] if "share_%" in top.columns else [])],
                                use_container_width=True,
                                hide_index=True,
                            )

# ==================================================
# TAB 5: ASK THE DATA (AI CENSUS QUERY) — INVESTIGATIVE
# ==================================================

if st.session_state["active_tab"] == "Ask the Data":
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)

        st.markdown("### Ask the Data")
        st.caption(
            "Ask a question about Massachusetts ACS data and the assistant will generate a chart + investigative context."
        )

        st.markdown(
            """
            **Examples**
            • Which gateway cities have the highest median household income?  
            • Where is poverty highest among gateway cities?  
            • Do places with higher renter share also have higher median rent?  
            • Compare foreign-born share vs median income across gateway cities  
            """
        )

        # ==================================================
        # CONFIG
        # ==================================================
        DEFAULT_YEAR = 2024  # latest (target)
        DEFAULT_GEO_PLACE = "place:*&in=state:25"
        DEFAULT_GEO_COUNTY = "county:*&in=state:25"

        # Curated, journalist-friendly list categorized by reporting beat
        CENSUS_VARIABLES = """
        ### DEMOGRAPHICS & POPULATION
        B01003_001E: Total population (count)
        B01002_001E: Median age (years)
        S0501_C02_001E: Foreign-born share (%)
        B03002_003E: Non-Hispanic White population (count)
        B03002_004E: Black or African American population (count)
        B03002_006E: Asian population (count)
        B03002_012E: Hispanic or Latino population (count)

        ### INCOME & INEQUALITY
        B19013_001E: Median household income ($)
        B19301_001E: Per capita income ($)
        B19083_001E: Gini Index of Income Inequality (score 0 to 1)
        S1701_C03_001E: Poverty rate (%)
        S1701_C03_002E: Child poverty rate (%)
        S1701_C03_010E: Senior (65+) poverty rate (%)

        ### HOUSING & LIVING ARRANGEMENTS
        B25077_001E: Median home value ($)
        B25064_001E: Median gross rent ($)
        S2502_C01_013E: Renter share (%)
        DP04_0142PE: Gross rent as 35% or more of household income (severe rent burden %)
        B25002_003E: Vacant housing units (count)

        ### EDUCATION & WORK
        S1501_C02_014E: High school graduate or higher share (%)
        S1501_C02_015E: Bachelor's degree or higher share (%)
        S2301_C04_001E: Unemployment rate (%)
        S0801_C01_009E: Workers commuting by public transit (%)
        S0801_C01_013E: Workers who worked from home (%)
        """.strip()

        # Expanded aliases mapping natural language to codes
        VARIABLE_ALIASES = """
        Aliases (use these in reasoning; output must still use codes):
        - "population" / "total people" -> B01003_001E
        - "median age" -> B01002_001E
        - "foreign-born share" / "immigrant share" -> S0501_C02_001E
        - "white population" -> B03002_003E
        - "black population" -> B03002_004E
        - "asian population" -> B03002_006E
        - "hispanic population" / "latino population" -> B03002_012E
        - "median income" -> B19013_001E
        - "income per person" / "per capita income" -> B19301_001E
        - "inequality" / "gini index" / "wealth gap" -> B19083_001E
        - "poverty" / "poverty rate" -> S1701_C03_001E
        - "child poverty" -> S1701_C03_002E
        - "elderly poverty" / "senior poverty" -> S1701_C03_010E
        - "median home value" / "house price" -> B25077_001E
        - "median rent" -> B25064_001E
        - "renter share" -> S2502_C01_013E
        - "rent burden" / "housing cost burden" -> DP04_0142PE
        - "vacant homes" / "vacancy" -> B25002_003E
        - "high school diploma" -> S1501_C02_014E
        - "college degree" / "bachelor's" / "highly educated" -> S1501_C02_015E
        - "unemployment" / "jobless rate" -> S2301_C04_001E
        - "public transit" / "commute by bus/train" -> S0801_C01_009E
        - "work from home" / "remote workers" -> S0801_C01_013E
        """.strip()

        # Parse allowed vars from the curated block
        ALLOWED_VARS = [
            line.split(":")[0].strip()
            for line in CENSUS_VARIABLES.splitlines()
            if ":" in line
        ]
        # Nice labels for UI / context
        VAR_LABEL = {
            line.split(":")[0].strip(): line.split(":", 1)[1].strip()
            for line in CENSUS_VARIABLES.splitlines()
            if ":" in line
        }

        # ==================================================
        # VARIABLE UNIT REGISTRY (prevents % vs count errors)
        # ==================================================

        VAR_UNIT = {}

        for var, label in VAR_LABEL.items():
            label_lower = label.lower()

            if "%" in label_lower or "share" in label_lower or "rate" in label_lower:
                VAR_UNIT[var] = "percent"
            elif "$" in label_lower or "income" in label_lower or "rent" in label_lower or "value" in label_lower:
                VAR_UNIT[var] = "dollars"
            elif "age" in label_lower:
                VAR_UNIT[var] = "years"
            else:
                VAR_UNIT[var] = "count"
        
        # ==================================================
        # SYSTEM PROMPT (tighter schema + journalist rules)
        # ==================================================
        CENSUS_QUERY_SYSTEM_PROMPT = f"""
You are a Massachusetts ACS data assistant helping journalists investigate.

Return ONLY a JSON object with these fields:
variables (list of 1 or 2 ACS codes from allowed list)
year (int)
geo (string)
chart_type ("bar"|"scatter")
x_col (string)
y_col (string)
title (string)
x_label (string)
y_label (string)

Rules:
1) variables MUST be chosen from the allowed list (exact ACS codes).
2) Prefer 1 variable for rankings (bar). Use 2 variables only for relationships (scatter).
3) If the question is about cities/towns, geo = "{DEFAULT_GEO_PLACE}"
4) If the question is about counties, geo = "{DEFAULT_GEO_COUNTY}"
5) Default year = {DEFAULT_YEAR} unless user specifies otherwise.
6) For bar charts: x_col="NAME", y_col=<variable>, chart_type="bar"
7) For scatter charts: x_col=<var1>, y_col=<var2>, chart_type="scatter"
8) Titles and axis labels should be journalist-readable (no ACS codes in labels).

Allowed variables:
{CENSUS_VARIABLES}

{VARIABLE_ALIASES}

Example (ranking):
{{
  "variables":["B19013_001E"],
  "year":{DEFAULT_YEAR},
  "geo":"{DEFAULT_GEO_PLACE}",
  "chart_type":"bar",
  "x_col":"NAME",
  "y_col":"B19013_001E",
  "title":"Median Household Income by City (Gateway Cities)",
  "x_label":"City",
  "y_label":"Median household income ($)"
}}

Example (relationship):
{{
  "variables":["S2502_C01_013E","B25064_001E"],
  "year":{DEFAULT_YEAR},
  "geo":"{DEFAULT_GEO_PLACE}",
  "chart_type":"scatter",
  "x_col":"S2502_C01_013E",
  "y_col":"B25064_001E",
  "title":"Renter Share vs Median Gross Rent (Gateway Cities)",
  "x_label":"Renter share (%)",
  "y_label":"Median gross rent ($)"
}}

Return JSON only.
""".strip()

        def split_variables_by_dataset(variables):
            groups = {
                "acs/acs5": [],
                "acs/acs5/subject": [],
                "acs/acs5/profile": []
            }

            for v in variables:
                if v.startswith("S"):
                    groups["acs/acs5/subject"].append(v)
                elif v.startswith("DP"):
                    groups["acs/acs5/profile"].append(v)
                else:
                    groups["acs/acs5"].append(v)

            return {k: v for k, v in groups.items() if v}

        # ==================================================
        # Census fetch (robust ACS handling)
        # ==================================================
        @st.cache_data(ttl=3600)
        def fetch_census_data(variables: List[str], geo: str, year: int) -> pd.DataFrame:

            import requests
            import pandas as pd
            from functools import reduce

            groups = split_variables_by_dataset(variables)

            all_dfs = []

            census_key = st.secrets.get("CENSUS_API_KEY")

            for dataset, vars_group in groups.items():

                get_cols = ",".join(vars_group) + ",NAME"
                params = {"get": get_cols}

                for part in geo.split("&"):
                    part = part.strip()
                    if part.startswith("in="):
                        params["in"] = part[3:]
                    else:
                        params["for"] = part.replace("for=", "")

                if census_key:
                    params["key"] = census_key

                base_url = f"https://api.census.gov/data/{year}/{dataset}"

                r = requests.get(base_url, params=params, timeout=20)
                r.raise_for_status()

                data = r.json()

                if not data or len(data) < 2:
                    raise ValueError(f"No data returned for {dataset} ({vars_group})")

                df = pd.DataFrame(data[1:], columns=data[0])

                # convert numeric columns
                for v in vars_group:
                    df[v] = pd.to_numeric(df[v], errors="coerce")

                all_dfs.append(df)

            # ------------------------------------------------
            # MERGE DATASETS (CRITICAL FIX)
            # ------------------------------------------------
            def merge_keys(df1, df2):
                return [c for c in ["NAME", "state", "place", "county"] if c in df1.columns and c in df2.columns]

            df_final = all_dfs[0]

            for df in all_dfs[1:]:
                keys = merge_keys(df_final, df)
                df_final = pd.merge(df_final, df, on=keys, how="outer")
                
            df_final["NAME"] = df_final["NAME"].str.replace(", Massachusetts", "", regex=False)

            return df_final

        def fetch_with_fallback_years(variables: List[str], geo: str, year: int) -> Tuple[Optional[pd.DataFrame], Optional[int], Optional[str]]:
            """
            If year fails (API/data availability), try year-1 and year-2.
            Returns (df, used_year, error_msg)
            """
            for y in [year, year - 1]:
                try:
                    df = fetch_census_data(variables, geo, int(y))
                    return df, int(y), None
                except Exception as e:
                    last_err = str(e)
            return None, None, last_err

        # ==================================================
        # AI interpreter (DeepSeek)
        # ==================================================
        def ask_ai(question: str) -> dict:
            client = OpenAI(
                api_key=st.secrets["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CENSUS_QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)

        # ==================================================
        # Helpers: validation, stats, investigative signals
        # ==================================================
        def validate_query(q: dict) -> Tuple[bool, str]:
            if not isinstance(q, dict):
                return False, "AI response is not a JSON object."

            vars_ = q.get("variables")
            if not isinstance(vars_, list) or len(vars_) < 1 or len(vars_) > 2:
                return False, "variables must be a list of length 1 or 2."

            vars_ = [str(v).strip() for v in vars_]
            vars_ = [v for v in vars_ if v in ALLOWED_VARS]
            q["variables"] = vars_
            if not q["variables"]:
                return False, "AI requested unsupported variables."

            year = q.get("year", DEFAULT_YEAR)
            try:
                q["year"] = int(year)
            except Exception:
                q["year"] = DEFAULT_YEAR

            geo = str(q.get("geo", DEFAULT_GEO_PLACE)).strip()
            # lock geo to the two supported patterns (prevents weird API queries)
            if "county:*" in geo:
                q["geo"] = DEFAULT_GEO_COUNTY
            else:
                q["geo"] = DEFAULT_GEO_PLACE

            chart_type = str(q.get("chart_type", "bar")).strip().lower()
            if chart_type not in ["bar", "scatter"]:
                chart_type = "bar"
            q["chart_type"] = chart_type

            # enforce consistent x/y behavior
            if q["chart_type"] == "bar":
                q["x_col"] = "NAME"
                q["y_col"] = q.get("y_col") if q.get("y_col") in q["variables"] else q["variables"][0]
            else:
                # scatter must use variable codes on axes
                if len(q["variables"]) < 2:
                    q["chart_type"] = "bar"
                    q["x_col"] = "NAME"
                    q["y_col"] = q["variables"][0]
                else:
                    q["x_col"] = q.get("x_col") if q.get("x_col") in q["variables"] else q["variables"][0]
                    other = q["variables"][1] if q["x_col"] == q["variables"][0] else q["variables"][0]
                    q["y_col"] = q.get("y_col") if q.get("y_col") in q["variables"] else other

            # reasonable defaults for titles/labels
            def_label_y = VAR_LABEL.get(q["y_col"], q["y_col"])
            def_label_x = "City" if q["chart_type"] == "bar" else VAR_LABEL.get(q["x_col"], q["x_col"])
            q["title"] = str(q.get("title") or "ACS Comparison (Gateway Cities)").strip()
            q["x_label"] = str(q.get("x_label") or def_label_x).strip()
            q["y_label"] = str(q.get("y_label") or def_label_y).strip()

            return True, ""

        def compute_rank_context(df: pd.DataFrame, y: str, selected_name: str) -> dict:
            out = {"mean": None, "median": None, "rank": None, "n": None, "value": None, "z": None}
            if df is None or df.empty or y not in df.columns:
                return out
            ser = pd.to_numeric(df[y], errors="coerce")
            d = df.copy()
            d[y] = ser
            d = d.dropna(subset=[y])
            if d.empty:
                return out

            out["mean"] = float(d[y].mean())
            out["median"] = float(d[y].median())
            out["n"] = int(len(d))

            if selected_name in d["NAME"].values:
                v = float(d.loc[d["NAME"] == selected_name, y].iloc[0])
                out["value"] = v
                ranks = d[y].rank(ascending=False, method="min")
                out["rank"] = int(ranks[d["NAME"] == selected_name].iloc[0])

                std = float(d[y].std(ddof=0))
                if std > 0:
                    out["z"] = (v - out["mean"]) / std
            return out

        def compute_scatter_stats(df: pd.DataFrame, x: str, y: str) -> dict:
            out = {"r": None, "r2": None, "slope": None, "intercept": None, "n": None}
            if df is None or df.empty or x not in df.columns or y not in df.columns:
                return out
            d = df.copy()
            d[x] = pd.to_numeric(d[x], errors="coerce")
            d[y] = pd.to_numeric(d[y], errors="coerce")
            d = d.dropna(subset=[x, y])
            if len(d) < 3:
                out["n"] = int(len(d))
                return out

            xx = d[x].to_numpy()
            yy = d[y].to_numpy()

            try:
                r = float(np.corrcoef(xx, yy)[0, 1])
            except Exception:
                r = None
            out["r"] = r
            out["r2"] = float(r * r) if r is not None else None
            out["n"] = int(len(d))

            try:
                m, b = np.polyfit(xx, yy, 1)
                out["slope"] = float(m)
                out["intercept"] = float(b)
            except Exception:
                pass

            return out

        def detect_outliers_z(df: pd.DataFrame, y: str, z_thresh: float = 2.0) -> pd.DataFrame:
            if df is None or df.empty or y not in df.columns:
                return pd.DataFrame()
            d = df.copy()
            d[y] = pd.to_numeric(d[y], errors="coerce")
            d = d.dropna(subset=[y])
            if len(d) < 8:
                return pd.DataFrame()
            mu = d[y].mean()
            sd = d[y].std(ddof=0)
            if sd == 0 or pd.isna(sd):
                return pd.DataFrame()
            d["z"] = (d[y] - mu) / sd
            outs = d[np.abs(d["z"]) >= z_thresh].copy()
            outs = outs.sort_values("z", ascending=False)
            return outs[["NAME", y, "z"]]

        # ==================================================
        # UI CONTROLS (journalist-friendly)
        # ==================================================
        left, right = st.columns([2.2, 1.2])

        with left:
            question = st.text_input(
                "",
                placeholder="Example: Do gateway cities with higher renter share also have higher median rent?",
                key="ask_data_question",
            )
        with right:
            ui_year = st.selectbox(
                "Default year",
                options=[2024, 2023, 2022, 2021],
                index=0,
                help="Used when the question doesn't specify a year.",
                key="ask_data_default_year",
            )
            DEFAULT_YEAR = int(ui_year)  # override default

        cA, cB, cC = st.columns([1.2, 1.0, 1.0])
        with cA:
            restrict_gateway = st.toggle(
                "Restrict to Gateway Cities",
                value=True,
                help="If off, results include all MA places or counties.",
                key="ask_data_gateway_only",
            )
        with cB:
            top_n = st.selectbox("Top N (bar charts)", [10, 15, 20, 25, 39], index=2, key="ask_data_topn")
        with cC:
            show_table = st.toggle("Show data table", value=True, key="ask_data_show_table")

        submit = st.button("Search", key="ask_data_submit")

        # ==================================================
        # EXECUTION
        # ==================================================
        if submit and question:
            with st.spinner("Interpreting question..."):
                try:
                    query = ask_ai(question)
                except Exception as e:
                    st.error(f"AI error: {e}")
                    query = None

            if query:
                ok, msg = validate_query(query)
                with st.expander("AI interpretation (validated)"):
                    st.json(query)

                if not ok:
                    st.error(msg)
                    st.stop()

                # Enforce the chosen default year if AI returned none / weird
                if not query.get("year"):
                    query["year"] = DEFAULT_YEAR

                # Fetch with fallback years
                with st.spinner("Fetching ACS data..."):
                    df, used_year, err = fetch_with_fallback_years(
                        query["variables"], query["geo"], int(query["year"])
                    )
                    if df is None:
                        st.error(f"ACS API error (year {query['year']} and fallbacks): {err}")
                        st.stop()

                # Filter to gateway cities if needed and geo is place-level
                if restrict_gateway and "place:*" in query["geo"]:
                    gateway_names = [c.replace(", Massachusetts", "") for c in gateway_city_options]
                    df = df[df["NAME"].isin(gateway_names)].copy()

                if df is None or df.empty:
                    st.warning("No data returned for that query.")
                    st.stop()

                # Selected city name (for highlighting + context)
                selected_city_full = st.session_state.get("selected_city", "")
                selected_city_name = selected_city_full.split(",")[0] if selected_city_full else ""

                chart_type = query["chart_type"]
                x = query["x_col"]
                y = query["y_col"]

                # ==================================================
                # INVESTIGATIVE CONTEXT (bar)
                # ==================================================
                if chart_type == "bar":
                    y = y if y in df.columns else query["variables"][0]
                    df[y] = pd.to_numeric(df[y], errors="coerce")

                    df_clean = df.dropna(subset=[y]).copy()
                    df_sorted = df_clean.sort_values(by=y, ascending=False).head(int(top_n))

                    # context summary
                    ctx = compute_rank_context(df_clean, y, selected_city_name)

                    # pretty labels
                    y_label = query.get("y_label") or VAR_LABEL.get(y, y)
                    title = (query.get("title") or "ACS Ranking").strip()
                    title = f"{title} — {used_year}" if used_year else title

                    # story leads: outliers
                    outs = detect_outliers_z(df_clean, y, z_thresh=2.0)

                    # display info panel
                    info_lines = []
                    if ctx["mean"] is not None:
                        info_lines.append(f"Average: **{ctx['mean']:,.2f}**")
                    if ctx["median"] is not None:
                        info_lines.append(f"Median: **{ctx['median']:,.2f}**")
                    if ctx["rank"] is not None and ctx["n"] is not None:
                        info_lines.append(f"**{selected_city_name} rank:** {ctx['rank']} of {ctx['n']}")
                    if ctx["z"] is not None:
                        info_lines.append(f"Z-score (vs gateway distribution): **{ctx['z']:+.2f}**")

                    st.info("Gateway City Context\n\n" + "\n\n".join(info_lines) if info_lines else "Context unavailable.")

                    # chart
                    fig = px.bar(
                        df_sorted,
                        x=y,
                        y="NAME",
                        orientation="h",
                        title=title,
                        labels={y: y_label, "NAME": "City"},
                    )
                    fig.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), height=560)
                    st.plotly_chart(fig, use_container_width=True)

                    # outliers
                    if outs is not None and not outs.empty:
                        st.markdown("#### Investigative leads: statistical outliers")
                        st.caption("Cities with unusually high/low values relative to the gateway distribution (|z| ≥ 2).")
                        st.dataframe(outs, use_container_width=True, hide_index=True)

                    if show_table:
                        st.markdown("#### Data")
                        st.dataframe(df_clean[["NAME", y]].sort_values(y, ascending=False), use_container_width=True, hide_index=True)

                    st.download_button(
                        "Download CSV",
                        df_clean.to_csv(index=False),
                        "acs_ask_the_data.csv",
                        "text/csv",
                        key="ask_data_download_bar",
                    )

                # ==================================================
                # INVESTIGATIVE CONTEXT (scatter)
                # ==================================================
                else:
                    # enforce numeric scatter
                    if x not in df.columns or y not in df.columns:
                        st.error("Scatter requires two valid variable columns.")
                        st.stop()

                    d = df.copy()
                    d[x] = pd.to_numeric(d[x], errors="coerce")
                    d[y] = pd.to_numeric(d[y], errors="coerce")
                    d = d.dropna(subset=[x, y]).copy()

                    if len(d) < 5:
                        st.warning("Not enough data points to compute a relationship.")
                        st.stop()

                    stats = compute_scatter_stats(d, x, y)

                    # highlight selected city if present
                    d["selected"] = d["NAME"] == selected_city_name

                    x_label = query.get("x_label") or VAR_LABEL.get(x, x)
                    y_label = query.get("y_label") or VAR_LABEL.get(y, y)
                    title = (query.get("title") or "ACS Relationship").strip()
                    title = f"{title} — {used_year}" if used_year else title

                    fig = px.scatter(
                        d,
                        x=x,
                        y=y,
                        hover_name="NAME",
                        title=title,
                        labels={x: x_label, y: y_label},
                    )

                    # OLS line if available
                    if stats["slope"] is not None and stats["intercept"] is not None:
                        xx = np.linspace(float(d[x].min()), float(d[x].max()), 80)
                        yy = stats["slope"] * xx + stats["intercept"]
                        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="OLS fit"))

                        # simple outlier marking via residuals (2*sd)
                        d["pred"] = stats["slope"] * d[x] + stats["intercept"]
                        d["resid"] = d[y] - d["pred"]
                        r_sd = float(d["resid"].std(ddof=0)) if len(d) else 0.0
                        d["outlier"] = np.abs(d["resid"]) >= (2.0 * r_sd) if r_sd > 0 else False

                        outs = d[d["outlier"]].copy()
                    else:
                        outs = pd.DataFrame()

                    fig.update_layout(template="plotly_white", height=560)
                    st.plotly_chart(fig, use_container_width=True)

                    # relationship summary
                    rel_lines = []
                    if stats["r"] is not None:
                        rel_lines.append(f"Correlation (r): **{stats['r']:+.2f}**")
                    if stats["r2"] is not None:
                        rel_lines.append(f"R²: **{stats['r2']:.2f}**")
                    if stats["n"] is not None:
                        rel_lines.append(f"N: **{stats['n']}**")
                    if stats["slope"] is not None:
                        rel_lines.append(f"OLS slope: **{stats['slope']:+.3g}** (per 1 unit of x)")
                    st.info("Relationship summary\n\n" + "\n\n".join(rel_lines))

                    # investigative leads
                    if outs is not None and not outs.empty:
                        st.markdown("#### Investigative leads: relationship outliers")
                        st.caption("Cities far above/below the fitted trend (|residual| ≥ 2σ).")
                        show_cols = ["NAME", x, y, "resid"]
                        st.dataframe(outs[show_cols].sort_values("resid", ascending=False), use_container_width=True, hide_index=True)

                    if show_table:
                        st.markdown("#### Data")
                        st.dataframe(d[["NAME", x, y]].sort_values(y, ascending=False), use_container_width=True, hide_index=True)

                    st.download_button(
                        "Download CSV",
                        d.to_csv(index=False),
                        "acs_ask_the_data.csv",
                        "text/csv",
                        key="ask_data_download_scatter",
                    )
            
# ==================================================
# TAB 6: METHODOLOGY (Academic only; tied to mode, not extra toggles)
# ==================================================

if st.session_state["active_tab"] == "Methodology":
    with st.container():
        st.markdown('<span class="section-card-marker"></span>', unsafe_allow_html=True)
        st.markdown("### Methodology & Notes (Academic Mode)")

        st.markdown(
            """
**Data source**  
- American Community Survey (ACS) **5-year** estimates, using **endpoint years** (e.g., “2022” represents the 2018–2022 pooled estimate).

**Interpretation**  
- 5-year ACS smooths year-to-year volatility; endpoints should be interpreted as rolling windows, not point-in-time measurements.

**Ranking / Distribution context**  
- For a given metric and year, *Gateway* distribution is taken across Gateway cities for that year.
- Z-score shown (when Advanced is enabled) is: *(city_value − gateway_mean) / gateway_std* (population weighting not applied unless your query layer enforces it).

**Scatter statistics**  
- Pearson correlation r and OLS line (if shown) reflect cross-sectional association across Gateway cities for the selected year.
- No causal claims are implied.

**B05006 origins**  
- Country parsing is heuristic; validate label conventions in your ETL and adjust parsing rules if needed.
            """
        )