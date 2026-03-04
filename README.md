# Gateway Cities – Immigration & Economic Trends Dashboard  
**CityHack 2026 – GBH Gateway Cities Challenge Submission**

## Live Demo
**Interactive Dashboard:**  
https://ma-gateway-cities.streamlit.app

---

# 1. Project Overview

This project analyzes **demographic and economic trends in Massachusetts Gateway Cities** using data from the **U.S. Census American Community Survey (ACS) 5-Year Estimates**.

The system combines:

- ACS Census API ingestion
- Large-scale data engineering
- Investigative trend analysis
- Interactive visualizations
- AI-assisted data exploration

to support **journalists, civic researchers, and policy analysts** in identifying meaningful demographic and economic shifts.

Unlike traditional dashboards that query Census APIs directly, this system required building a **structured analytics platform capable of transforming raw ACS data into journalist-ready metrics**.

The dashboard is built with **Streamlit**, while the underlying data infrastructure is powered by **Supabase (PostgreSQL)**.

Key goals include:

- Detecting demographic change across Gateway Cities
- Surfacing structural economic differences between cities
- Enabling fast fact-checking and story discovery
- Supporting data-driven reporting

---

# 2. Research Focus

Primary emphasis:

**Foreign-born population trends and economic conditions across Massachusetts cities.**

Gateway Cities represent historically industrial communities undergoing economic and demographic transformation.

The dashboard focuses on identifying patterns such as:

- Immigration concentration
- Economic opportunity gaps
- Housing pressure
- Structural poverty differences
- Changing labor market dynamics

---

# 3. Core Investigative Questions

1. Which Gateway Cities have the **largest foreign-born populations**?
2. Which cities are experiencing the **fastest growth in immigrant populations**?
3. How do **countries of origin differ across cities**?
4. Are immigration patterns associated with changes in:

   - Median household income  
   - Poverty rate  
   - Housing cost burden  
   - Employment levels  
   - Educational attainment  

5. Which cities diverge most strongly from **statewide economic trends**?

---

# 4. Data Sources

All data is sourced from the **U.S. Census Bureau – American Community Survey (ACS) 5-Year Estimates**.

Primary source:

https://www.census.gov/programs-surveys/acs/data.html

Reference tools used during development:

- https://data.census.gov
- https://censusreporter.org
- https://buspark.io/documentation/project-guides/census_tutorial

The dashboard retrieves data via the **Census API** and stores structured datasets in **Supabase PostgreSQL** for analysis.

---

# 5. ACS Tables Used

### Immigration & Demographics

| Table | Description |
|------|-------------|
| B05006 | Place of Birth (Foreign-Born by Country) |
| B05015 | Year of Entry for Foreign-Born Population |
| B01003 | Total Population |
| S0501 | Selected Characteristics of the Foreign-Born Population |

### Income & Economic Indicators

| Table | Description |
|------|-------------|
| S1901 | Household Income |
| S1701 | Poverty Status |
| B19083 | Gini Index (Income Inequality) |
| B23025 | Employment Status |
| S1501 | Educational Attainment |

### Housing

| Table | Description |
|------|-------------|
| B25002 | Housing Occupancy |
| B25003 | Owner vs Renter Occupancy |
| B25034 | Year Structure Built |
| B25077 | Median Home Value |
| B25070 | Rent as Percentage of Income |

### Transportation

| Table | Description |
|------|-------------|
| B08301 | Means of Transportation to Work |
| B08126 | Travel Time to Work |

---

# 6. Geographic Scope

The dashboard analyzes:

### Massachusetts Gateway Cities (MassINC Definition)

26 Gateway Cities including:

- Brockton  
- Chelsea  
- Chicopee  
- Everett  
- Fall River  
- Fitchburg  
- Haverhill  
- Holyoke  
- Lawrence  
- Lowell  
- Lynn  
- Malden  
- Methuen  
- New Bedford  
- Peabody  
- Pittsfield  
- Quincy  
- Revere  
- Salem  
- Springfield  
- Taunton  
- Westfield  
- Worcester  

### Comparison Cities

- Boston  
- Cambridge  
- Weymouth  
- Marlborough  

### Statewide Baseline

Massachusetts totals are included as a **reference benchmark**.

---

# 7. Time Coverage

The dashboard uses **ACS 5-Year Estimates covering 2010–latest available year**.

Because ACS releases data with a lag, the most recent year depends on Census publication cycles.

---

# 8. Data Engineering Architecture (Supabase)

A significant portion of the project involved building a **data engineering pipeline to transform raw Census API outputs into a structured analytics database**.

The ACS API provides data in a **wide format designed for statistical tables**, not for investigative analytics. As a result, substantial transformation was required.

The system therefore uses **Supabase PostgreSQL as a structured data warehouse layer**.

### Architecture Overview

```
Census API
   ↓
Python ingestion scripts
   ↓
Normalization & cleaning
   ↓
Long-format metric warehouse
   ↓
Supabase PostgreSQL
   ↓
SQL query layer
   ↓
Streamlit dashboard
```

---

# 9. Census Data Ingestion Pipeline

### 9.1 API Extraction

The pipeline programmatically retrieves ACS tables using the **Census API**.

Example endpoint structure:

```
https://api.census.gov/data/{year}/acs/acs5
```

The system automatically determines the correct dataset:

| Table Prefix | Dataset |
|--------------|--------|
| Bxxxx | acs5 |
| Sxxxx | acs5/subject |
| DPxx | acs5/profile |

Extraction logic includes:

- automated year iteration
- variable discovery
- dataset routing
- error handling for unavailable variables

---

# 10. Data Normalization

Raw Census API outputs return **wide tables where each variable is a separate column**.

To enable scalable analysis, the pipeline converts data into a **long-format metric warehouse**.

Example transformation:

Wide ACS format:

| city | B01003_001E | B19013_001E |
|-----|-------------|-------------|
| Chelsea | 40121 | 62750 |

Long normalized format:

| city | year | variable | value |
|-----|------|----------|------|
| Chelsea | 2022 | total_population | 40121 |
| Chelsea | 2022 | median_income | 62750 |

Benefits include:

- simplified aggregation
- scalable metric additions
- faster analytical queries
- consistent schema across tables

---

# 11. Geographic Standardization

Census place names often appear in inconsistent formats.

Examples:

```
Chelsea city, Massachusetts
Chelsea CDP
CHELSEA
```

A geographic normalization layer was implemented to:

- standardize city names
- map Census places to Gateway Cities
- ensure consistent joins across datasets
- align geographic units with the GeoJSON map layer

This process involved:

- place-name cleaning
- uppercase normalization
- custom city mapping tables
- FIPS code validation

---

# 12. Metric Catalog Layer

To simplify analysis for journalists, the system includes a **metric catalog abstraction layer**.

This maps raw ACS variables into human-readable indicators.

| Metric | ACS Variable |
|------|--------------|
| total_population | B01003_001E |
| median_household_income | B19013_001E |
| poverty_rate | S1701_C03_001E |
| foreign_born_share | S0501_C02_001E |
| renter_share | S2502_C01_013E |

This allows the dashboard to query **semantic metrics instead of raw Census variables**.

---

# 13. Query Optimization

The Supabase layer enables efficient analytics queries including:

- city-level trend analysis
- cross-city metric comparisons
- ranking queries
- scatterplot datasets
- time-series aggregation

SQL queries are wrapped in reusable Python functions such as:

- `get_gateway_metric_snapshot()`
- `get_gateway_metric_trend()`
- `get_gateway_ranking()`
- `get_gateway_scatter()`

This architecture separates:

- data retrieval
- analysis logic
- visualization logic

---

# 14. Dashboard Architecture

The Streamlit application contains several investigative modules.

### Map View

Interactive Massachusetts map showing:

- city-level ACS metrics
- comparative shading
- geographic context

### Investigative Themes

Prebuilt analyses including:

- immigration concentration
- housing pressure
- economic mobility
- poverty patterns

### Metric Comparison

Allows cross-city comparison of indicators such as:

- income
- rent burden
- foreign-born share
- educational attainment

### Country-of-Origin Analysis

Foreign-born populations broken down by **country of birth** using **table B05006**.

### Ask the Data (AI Exploration)

Users can ask natural-language questions such as:

```
Which Gateway Cities have the highest foreign-born population?
```

The system translates queries into **metric lookups and visualizations**.

### Methodology Panel

Provides transparency regarding:

- data sources
- ACS methodology
- definitions
- limitations

---

# 15. Investigative Analytics

The dashboard identifies patterns including:

### Trend Detection

- multi-year slope analysis
- growth-rate ranking
- city-level trend comparison

### Outlier Detection

Outliers are identified using:

- percent change
- distribution comparison
- deviation from statewide averages

### Comparative Analysis

Cities can be compared across metrics including:

- immigration
- housing
- labor markets
- income inequality

---

# 16. Ethical & Responsible Data Use

### Transparency

All data sources are:

- publicly accessible
- documented
- reproducible

### Interpretation Limits

The system **does not imply causation** between variables.

It highlights **correlations and patterns** to support responsible reporting.

### Privacy

All data used is **aggregated public Census data**.

No individual-level data is included.

### ACS Limitations

- ACS 5-year estimates smooth short-term fluctuations
- margins of error may be large for small populations
- demographic changes may reflect sampling variation

---

# 17. Reproducibility

### Requirements

- Python 3.9+
- Streamlit
- pandas
- requests
- plotly
- SQLAlchemy
- Supabase PostgreSQL

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

---

# 18. Project Structure

```
project/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── ma_municipalities.geojson
│
├── src/
│   ├── queries.py
│   ├── census_fetch.py
│   └── story_angles.py
│
└── assets/
    └── styles.css
```

---

# 19. Intended Impact

This project aims to support:

- accountability journalism
- immigration reporting
- local economic analysis
- public-interest data transparency

By transforming complex Census datasets into interpretable civic insights, the dashboard enables journalists to:

- identify story leads
- verify claims with data
- surface underreported demographic trends

---

# 20. Future Development

Planned enhancements include:

- tract-level spatial analysis
- immigration cohort trend modeling
- automated anomaly detection
- downloadable city-level story briefs
- expanded economic indicator coverage

---

# 21. Contact

CityHack 2026 Team  
Gateway Cities Challenge Submission