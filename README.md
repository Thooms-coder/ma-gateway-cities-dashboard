# Gateway Cities – Immigration & Economic Trends Dashboard  
**CityHack 2026 – GBH Gateway Cities Challenge Submission**

---

# 1. Project Overview

This project analyzes **demographic and economic trends in Massachusetts Gateway Cities** using data from the **U.S. Census American Community Survey (ACS) 5-Year Estimates**.

The system combines:

- **ACS Census API queries**
- **Investigative trend analysis**
- **Interactive visualizations**
- **AI-assisted data exploration**

to support **journalists, civic researchers, and policy analysts** in identifying meaningful demographic and economic shifts.

The dashboard is built with **Streamlit** and designed specifically for **investigative data workflows** rather than general business analytics.

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

The dashboard accesses ACS data **directly through the Census API**.

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
- plus additional MassINC-designated municipalities.

### Comparison Cities

To contextualize trends:

- Boston  
- Cambridge  
- Weymouth  
- Marlborough  

### Statewide Baseline

Massachusetts totals are included as a **reference benchmark**.

---

# 7. Time Coverage

The dashboard uses **ACS 5-Year Estimates** covering:

2010 – Latest Available ACS Release

Because ACS is released with a lag, the latest available year depends on Census publication cycles.

---

# 8. Data Retrieval Architecture

Data is retrieved dynamically from the **Census API**.

Example query structure:

https://api.census.gov/data/{year}/acs/acs5/subject


The system automatically selects the correct dataset depending on the ACS table type:

| Table Prefix | Dataset |
|---------------|--------|
| Bxxxx | acs5 |
| Sxxxx | acs5/subject |
| DPxx | acs5/profile |

The query engine includes:

- automatic dataset detection
- year fallback logic
- numeric coercion and validation
- caching via Streamlit

---

# 9. Data Processing Pipeline

### 9.1 Retrieval

- ACS API queried by table and year
- Massachusetts place-level geography
- Gateway city filtering applied

### 9.2 Cleaning

Data cleaning includes:

- removal of malformed rows
- numeric coercion
- unit validation
- handling of missing estimates

### 9.3 Validation

The pipeline validates:

- percentage variables
- dataset selection
- year availability
- value ranges

This prevents common ACS issues such as:

- counts labeled as percentages
- incorrect dataset endpoints
- clipped values

---

# 10. Dashboard Architecture

The Streamlit application contains several investigative modules.

### Map View

Interactive map of Massachusetts cities showing:

- selected ACS metrics
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

Foreign-born populations broken down by **country of birth** using table **B05006**.

### Ask the Data (AI Exploration)

Users can ask natural-language questions such as:

Which Gateway Cities have the highest foreign-born population?


The AI agent translates queries into **Census API calls and visualizations**.

### Methodology Panel

Provides transparency regarding:

- data sources
- limitations
- definitions
- ACS methodology

---

# 11. Investigative Analytics

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

# 12. Ethical & Responsible Data Use

### Transparency

All data sources are:

- publicly accessible
- documented
- reproducible

### Interpretation Limits

The system **does not imply causation** between variables.

It highlights **correlations and patterns** to support responsible reporting.

### Privacy

All data is **aggregated public Census data**.

No individual-level data is used.

### ACS Limitations

- ACS 5-year estimates smooth short-term fluctuations
- margins of error can be large for small populations
- demographic changes may reflect sampling variation

---

# 13. Reproducibility

### Requirements

- Python 3.9+
- Streamlit
- pandas
- requests
- plotly

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run app.py
```

---

# 14. Project Structure

project/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── ma_municipalities.geojson
│
├── src/
│   ├── queries.py
│   ├── story_angles.py
│   ├── census_fetch.py
│
└── assets/
    ├── styles.css
   
---