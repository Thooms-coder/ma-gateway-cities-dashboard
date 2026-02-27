STORY_ANGLES = {

  # ==========================================================
  # 1️⃣ Housing Pressure
  # ==========================================================
  "housing_pressure": {
    "title": "Housing Pressure",
    "metrics": [
      "rent_burden_30_plus",
      "severe_rent_burden_50_plus",
      "median_home_value",
      "percent_renters",
      "vacancy_rate",
      "median_income",
    ],
    "investigative_pairs": [
      ("rent_burden_30_plus", "poverty_rate"),
      ("rent_burden_30_plus", "median_income"),
      ("severe_rent_burden_50_plus", "vacancy_rate"),
      ("percent_renters", "median_home_value"),
    ],
  },


  # ==========================================================
  # 2️⃣ Economic Stress
  # ==========================================================
  "economic_stress": {
    "title": "Economic Stress",
    "metrics": [
      "poverty_rate",
      "child_poverty_rate",
      "median_income",
      "unemployment_rate",
      "gini_index",
      "rent_burden_30_plus",
    ],
    "investigative_pairs": [
      ("poverty_rate", "median_income"),
      ("poverty_rate", "unemployment_rate"),
      ("median_income", "gini_index"),
      ("rent_burden_30_plus", "poverty_rate"),
    ],
  },


  # ==========================================================
  # 3️⃣ Demographic Change
  # ==========================================================
  "demographic_change": {
    "title": "Demographic Change",
    "metrics": [
      "total_population",
      "foreign_born_share",
      "hispanic_share",
      "black_share",
      "asian_share",
      "ba_or_higher",
    ],
    "investigative_pairs": [
      ("foreign_born_share", "poverty_rate"),
      ("hispanic_share", "median_income"),
      ("total_population", "rent_burden_30_plus"),
      ("ba_or_higher", "median_income"),
    ],
  },


  # ==========================================================
  # 4️⃣ Workforce Stability
  # ==========================================================
  "workforce_stability": {
    "title": "Workforce Stability",
    "metrics": [
      "unemployment_rate",
      "labor_force_participation",
      "median_income",
      "poverty_rate",
      "hs_or_higher",
      "ba_or_higher",
      "median_earnings_25plus",
    ],
    "investigative_pairs": [
      ("unemployment_rate", "poverty_rate"),
      ("labor_force_participation", "median_income"),
      ("ba_or_higher", "median_income"),
      ("hs_or_higher", "poverty_rate"),
    ],
  },


  # ==========================================================
  # 5️⃣ Education & Mobility (NEW – Strongest Addition)
  # ==========================================================
  "education_mobility": {
    "title": "Education & Economic Mobility",
    "metrics": [
      "hs_or_higher",
      "ba_or_higher",
      "median_earnings_25plus",
      "poverty_rate",
      "median_income",
    ],
    "investigative_pairs": [
      ("ba_or_higher", "median_income"),
      ("hs_or_higher", "poverty_rate"),
      ("ba_or_higher", "poverty_rate"),
      ("median_earnings_25plus", "poverty_rate"),
    ],
  },

}