import requests
import pandas as pd

# --------------------------------------------------
# Official MA Gateway Cities (exact Census names)
# --------------------------------------------------

GATEWAY_CITIES = {
    'Attleboro city, Massachusetts',
    'Barnstable Town city, Massachusetts',
    'Brockton city, Massachusetts',
    'Chelsea city, Massachusetts',
    'Chicopee city, Massachusetts',
    'Everett city, Massachusetts',
    'Fall River city, Massachusetts',
    'Fitchburg city, Massachusetts',
    'Haverhill city, Massachusetts',
    'Holyoke city, Massachusetts',
    'Lawrence city, Massachusetts',
    'Leominster city, Massachusetts',
    'Lowell city, Massachusetts',
    'Lynn city, Massachusetts',
    'Malden city, Massachusetts',
    'Methuen Town city, Massachusetts',
    'New Bedford city, Massachusetts',
    'Peabody city, Massachusetts',
    'Pittsfield city, Massachusetts',
    'Quincy city, Massachusetts',
    'Revere city, Massachusetts',
    'Salem city, Massachusetts',
    'Springfield city, Massachusetts',
    'Taunton city, Massachusetts',
    'Westfield city, Massachusetts',
    'Worcester city, Massachusetts'
}

# --------------------------------------------------
# Fetch all MA places from Census API
# --------------------------------------------------

print("Fetching Massachusetts place list from Census API...")

url = "https://api.census.gov/data/2022/acs/acs5"
params = {
    "get": "NAME",
    "for": "place:*",
    "in": "state:25"
}

response = requests.get(url, params=params, timeout=15)
response.raise_for_status()
data = response.json()

df = pd.DataFrame(data[1:], columns=data[0])

# --------------------------------------------------
# Build full 7-digit GEOID (state + place)
# --------------------------------------------------

df["place_fips"] = "25" + df["place"].str.zfill(5)
df["place_name"] = df["NAME"]
df["is_gateway_city"] = df["NAME"].isin(GATEWAY_CITIES)

# Keep only required columns
df = df[["place_fips", "place_name", "is_gateway_city"]]

# Sort for cleanliness
df = df.sort_values("place_name").reset_index(drop=True)

# --------------------------------------------------
# Sanity Check
# --------------------------------------------------

gateway_count = df["is_gateway_city"].sum()
print(f"Gateway cities detected: {gateway_count}")

if gateway_count != 26:
    print("⚠ WARNING: Gateway city count is not 26. Check name mismatches.")
else:
    print("✓ Correct number of gateway cities identified.")

# --------------------------------------------------
# Save CSV
# --------------------------------------------------

output_file = "gateway_cities_correct_full_geoid.csv"
df.to_csv(output_file, index=False)

print(f"\nCSV successfully generated: {output_file}")