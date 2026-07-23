# -*- coding: utf-8 -*-
"""
Script 16 — Production Zone Assignment for APMC Markets
=========================================================
Assigns every APMC market in the TOP price panel to its nearest
production zone by Haversine distance between the market's district
centroid (geocoded via OpenStreetMap Nominatim) and the zone centroid.

Processing:
  1. Read market panels → unique (crop, state, district) combinations
  2. Geocode each district via Nominatim (cached to avoid repeat calls)
  3. Fallback: state centroid if district geocoding fails
  4. Compute Haversine distance to all same-crop zone centroids
  5. Assign nearest zone
  6. Save zone_assignment.csv + diagnostic figures

Outputs:
  data/zone_assignment.csv
      market_id, market, crop, state, district,
      lat, lon, geocode_level,
      zone_id, zone_market, zone_state, dist_km

Run BEFORE Script 14 (Script 14 reads zone_assignment.csv if present).

Usage:
  python scripts/16_Zone_Assignment.py
"""

import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJ      = Path(__file__).resolve().parent.parent
PANEL_DIR = Path(r'C:\Users\masro\Downloads\Agmarknet_Weekly')
DATA_DIR  = PROJ / 'data'
FIG_DIR   = PROJ / 'Model_Output'

CACHE_FILE  = DATA_DIR / 'geocode_cache.json'
OUT_FILE    = DATA_DIR / 'zone_assignment.csv'

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Production zone centroids (17 zones)
# ─────────────────────────────────────────────────────────────────────────────
ZONES = {
    # Tomato
    'T1_Kolar':          {'crop': 'tomato', 'lat': 13.1390, 'lon': 78.1320, 'zone_market': 'Kolar APMC',        'zone_state': 'Karnataka'},
    'T2_Madanapalle':    {'crop': 'tomato', 'lat': 13.5504, 'lon': 78.5025, 'zone_market': 'Madanapalle APMC',  'zone_state': 'Andhra Pradesh'},
    'T3_Nashik_Tomato':  {'crop': 'tomato', 'lat': 19.9975, 'lon': 73.7898, 'zone_market': 'Nashik APMC',       'zone_state': 'Maharashtra'},
    'T4_Solan':          {'crop': 'tomato', 'lat': 30.9045, 'lon': 77.1167, 'zone_market': 'Solan APMC',        'zone_state': 'Himachal Pradesh'},
    'T5_Navsari':        {'crop': 'tomato', 'lat': 20.9476, 'lon': 72.9031, 'zone_market': 'Navsari APMC',      'zone_state': 'Gujarat'},
    # Onion
    'O1_Lasalgaon':      {'crop': 'onion',  'lat': 20.4061, 'lon': 74.0088, 'zone_market': 'Lasalgaon APMC',    'zone_state': 'Maharashtra'},
    'O2_Pimpalgaon':     {'crop': 'onion',  'lat': 20.0833, 'lon': 74.0167, 'zone_market': 'Pimpalgaon APMC',   'zone_state': 'Maharashtra'},
    'O3_Mahuva':         {'crop': 'onion',  'lat': 21.0888, 'lon': 71.7744, 'zone_market': 'Mahuva APMC',       'zone_state': 'Gujarat'},
    'O6_Hubli':          {'crop': 'onion',  'lat': 15.3647, 'lon': 75.1239, 'zone_market': 'Hubli APMC',        'zone_state': 'Karnataka'},
    'O7_Solapur':        {'crop': 'onion',  'lat': 17.6854, 'lon': 75.9064, 'zone_market': 'Solapur APMC',      'zone_state': 'Maharashtra'},
    'O8_Manmad':         {'crop': 'onion',  'lat': 20.2500, 'lon': 74.4367, 'zone_market': 'Manmad APMC',       'zone_state': 'Maharashtra'},
    'O9_Kurnool':        {'crop': 'onion',  'lat': 15.8281, 'lon': 78.0373, 'zone_market': 'Kurnool APMC',      'zone_state': 'Andhra Pradesh'},
    'O10_Gondal':        {'crop': 'onion',  'lat': 21.9608, 'lon': 70.7980, 'zone_market': 'Gondal APMC',       'zone_state': 'Gujarat'},
    # Potato
    'P1_Agra':           {'crop': 'potato', 'lat': 27.1767, 'lon': 78.0081, 'zone_market': 'Agra APMC',         'zone_state': 'Uttar Pradesh'},
    'P2_Farrukhabad':    {'crop': 'potato', 'lat': 27.3900, 'lon': 79.5800, 'zone_market': 'Farrukhabad APMC',  'zone_state': 'Uttar Pradesh'},
    'P3_Jalandhar':      {'crop': 'potato', 'lat': 31.3260, 'lon': 75.5762, 'zone_market': 'Jalandhar APMC',    'zone_state': 'Punjab'},
    'P4_Bardhaman':      {'crop': 'potato', 'lat': 23.2330, 'lon': 87.8550, 'zone_market': 'Bardhaman APMC',    'zone_state': 'West Bengal'},
}

ZONE_DF = pd.DataFrame(ZONES).T.reset_index().rename(columns={'index': 'zone_id'})
ZONE_DF['lat'] = ZONE_DF['lat'].astype(float)
ZONE_DF['lon'] = ZONE_DF['lon'].astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# State centroid fallbacks (approximate, for districts that fail to geocode)
# ─────────────────────────────────────────────────────────────────────────────
STATE_CENTROIDS = {
    'Andhra Pradesh':      (15.9129, 79.7400),
    'Arunachal Pradesh':   (28.2180, 94.7278),
    'Assam':               (26.2006, 92.9376),
    'Bihar':               (25.0961, 85.3131),
    'Chhattisgarh':        (21.2787, 81.8661),
    'Goa':                 (15.2993, 74.1240),
    'Gujarat':             (22.2587, 71.1924),
    'Haryana':             (29.0588, 76.0856),
    'Himachal Pradesh':    (31.1048, 77.1734),
    'Jharkhand':           (23.6102, 85.2799),
    'Karnataka':           (15.3173, 75.7139),
    'Kerala':              (10.8505, 76.2711),
    'Madhya Pradesh':      (22.9734, 78.6569),
    'Maharashtra':         (19.7515, 75.7139),
    'Manipur':             (24.6637, 93.9063),
    'Meghalaya':           (25.4670, 91.3662),
    'Mizoram':             (23.1645, 92.9376),
    'Nagaland':            (26.1584, 94.5624),
    'Odisha':              (20.9517, 85.0985),
    'Punjab':              (31.1471, 75.3412),
    'Rajasthan':           (27.0238, 74.2179),
    'Sikkim':              (27.5330, 88.5122),
    'Tamil Nadu':          (11.1271, 78.6569),
    'Telangana':           (18.1124, 79.0193),
    'Tripura':             (23.9408, 91.9882),
    'Uttar Pradesh':       (26.8467, 80.9462),
    'Uttarakhand':         (30.0668, 79.0193),
    'West Bengal':         (22.9868, 87.8550),
    'Delhi':               (28.7041, 77.1025),
    'Jammu and Kashmir':   (33.7782, 76.5762),
    'Jammu & Kashmir':     (33.7782, 76.5762),
    'Ladakh':              (34.1526, 77.5770),
    'Chandigarh':          (30.7333, 76.7794),
    'Puducherry':          (11.9416, 79.8083),
    'Andaman and Nicobar Islands': (11.7401, 92.6586),
}

# ─────────────────────────────────────────────────────────────────────────────
# Geocoding via Nominatim (no external dependencies — uses urllib only)
# ─────────────────────────────────────────────────────────────────────────────
NOMINATIM = 'https://nominatim.openstreetmap.org/search'
USER_AGENT = 'TOP_DigitalTwin_SKUAST_Research'


def _nominatim_query(query: str) -> tuple:
    """Return (lat, lon) or None. Caller must enforce rate limiting."""
    params = urllib.parse.urlencode({
        'q': query,
        'format': 'json',
        'limit': 1,
        'countrycodes': 'in',
    })
    url = f'{NOMINATIM}?{params}'
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        pass
    return None


def geocode_district(district: str, state: str, cache: dict) -> tuple:
    """
    Try progressively broader queries until one resolves.
    Returns (lat, lon, level) where level indicates resolution quality.
    """
    # Normalise state name for cache key
    state_clean = state.strip()
    dist_clean  = district.strip()

    queries = [
        (f'{dist_clean}, {state_clean}, India', 'district'),
        (f'{dist_clean} district, India',       'district_only'),
        (f'{dist_clean}, India',                'name_only'),
    ]

    for query, level in queries:
        if query in cache:
            result = cache[query]
            if result is not None:
                return result[0], result[1], level
            continue  # cached miss — skip
        time.sleep(1.1)  # Nominatim policy: max 1 req/s
        result = _nominatim_query(query)
        cache[query] = result  # cache hit or miss
        if result is not None:
            return result[0], result[1], level

    # State-level fallback (no API call needed)
    if state_clean in STATE_CENTROIDS:
        lat, lon = STATE_CENTROIDS[state_clean]
        return lat, lon, 'state'

    return None, None, 'failed'


# ─────────────────────────────────────────────────────────────────────────────
# Haversine distance (km)
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def assign_zone(lat: float, lon: float, crop: str) -> tuple:
    """Return (zone_id, zone_market, zone_state, dist_km) for nearest same-crop zone."""
    candidates = ZONE_DF[ZONE_DF['crop'] == crop]
    best_zone, best_dist = None, float('inf')
    for _, row in candidates.iterrows():
        d = haversine(lat, lon, row['lat'], row['lon'])
        if d < best_dist:
            best_dist = d
            best_zone = row
    if best_zone is None:
        return None, None, None, None
    return best_zone['zone_id'], best_zone['zone_market'], best_zone['zone_state'], round(best_dist, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load market panels and extract unique (crop, state, district) combos
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('SCRIPT 16 — Production Zone Assignment')
print('=' * 65)

print('\nSTEP 1: Loading market panels')
print('-' * 45)

all_markets = []
for crop in ['tomato', 'onion', 'potato']:
    fpath = PANEL_DIR / f'{crop}_weekly_panel.csv'
    if not fpath.exists():
        print(f'  WARNING: {fpath.name} not found — skipping {crop}')
        continue
    df = pd.read_csv(fpath, usecols=['market_id', 'market', 'state', 'district', 'crop'],
                     low_memory=False)
    df = df.drop_duplicates('market_id')
    all_markets.append(df)
    print(f'  {crop:<8}: {len(df):>4} unique markets, {df["state"].nunique()} states')

markets = pd.concat(all_markets, ignore_index=True)
markets['crop'] = markets['crop'].str.lower().str.strip()
print(f'\n  Total unique markets: {len(markets):,}')

# Unique (district, state) pairs to geocode
unique_locs = (markets[['district', 'state']]
               .drop_duplicates()
               .sort_values(['state', 'district'])
               .reset_index(drop=True))
print(f'  Unique (district, state) pairs: {len(unique_locs)}')

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Geocode each unique district
# ─────────────────────────────────────────────────────────────────────────────
print('\nSTEP 2: Geocoding districts via Nominatim (cached)')
print('-' * 45)

# Load existing cache
cache = {}
if CACHE_FILE.exists():
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    print(f'  Loaded {len(cache)} cached entries from {CACHE_FILE.name}')
else:
    print(f'  No cache found — will geocode all {len(unique_locs)} pairs (takes ~{len(unique_locs)*1.2/60:.0f} min)')

geo_results = {}
n_cached, n_api, n_fallback, n_failed = 0, 0, 0, 0

for i, (_, row) in enumerate(unique_locs.iterrows(), 1):
    district, state = row['district'], row['state']
    key = f'{district}||{state}'

    # Check if this (district, state) pair already resolved
    query_d = f'{district.strip()}, {state.strip()}, India'
    if query_d in cache and cache[query_d] is not None:
        r = cache[query_d]
        geo_results[key] = (r[0], r[1], 'district')
        n_cached += 1
    else:
        lat, lon, level = geocode_district(district, state, cache)
        geo_results[key] = (lat, lon, level)
        if level in ('district', 'district_only', 'name_only'):
            n_api += 1
        elif level == 'state':
            n_fallback += 1
        else:
            n_failed += 1

    if i % 25 == 0 or i == len(unique_locs):
        print(f'  [{i:>4}/{len(unique_locs)}]  cached={n_cached}  '
              f'api={n_api}  state_fallback={n_fallback}  failed={n_failed}')

    # Save cache periodically
    if i % 50 == 0:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

# Final cache save
with open(CACHE_FILE, 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)
print(f'\n  Cache saved: {CACHE_FILE.name}')
print(f'  Resolution summary: district={n_cached + n_api}  state_fallback={n_fallback}  failed={n_failed}')

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Assign each market to nearest same-crop zone
# ─────────────────────────────────────────────────────────────────────────────
print('\nSTEP 3: Assigning markets to nearest production zone')
print('-' * 45)

rows = []
for _, mkt in markets.iterrows():
    key = f'{mkt["district"]}||{mkt["state"]}'
    lat, lon, geocode_level = geo_results.get(key, (None, None, 'failed'))

    if lat is None:
        zone_id = zone_market = zone_state = None
        dist_km = None
    else:
        zone_id, zone_market, zone_state, dist_km = assign_zone(lat, lon, mkt['crop'])

    rows.append({
        'market_id':     mkt['market_id'],
        'market':        mkt['market'],
        'crop':          mkt['crop'],
        'state':         mkt['state'],
        'district':      mkt['district'],
        'lat':           lat,
        'lon':           lon,
        'geocode_level': geocode_level,
        'zone_id':       zone_id,
        'zone_market':   zone_market,
        'zone_state':    zone_state,
        'dist_km':       dist_km,
    })

assignment = pd.DataFrame(rows)
assignment.to_csv(OUT_FILE, index=False)
print(f'  Saved: {OUT_FILE}')
print(f'  Total markets assigned: {assignment["zone_id"].notna().sum()} / {len(assignment)}')

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Summary tables
# ─────────────────────────────────────────────────────────────────────────────
print('\nSTEP 4: Assignment summary')
print('=' * 65)

for crop in ['tomato', 'onion', 'potato']:
    sub = assignment[assignment['crop'] == crop]
    print(f'\n  {crop.upper()} — {len(sub)} markets across {sub["state"].nunique()} states')
    print(f'  {"Zone":<22} {"Markets":>8}  {"States covered":<45}  {"Mean dist km":>12}')
    print(f'  {"-"*95}')
    for zone_id, grp in sub.groupby('zone_id'):
        states = ', '.join(sorted(grp['state'].unique()))
        mean_d = grp['dist_km'].mean()
        print(f'  {zone_id:<22} {len(grp):>8}  {states[:45]:<45}  {mean_d:>12.0f}')

print(f'\n  Geocode level breakdown:')
print(assignment['geocode_level'].value_counts().to_string())

print(f'\n  Distance to assigned zone (km):')
print(assignment['dist_km'].describe().round(1).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Diagnostic figures
# ─────────────────────────────────────────────────────────────────────────────
print('\nSTEP 5: Generating diagnostic figures')
print('-' * 45)

ZONE_COLORS = {
    # Tomato
    'T1_Kolar':         '#d62728',
    'T2_Madanapalle':   '#ff7f0e',
    'T3_Nashik_Tomato': '#e377c2',
    'T4_Solan':         '#bcbd22',
    'T5_Navsari':       '#f7b6d2',
    # Onion
    'O1_Lasalgaon':     '#1f77b4',
    'O2_Pimpalgaon':    '#aec7e8',
    'O3_Mahuva':        '#9467bd',
    'O6_Hubli':         '#c5b0d5',
    'O7_Solapur':       '#17becf',
    'O8_Manmad':        '#7fdbff',
    'O9_Kurnool':       '#0074d9',
    'O10_Gondal':       '#b5c0d0',
    # Potato
    'P1_Agra':          '#2ca02c',
    'P2_Farrukhabad':   '#98df8a',
    'P3_Jalandhar':     '#8c564b',
    'P4_Bardhaman':     '#c49c94',
}

assigned = assignment.dropna(subset=['lat', 'lon', 'zone_id'])

# ── Figure 1: Scatter map of markets coloured by assigned zone ────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
crop_list = ['tomato', 'onion', 'potato']
crop_titles = {'tomato': 'Tomato (T1–T5)', 'onion': 'Onion (O1–O10)', 'potato': 'Potato (P1–P4)'}

for ax, crop in zip(axes, crop_list):
    sub = assigned[assigned['crop'] == crop]
    zones_crop = ZONE_DF[ZONE_DF['crop'] == crop]

    # Market scatter
    for zone_id, grp in sub.groupby('zone_id'):
        color = ZONE_COLORS.get(zone_id, 'grey')
        ax.scatter(grp['lon'], grp['lat'], c=color, s=18, alpha=0.6,
                   zorder=2, label=zone_id)

    # Zone centroids as large stars
    for _, zrow in zones_crop.iterrows():
        color = ZONE_COLORS.get(zrow['zone_id'], 'grey')
        ax.scatter(zrow['lon'], zrow['lat'], c=color, s=250, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.annotate(zrow['zone_id'], (zrow['lon'], zrow['lat']),
                    textcoords='offset points', xytext=(5, 4), fontsize=6.5,
                    fontweight='bold', color='black')

    ax.set_title(crop_titles[crop], fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(67, 98)
    ax.set_ylim(7, 37)

    legend_patches = [mpatches.Patch(color=ZONE_COLORS.get(z, 'grey'), label=z)
                      for z in zones_crop['zone_id']]
    ax.legend(handles=legend_patches, fontsize=6.5, loc='lower right',
              framealpha=0.8, ncol=2)

plt.suptitle('APMC Market → Nearest Production Zone Assignment\n'
             '(stars = zone centroids; dots = APMC markets)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig1_path = FIG_DIR / 'fig_zone_assignment_map.png'
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig1_path.name}')

# ── Figure 2: Markets per zone + mean distance bar chart ─────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

zone_counts = assignment['zone_id'].value_counts().reindex(list(ZONE_COLORS.keys()), fill_value=0)
zone_dist   = assignment.groupby('zone_id')['dist_km'].mean().reindex(list(ZONE_COLORS.keys()))

colors = [ZONE_COLORS.get(z, 'grey') for z in zone_counts.index]

ax1.barh(zone_counts.index, zone_counts.values, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_xlabel('Number of APMC markets', fontsize=10)
ax1.set_title('Markets assigned to each production zone', fontsize=11, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(zone_counts.values):
    ax1.text(v + 0.3, i, str(v), va='center', fontsize=8)
ax1.grid(axis='x', alpha=0.3)

ax2.barh(zone_dist.index, zone_dist.values, color=colors, edgecolor='white', linewidth=0.5)
ax2.set_xlabel('Mean distance to zone centroid (km)', fontsize=10)
ax2.set_title('Mean distance — market to assigned zone', fontsize=11, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(zone_dist.values):
    if not np.isnan(v):
        ax2.text(v + 2, i, f'{v:.0f}', va='center', fontsize=8)
ax2.axvline(300, color='red', lw=1, linestyle='--', alpha=0.6, label='300 km')
ax2.legend(fontsize=9)
ax2.grid(axis='x', alpha=0.3)

plt.suptitle('Zone Assignment: Market Count and Distance Summary',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig2_path = FIG_DIR / 'fig_zone_assignment_summary.png'
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig2_path.name}')

# ── Figure 3: Distance distribution histogram per crop ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
crop_colors = {'tomato': '#d62728', 'onion': '#1f77b4', 'potato': '#2ca02c'}

for ax, crop in zip(axes, ['tomato', 'onion', 'potato']):
    sub = assignment[(assignment['crop'] == crop) & assignment['dist_km'].notna()]
    ax.hist(sub['dist_km'], bins=30, color=crop_colors[crop], alpha=0.75, edgecolor='white')
    ax.axvline(sub['dist_km'].median(), color='black', lw=1.5, linestyle='--',
               label=f'Median {sub["dist_km"].median():.0f} km')
    ax.set_title(crop.capitalize(), fontsize=11, fontweight='bold')
    ax.set_xlabel('Distance to assigned zone (km)', fontsize=9)
    ax.set_ylabel('Number of markets', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Distribution of Market-to-Zone Distances by Crop',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig3_path = FIG_DIR / 'fig_zone_assignment_distances.png'
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig3_path.name}')

# ─────────────────────────────────────────────────────────────────────────────
# Final
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('Script 16 complete.')
print('=' * 65)
print(f'\n  Output : {OUT_FILE}')
print(f'  Markets : {len(assignment):,}  ({assignment["zone_id"].notna().sum()} assigned)')
print(f'\n  Next step: re-run Script 14 to generate market_zone_features.csv')
print(f'  Join in modelling scripts:')
print(f'    zone_assign = pd.read_csv("data/zone_assignment.csv")')
print(f'    panel = panel.merge(zone_assign[["market_id","zone_id"]], on="market_id")')
print(f'    panel = panel.merge(zone_feat, on=["zone_id","week_start"])')
