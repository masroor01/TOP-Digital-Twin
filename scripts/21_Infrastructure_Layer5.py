# -*- coding: utf-8 -*-
"""
Script 21 — Layer 5: Infrastructure (Cold Storage + Road Density)
=====================================================================
Builds two panel-joinable infrastructure features per state:

1. Cold storage (static, single snapshot)
   Source: Downloads/RS_Session_266_AU_517_A_i.csv — Rajya Sabha Unstarred
   Question 517 answer (Session 266), state-wise cold storage count and
   capacity (MT). Treated as static since cold storage capacity changes
   slowly relative to a weekly price panel.

   Known data quirk: the source has BOTH a combined "Andhra Pradesh and
   Telangana" row and a separate "Telangana" row, left over from the 2014
   state split (many NHB registrations were never re-tagged). Per user
   decision, Telangana's row is treated as a subset already included in
   the combined row, so:
       Andhra Pradesh (resolved) = combined − Telangana

2. Road density (annual time series, forward-filled)
   Source: Downloads/Scheme II-00527542-A.xlsx — CEIC-sourced Labour
   Bureau/MORTH road density series, "Road density per 100 sq km: All
   (excluding railway roads)", state-wise, annual, 2010-2020. All panel
   states have a direct match, including Telangana (its own series from
   Jun-2014) — no proxies needed, unlike the wage data in Script 20.
   The series ends FY2019-20 (last available year); years 2021-2025 are
   forward-filled from the 2020 value since road infrastructure changes
   slowly — a documented assumption, not measured data.

Outputs (data/infrastructure/):
  cold_storage_by_state.csv      state, state_code, n_facilities, capacity_mt, source_note
  road_density_state_annual.csv  state, state_code, year, road_density_per_100_sqkm, is_forward_filled

Run: python scripts/21_Infrastructure_Layer5.py
"""

import os
import pandas as pd
import numpy as np

BASE        = r'C:\Users\masro\Documents\TOP_Digital_Twin'
COLD_FILE   = r'C:\Users\masro\Downloads\RS_Session_266_AU_517_A_i.csv'
ROAD_FILE   = r'C:\Users\masro\Downloads\Scheme II-00527542-A.xlsx'
PANEL_FILE  = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR     = os.path.join(BASE, 'data', 'infrastructure')
os.makedirs(OUT_DIR, exist_ok=True)

PANEL_YEARS = list(range(2017, 2026))
LAST_MEASURED_YEAR = 2020   # CEIC road series ends FY2019-20 (reported as 2020-03-31)

print('=' * 65)
print('SCRIPT 21: LAYER 5 — INFRASTRUCTURE (COLD STORAGE + ROADS)')
print('=' * 65)

panel_states = pd.read_csv(PANEL_FILE, usecols=['state', 'state_code']).drop_duplicates()

# ─────────────────────────────────────────────────────────────────────────────
# 1. COLD STORAGE
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Cold storage (Rajya Sabha Q517, Session 266) ...')
cold = pd.read_csv(COLD_FILE)
cold.columns = ['sl_no', 'state', 'n_facilities', 'capacity_mt']
cold = cold[cold['state'] != 'Total'].copy()

# Resolve the AP + Telangana combined-row quirk: AP = combined - Telangana
combined = cold[cold['state'] == 'Andhra Pradesh and Telangana'].iloc[0]
telangana = cold[cold['state'] == 'Telangana'].iloc[0]
ap_resolved = {
    'state': 'Andhra Pradesh',
    'n_facilities': combined['n_facilities'] - telangana['n_facilities'],
    'capacity_mt': combined['capacity_mt'] - telangana['capacity_mt'],
}
cold = cold[cold['state'] != 'Andhra Pradesh and Telangana'].copy()
cold = pd.concat([cold, pd.DataFrame([ap_resolved])], ignore_index=True)
print(f"  Resolved Andhra Pradesh (combined - Telangana): "
      f"{ap_resolved['n_facilities']} facilities, {ap_resolved['capacity_mt']:,} MT")

COLD_STATE_MAP = {
    'Andaman and Nicobar Islands': 'Andaman and Nicobar',
    'Chhattisgarh': 'Chattisgarh',
    'Delhi': 'NCT of Delhi',
    'Uttrakhand': 'Uttarakhand',   # typo in source
}
cold['state'] = cold['state'].replace(COLD_STATE_MAP)

cold_rows = []
for _, prow in panel_states.iterrows():
    pstate, pcode = prow['state'], prow['state_code']
    lookup_state = 'Kerala' if pstate == 'Keralam' else pstate
    match = cold[cold['state'] == lookup_state]
    if not match.empty:
        cold_rows.append({
            'state': pstate, 'state_code': pcode,
            'n_facilities': int(match['n_facilities'].iloc[0]),
            'capacity_mt': int(match['capacity_mt'].iloc[0]),
            'source_note': 'direct' if pstate != 'Keralam' else 'Kerala (alternate spelling in panel)',
        })
    else:
        cold_rows.append({
            'state': pstate, 'state_code': pcode,
            'n_facilities': np.nan, 'capacity_mt': np.nan,
            'source_note': 'NOT FOUND in source — no fallback applied, verify manually',
        })

cold_out = pd.DataFrame(cold_rows)
cold_path = os.path.join(OUT_DIR, 'cold_storage_by_state.csv')
cold_out.to_csv(cold_path, index=False, encoding='utf-8')
print(f'  Saved: {cold_path}  ({len(cold_out)} states)')
n_missing = cold_out['n_facilities'].isna().sum()
if n_missing:
    print(f'  WARNING: {n_missing} panel states had no match:')
    print('   ', cold_out[cold_out['n_facilities'].isna()]['state'].tolist())
else:
    print('  All panel states matched directly (no fallback needed).')


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROAD DENSITY
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Road density (CEIC/MORTH annual series, 2010-2020) ...')
raw = pd.read_excel(ROAD_FILE, sheet_name='Sheet1', header=None)
region_row = raw.iloc[12]
data_block = raw.iloc[13:].reset_index(drop=True)
data_block.columns = range(raw.shape[1])
dates = pd.to_datetime(data_block[0], format='%d-%b-%Y', errors='coerce')

# "Road density per 100 sq km: All (excluding jry roads)" is the first
# category block, columns 1-42 (confirmed by inspection)
DENSITY_COL_RANGE = (1, 42)

groups = {}
for i in range(DENSITY_COL_RANGE[0], DENSITY_COL_RANGE[1] + 1):
    label = str(region_row[i])
    state = 'India' if label == 'India' else label.split(',')[0].strip()
    groups.setdefault(state, []).append(i)

density_series = {s: data_block[c].bfill(axis=1).iloc[:, 0] for s, c in groups.items()}
density_df = pd.DataFrame(density_series)
density_df['date'] = dates.values
density_df['year'] = density_df['date'].dt.year
density_long = density_df.melt(id_vars=['date', 'year'], var_name='lb_state',
                                value_name='road_density_per_100_sqkm')
density_long = density_long.dropna(subset=['road_density_per_100_sqkm'])

ROAD_STATE_MAP = {
    'Andaman & Nicobar Islands': 'Andaman and Nicobar',
    'Chhattisgarh': 'Chattisgarh',
    'Jammu & Kashmir': 'Jammu and Kashmir',
}
density_long['panel_state'] = density_long['lb_state'].replace(ROAD_STATE_MAP)

road_rows = []
for _, prow in panel_states.iterrows():
    pstate, pcode = prow['state'], prow['state_code']
    lookup_state = 'Kerala' if pstate == 'Keralam' else pstate
    match = density_long[density_long['panel_state'] == lookup_state].sort_values('year')

    if match.empty:
        print(f'  WARNING: no road density match for {pstate}')
        continue

    last_row = match[match['year'] == match['year'].max()].iloc[0]
    for year in PANEL_YEARS:
        exact = match[match['year'] == year]
        if not exact.empty:
            road_rows.append({
                'state': pstate, 'state_code': pcode, 'year': year,
                'road_density_per_100_sqkm': exact['road_density_per_100_sqkm'].iloc[0],
                'is_forward_filled': False,
            })
        else:
            road_rows.append({
                'state': pstate, 'state_code': pcode, 'year': year,
                'road_density_per_100_sqkm': last_row['road_density_per_100_sqkm'],
                'is_forward_filled': True,
            })

road_out = pd.DataFrame(road_rows)
road_path = os.path.join(OUT_DIR, 'road_density_state_annual.csv')
road_out.to_csv(road_path, index=False, encoding='utf-8')
print(f'  Saved: {road_path}  ({len(road_out):,} rows)')
n_ff = road_out['is_forward_filled'].sum()
print(f'  {n_ff}/{len(road_out)} rows are forward-filled from the last measured '
      f'year (series ends FY{LAST_MEASURED_YEAR-1}-{str(LAST_MEASURED_YEAR)[-2:]}).')

print('\n' + '=' * 65)
print('Script 21 complete.')
print('\nNext: join cold_storage_by_state.csv on (state) [static] and')
print('road_density_state_annual.csv on (state, year) onto the main panel.')
