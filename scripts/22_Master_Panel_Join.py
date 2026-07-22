# -*- coding: utf-8 -*-
"""
Script 22 — Master Panel Join (All Layers M0-M6)
====================================================
Joins every compiled data layer onto the base weekly market panel into a
single consolidated file — the input for extending the ablation study
(Script 15) from M0-M4 to M0-M6, and for the eventual full-capacity TFT run.

Layers joined, in order, each verified to preserve row count (a left join
whose right side isn't unique on the join key silently fans out rows —
every step checks for that and fails loudly rather than producing a
corrupted panel):

  Base      top_weekly_panel.csv                  (crop, state, market, week_start)
  M2 macro  CMIE + RBI + PPAC                      join key: (year, month)
  M3/M4     crop_weekly_features.csv (climate/sat) join key: (crop, week_start)
  M5a       wage_agri_state_monthly.csv            join key: (state, year, month)
  M5b       cold_storage_by_state.csv              join key: (state)          [static]
  M5c       road_density_state_annual.csv          join key: (state, year)
  M6        policy_weekly_features.csv             join key: (crop, week_start)

Output:
  data/master_weekly_panel_all_layers.csv

Run: python scripts/22_Master_Panel_Join.py
"""

import os
import pandas as pd
import numpy as np

BASE = r'C:\Users\masro\Documents\TOP_Digital_Twin'

PANEL_FILE   = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CMIE_FILE    = os.path.join(BASE, 'data', 'cmie_macro', 'cmie_macro_2017_2025.csv')
RBI_FILE     = os.path.join(BASE, 'data', 'rbi_dbie', 'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE    = os.path.join(BASE, 'data', 'ppac_macro', 'ppac_diesel_lpg_2017_2025.csv')
SAT_FILE     = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
WAGE_FILE    = os.path.join(BASE, 'data', 'labour_wages', 'wage_agri_state_monthly.csv')
COLD_FILE    = os.path.join(BASE, 'data', 'infrastructure', 'cold_storage_by_state.csv')
ROAD_FILE    = os.path.join(BASE, 'data', 'infrastructure', 'road_density_state_annual.csv')
POLICY_FILE  = os.path.join(BASE, 'data', 'policy_trade', 'policy_weekly_features.csv')

OUT_DIR  = os.path.join(BASE, 'data')
OUT_FILE = os.path.join(OUT_DIR, 'master_weekly_panel_all_layers.csv')

PANEL_START = '2017-01-01'
PANEL_END   = '2025-12-31'


def checked_merge(left, right, on, how, label):
    """Left-join that asserts row count is preserved, catching silent
    fan-out from a non-unique join key on the right side."""
    n_before = len(left)
    merged = left.merge(right, on=on, how=how)
    n_after = len(merged)
    status = 'OK' if n_after == n_before else 'ROW COUNT CHANGED'
    print(f'  [{label}] join on {on}: {n_before:,} -> {n_after:,} rows  [{status}]')
    if n_after != n_before:
        raise ValueError(
            f'{label}: row count changed from {n_before:,} to {n_after:,} after '
            f'joining on {on} — the right-hand table is not unique on that key. '
            f'Fix the source file or the join key before proceeding.')
    return merged


print('=' * 65)
print('SCRIPT 22: MASTER PANEL JOIN (ALL LAYERS M0-M6)')
print('=' * 65)

print('\n[1] Loading base panel ...')
df = pd.read_csv(PANEL_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= PANEL_START) & (df['week_start'] <= PANEL_END)].copy()
df['year']  = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month
print(f'  Base panel: {len(df):,} rows')

# ─────────────────────────────────────────────────────────────────────────────
# M2 — Macro (CMIE + RBI + PPAC), join key (year, month)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Joining macro (M2): CMIE + RBI + PPAC on (year, month) ...')
macro_dfs = []
for fpath in [CMIE_FILE, RBI_FILE, PPAC_FILE]:
    if os.path.exists(fpath):
        macro_dfs.append(pd.read_csv(fpath))
macro = macro_dfs[0]
for m in macro_dfs[1:]:
    macro = macro.merge(m, on=['year', 'month'], how='outer', suffixes=('', '_dup'))
    macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
macro = macro.drop(columns=[c for c in ['date', 'date_x', 'date_y'] if c in macro.columns])
assert macro[['year', 'month']].duplicated().sum() == 0, 'macro table has duplicate (year,month) rows'
df = checked_merge(df, macro, on=['year', 'month'], how='left', label='M2 macro')

# ─────────────────────────────────────────────────────────────────────────────
# M3/M4 — Climate + Satellite, join key (crop, week_start)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Joining climate/satellite (M3/M4) on (crop, week_start) ...')
sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'])
assert sat[['crop', 'week_start']].duplicated().sum() == 0, 'satellite table has duplicate (crop,week_start) rows'
df = checked_merge(df, sat, on=['crop', 'week_start'], how='left', label='M3/M4 climate/satellite')

# ─────────────────────────────────────────────────────────────────────────────
# M5a — Rural wages, join key (state, year, month)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Joining rural wages (M5a) on (state, year, month) ...')
wages = pd.read_csv(WAGE_FILE)[['state', 'year', 'month', 'wage_agri_men', 'wage_agri_women']]
assert wages[['state', 'year', 'month']].duplicated().sum() == 0, 'wage table has duplicate keys'
df = checked_merge(df, wages, on=['state', 'year', 'month'], how='left', label='M5a wages')

# ─────────────────────────────────────────────────────────────────────────────
# M5b — Cold storage, join key (state) — static
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Joining cold storage (M5b) on (state) [static] ...')
cold = pd.read_csv(COLD_FILE)[['state', 'n_facilities', 'capacity_mt']]
cold = cold.rename(columns={'n_facilities': 'cold_storage_n_facilities',
                             'capacity_mt': 'cold_storage_capacity_mt'})
assert cold['state'].duplicated().sum() == 0, 'cold storage table has duplicate state rows'
df = checked_merge(df, cold, on=['state'], how='left', label='M5b cold storage')

# ─────────────────────────────────────────────────────────────────────────────
# M5c — Road density, join key (state, year)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6] Joining road density (M5c) on (state, year) ...')
road = pd.read_csv(ROAD_FILE)[['state', 'year', 'road_density_per_100_sqkm']]
assert road[['state', 'year']].duplicated().sum() == 0, 'road density table has duplicate (state,year) rows'
df = checked_merge(df, road, on=['state', 'year'], how='left', label='M5c road density')

# ─────────────────────────────────────────────────────────────────────────────
# M6 — Policy/trade events, join key (crop, week_start)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[7] Joining policy/trade (M6) on (crop, week_start) ...')
policy = pd.read_csv(POLICY_FILE, parse_dates=['week_start'])
assert policy[['crop', 'week_start']].duplicated().sum() == 0, 'policy table has duplicate (crop,week_start) rows'
df = checked_merge(df, policy, on=['crop', 'week_start'], how='left', label='M6 policy')


# ─────────────────────────────────────────────────────────────────────────────
# 8. MISSING-VALUE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
print('\n[8] Missing-value check per joined layer ...')
LAYER_COLS = {
    'M2 macro':      [c for c in macro.columns if c not in ('year', 'month')],
    'M3/M4 climate/sat': [c for c in sat.columns if c not in ('crop', 'week_start')],
    'M5a wages':     ['wage_agri_men', 'wage_agri_women'],
    'M5b cold storage': ['cold_storage_n_facilities', 'cold_storage_capacity_mt'],
    'M5c road density': ['road_density_per_100_sqkm'],
    'M6 policy':     ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct'],
}
for label, cols in LAYER_COLS.items():
    present = [c for c in cols if c in df.columns]
    if not present:
        continue
    pct_missing = df[present].isna().any(axis=1).mean() * 100
    print(f'  {label:<22s}: {pct_missing:5.1f}% of rows have at least one missing value')


# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[9] Saving master panel ...')
df.to_csv(OUT_FILE, index=False, encoding='utf-8')
print(f'  Saved: {OUT_FILE}')
print(f'  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns')

print('\n' + '=' * 65)
print('Script 22 complete.')
print('\nNext: extend Script 15 (ablation study) to build M5 (+ wages/cold')
print('storage/roads) and M6 (+ export policy) variants from this file.')
