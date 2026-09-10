# -*- coding: utf-8 -*-
"""
Script 22 — Master Panel Join (All Layers M0-M7)
====================================================
Joins every compiled data layer onto the base weekly market panel into a
single consolidated file. NOTE: this consolidated file is NOT what Script
15 (the ablation study) actually trains on -- Script 15 loads and joins
every layer itself, since it layers extra work on top of several joins
(rolling climate/satellite features, forward-fill, M7 missingness flags)
that this script deliberately doesn't do. This file exists for the
eventual full-capacity TFT run and for anyone exploring the full joined
panel directly.

Both scripts share their actual join mechanics -- file paths, join keys,
uniqueness assertions, row-count checks -- via `scripts/panel_layers.py`,
so the two can no longer drift apart on those (see that module's
docstring for why this was extracted and what's still deliberately
per-script).

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
  M7a       trigger1_panel_weekly.csv (VEDAS)      join key: (state, district, week_start)
  M7b       cdi_panel_weekly.csv (IDM)             join key: (state, district, week_start)

M7 columns are structurally sparse by design (Trigger-1: Kharif-season
weeks, 2022+, crosswalked districts only; IDM CDI: 2021-07-14 onward) --
see Script 51/52/53/54 for why. This script does not forward-fill or
impute them; it joins them as-is, NaN where a layer has no reading for
that district/week. Whoever consumes this file needs to decide how to
handle that sparsity for their own purpose -- Script 15 handles it via an
explicit missingness-flag column per drought feature, not a blind fillna.

Output:
  data/master_weekly_panel_all_layers.csv

Run: python scripts/22_Master_Panel_Join.py
"""

import os
import pandas as pd
import numpy as np
import panel_layers as pl

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR    = os.path.join(BASE, 'data')
OUT_FILE   = os.path.join(OUT_DIR, 'master_weekly_panel_all_layers.csv')

PANEL_START = '2017-01-01'
PANEL_END   = '2030-12-31'  # generous future ceiling, not a real cutoff — matches
                             # Script 23's pattern so new data isn't silently truncated

checked_merge = pl.checked_merge  # re-exported for the diagnostics block below


print('=' * 65)
print('SCRIPT 22: MASTER PANEL JOIN (ALL LAYERS M0-M7)')
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
df, macro_cols = pl.join_macro(df)

# ─────────────────────────────────────────────────────────────────────────────
# M3/M4 — Climate + Satellite, join key (crop, week_start)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Joining climate/satellite (M3/M4) on (crop, week_start) ...')
sat = pl.load_satellite_climate()
df = checked_merge(df, sat, on=['crop', 'week_start'], how='left', label='M3/M4 climate/satellite')

# ─────────────────────────────────────────────────────────────────────────────
# M5a — Rural wages, join key (state, year, month)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Joining rural wages (M5a) on (state, year, month) ...')
df, _ = pl.join_wages(df)

# ─────────────────────────────────────────────────────────────────────────────
# M5b — Cold storage, join key (state) — static
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Joining cold storage (M5b) on (state) [static] ...')
df, _ = pl.join_cold_storage(df)

# ─────────────────────────────────────────────────────────────────────────────
# M5c — Road density, join key (state, year)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6] Joining road density (M5c) on (state, year) ...')
df, _ = pl.join_road_density(df)

# ─────────────────────────────────────────────────────────────────────────────
# M6 — Policy/trade events, join key (crop, week_start)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[7] Joining policy/trade (M6) on (crop, week_start) ...')
df, _ = pl.join_policy(df)

# ─────────────────────────────────────────────────────────────────────────────
# M7 — Drought (VEDAS Trigger-1 + IDM CDI), join key (state, district, week_start)
# add_missing_flags=False: this file is for open-ended exploration, so keep
# the raw trigger1_yn/drought_category columns instead of Script 15's
# model-oriented missingness flags.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[7b] Joining drought (M7): VEDAS Trigger-1 + IDM CDI on (state, district, week_start) ...')
df, _ = pl.join_drought(df, add_missing_flags=False)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MISSING-VALUE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
print('\n[8] Missing-value check per joined layer ...')
LAYER_COLS = {
    'M2 macro':      macro_cols,
    'M3/M4 climate/sat': [c for c in sat.columns if c not in ('crop', 'week_start')],
    'M5a wages':     ['wage_agri_men', 'wage_agri_women'],
    'M5b cold storage': ['cold_storage_n_facilities', 'cold_storage_capacity_mt'],
    'M5c road density': ['road_density_per_100_sqkm'],
    'M6 policy':     ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct'],
    'M7 drought':    ['trigger1', 'cdi'],
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
print('\nM7 (drought) columns are structurally sparse (see docstring) -- not a bug.')
