# -*- coding: utf-8 -*-
"""
Script 59 — Panel-Prep: Fertilizer MRP (M8a) + CPI-AL/RL (M8b)
==============================================================
Reshapes the two raw acquisitions from Scripts 57 and 58 (both long-format,
one row per product/state per month) into the wide, panel-joinable files
`panel_layers.py`'s new `join_fertilizer()` / `join_cpi_alrl()` expect --
the same "acquire, then a separate prep step builds the joinable file"
split Script 54 established for the drought layers, kept here so Scripts
57/58 stay acquisition-only per their own docstrings.

M8a — FERTILIZER MRP (national, no state dimension):
  Source has ~19-53 product variants per bulletin (grows over the archive).
  Wiring all of them in would bloat the panel with near-collinear NPK-blend
  variants that vary by only a few points of nutrient ratio. Narrowed to
  the three fertilizers actually meaningful as an input-cost signal for
  ANY crop grower, not npk-blend specifics: Urea, DAP, MOP -- confirmed all
  three have full coverage, all 98/98 bulletins, Jan-2018 through Jul-2026,
  no gaps, no duplicate (year, month) pairs per product (checked directly,
  2026-09-25).
  Join key: (year, month) -- same as M2 macro (CMIE/RBI/PPAC).

M8b — CPI-AL/RL (state-wise, Base 2019=100):
  Source uses official state names; the panel uses Agmarknet's own messier
  spellings (same quirks M5a wages / M5b cold storage already crosswalk):
  'Chhattisgarh'->'Chattisgarh', 'Delhi'->'NCT of Delhi', and Kerala's data
  is duplicated onto the panel's 'Keralam' rows too (both spellings appear
  in the panel). 'Chandigarh' has no source row and is left NaN -- Labour
  Bureau's 34-state/UT list genuinely doesn't include it (a fully-urban UT
  with no agricultural-labour population to sample), not a lookup miss.
  Long AL/RL rows pivoted to wide (cpi_al, cpi_rl columns).
  Join key: (state, year, month) -- same as M5a wages.

Output:
  data/fertilizer_mrp/fert_mrp_panel_monthly.csv
    (year, month, fert_urea_mrp, fert_dap_mrp, fert_mop_mrp)
  data/cpi_al_rl/cpi_alrl_panel_state_monthly.csv
    (state, year, month, cpi_al, cpi_rl, source_note)

Run: python scripts/59_Fertilizer_CPI_Panel_Prep.py
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FERT_SRC  = os.path.join(BASE, 'data', 'fertilizer_mrp', 'fert_mrp_monthly.csv')
CPI_SRC   = os.path.join(BASE, 'data', 'cpi_al_rl', 'cpi_alrl_statewise_monthly.csv')
PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')

FERT_OUT = os.path.join(BASE, 'data', 'fertilizer_mrp', 'fert_mrp_panel_monthly.csv')
CPI_OUT  = os.path.join(BASE, 'data', 'cpi_al_rl', 'cpi_alrl_panel_state_monthly.csv')

FERT_PRODUCTS = ['Urea', 'DAP', 'MOP']

# Keyed by CPI-AL/RL's own (source-side) state name, valued by the panel's
# spelling -- same direction Script 21's COLD_STATE_MAP uses.
CPI_STATE_MAP = {
    'Chhattisgarh': 'Chattisgarh',
    'Delhi': 'NCT of Delhi',
}


print('=' * 65)
print('SCRIPT 59: PANEL-PREP FOR FERTILIZER MRP (M8a) + CPI-AL/RL (M8b)')
print('=' * 65)

# ─────────────────────────────────────────────────────────────────────────────
# M8a — Fertilizer MRP: long (year, month, product) -> wide (year, month, ...)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Fertilizer MRP: pivoting Urea/DAP/MOP to wide ...')
fert = pd.read_csv(FERT_SRC)
fert = fert[fert['product'].isin(FERT_PRODUCTS)]

dupes = fert[fert.duplicated(subset=['product', 'year', 'month'], keep=False)]
if len(dupes):
    raise ValueError(f'{len(dupes)} duplicate (product, year, month) rows in {FERT_SRC} -- '
                      f'fix Script 57\'s dedup before pivoting.')

fert_wide = fert.pivot(index=['year', 'month'], columns='product', values='mrp_rs_per_mt').reset_index()
fert_wide = fert_wide.rename(columns={'Urea': 'fert_urea_mrp', 'DAP': 'fert_dap_mrp', 'MOP': 'fert_mop_mrp'})
fert_wide = fert_wide[['year', 'month', 'fert_urea_mrp', 'fert_dap_mrp', 'fert_mop_mrp']]
fert_wide = fert_wide.sort_values(['year', 'month'])

fert_wide.to_csv(FERT_OUT, index=False, encoding='utf-8')
first, last = fert_wide.iloc[0], fert_wide.iloc[-1]
print(f'  Saved: {FERT_OUT}  ({len(fert_wide)} months, '
      f'{int(first["year"])}-{int(first["month"]):02d} -> '
      f'{int(last["year"])}-{int(last["month"]):02d})')
missing = fert_wide[['fert_urea_mrp', 'fert_dap_mrp', 'fert_mop_mrp']].isna().sum()
print(f'  Missing values per product: {missing.to_dict()}')

# ─────────────────────────────────────────────────────────────────────────────
# M8b — CPI-AL/RL: long (state, series, year, month) -> wide, crosswalked
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] CPI-AL/RL: pivoting AL/RL to wide, crosswalking to panel state names ...')
cpi = pd.read_csv(CPI_SRC)
cpi = cpi[cpi['state'] != 'All India']  # national total, not a state -- not needed for a (state,...) join

cpi_wide = cpi.pivot(index=['state', 'year', 'month'], columns='series', values='index_value').reset_index()
cpi_wide = cpi_wide.rename(columns={'AL': 'cpi_al', 'RL': 'cpi_rl'})

panel_states = pd.read_csv(PANEL_FILE, usecols=['state']).drop_duplicates()['state'].tolist()
print(f'  Panel has {len(panel_states)} distinct state values to resolve')

rows = []
for pstate in panel_states:
    if pstate == 'Keralam':
        lookup_state, note = 'Kerala', 'Kerala (alternate spelling in panel)'
    else:
        lookup_state = next((src for src, tgt in CPI_STATE_MAP.items() if tgt == pstate), pstate)
        note = 'direct' if lookup_state == pstate else f'{lookup_state} (panel spelling differs)'
    match = cpi_wide[cpi_wide['state'] == lookup_state]
    if match.empty:
        print(f'  NOTE: no CPI-AL/RL source rows for panel state "{pstate}" '
              f'(looked up as "{lookup_state}") -- left as NaN, no fallback applied.')
        continue
    sub = match[['year', 'month', 'cpi_al', 'cpi_rl']].copy()
    sub['state'] = pstate
    sub['source_note'] = note
    rows.append(sub)

cpi_out = pd.concat(rows, ignore_index=True)[['state', 'year', 'month', 'cpi_al', 'cpi_rl', 'source_note']]
cpi_out = cpi_out.sort_values(['state', 'year', 'month'])

dupes = cpi_out[cpi_out.duplicated(subset=['state', 'year', 'month'], keep=False)]
if len(dupes):
    raise ValueError(f'{len(dupes)} duplicate (state, year, month) rows after crosswalk -- '
                      f'a source state must be mapping to more than one panel state incorrectly.')

cpi_out.to_csv(CPI_OUT, index=False, encoding='utf-8')
print(f'\n  Saved: {CPI_OUT}  ({len(cpi_out):,} state-months, '
      f'{cpi_out["state"].nunique()}/{len(panel_states)} panel states covered)')

print('\n' + '=' * 65)
print('Script 59 complete. See panel_layers.py join_fertilizer()/join_cpi_alrl() '
      'and Script 22 for the actual panel join (M8a/M8b).')
