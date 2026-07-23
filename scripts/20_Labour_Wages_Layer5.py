# -*- coding: utf-8 -*-
"""
Script 20 — Layer 5: Rural Agricultural Wages (state-wise)
==============================================================
Parses the Labour Bureau "Wage Rates in Rural India" state-wise series
(Agricultural occupations, Men and Women kept as SEPARATE features per
user decision — not averaged, to preserve gender-wage-gap signal) into a
panel-joinable monthly file.

Source file: Downloads/Scheme II-00632659-M.xlsx
  - CEIC-style wide export: metadata rows 0-12, data from row 13 (date-indexed)
  - Columns include All-India + ~20 individual states, several with two
    columns each because the state's boundaries changed mid-series (e.g.
    Andhra Pradesh splits at Jun-2014 when Telangana was carved out; Bihar,
    Madhya Pradesh, Uttar Pradesh split at Nov-2000 for Jharkhand,
    Chhattisgarh, Uttarakhand respectively) — these are coalesced into one
    continuous series per state.
  - "Children" wage category intentionally excluded (ethical + data-quality
    grounds: thin, noisy series, not a meaningful farm labor-cost driver).
  - "All occupations" category excluded in favor of "Agricultural
    occupations" specifically, the economically relevant series for
    farm-gate price transmission.

Coverage gap: the source has no standalone series for Chhattisgarh,
Jharkhand, Uttarakhand, Telangana, Goa, NCT of Delhi, Chandigarh, Andaman &
Nicobar, Arunachal Pradesh, Mizoram, or Nagaland. Where the missing state
was carved out of a parent state within the source's history, this script
falls back to the parent state's wage as a regional proxy (documented in
the `wage_source` column); otherwise it falls back to the All-India series.
This is a real approximation — check `wage_source` before treating those
states' wage features as precise.

Output:
  data/labour_wages/wage_agri_state_monthly.csv
    columns: state, state_code, year, month, wage_agri_men, wage_agri_women,
             wage_source

Run: python scripts/20_Labour_Wages_Layer5.py
"""

import os
import pandas as pd
import numpy as np

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_FILE   = r'C:\Users\masro\Downloads\Scheme II-00632659-M.xlsx'
PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR    = os.path.join(BASE, 'data', 'labour_wages')
os.makedirs(OUT_DIR, exist_ok=True)

PANEL_START = pd.Timestamp('2017-01-01')
PANEL_END   = pd.Timestamp('2025-12-31')

# Column index ranges in the source file (0-indexed, from the header/description
# row) — confirmed by inspection: Men = cols 1-25, Women = cols 73-94.
MEN_COL_RANGE   = (1, 25)
WOMEN_COL_RANGE = (73, 94)

# Labour Bureau state label -> panel state name (panel uses "Jammu and
# Kashmir" with "and", source uses "Jammu & Kashmir" with "&")
LB_TO_PANEL_STATE = {
    'India': 'India',
    'Andhra Pradesh': 'Andhra Pradesh',
    'Assam': 'Assam',
    'Bihar': 'Bihar',
    'Gujarat': 'Gujarat',
    'Haryana': 'Haryana',
    'Himachal Pradesh': 'Himachal Pradesh',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Karnataka': 'Karnataka',
    'Kerala': 'Kerala',
    'Madhya Pradesh': 'Madhya Pradesh',
    'Maharashtra': 'Maharashtra',
    'Manipur': 'Manipur',
    'Meghalaya': 'Meghalaya',
    'Odisha': 'Odisha',
    'Punjab': 'Punjab',
    'Rajasthan': 'Rajasthan',
    'Tamil Nadu': 'Tamil Nadu',
    'Tripura': 'Tripura',
    'Uttar Pradesh': 'Uttar Pradesh',
    'West Bengal': 'West Bengal',
}

# Panel states with no direct Labour Bureau series -> proxy state to borrow
# from, and why. 'India' proxy = All-India fallback (no regional proxy
# available).
PROXY_MAP = {
    'Telangana':           ('Andhra Pradesh', 'parent-state proxy (split Jun-2014)'),
    'Chattisgarh':         ('Madhya Pradesh', 'parent-state proxy (split Nov-2000)'),
    'Jharkhand':           ('Bihar',          'parent-state proxy (split Nov-2000)'),
    'Uttarakhand':         ('Uttar Pradesh',  'parent-state proxy (split Nov-2000)'),
    'Goa':                 ('India',          'no regional proxy - All-India fallback'),
    'NCT of Delhi':        ('India',          'no regional proxy - All-India fallback'),
    'Chandigarh':          ('India',          'no regional proxy - All-India fallback'),
    'Andaman and Nicobar': ('India',          'no regional proxy - All-India fallback'),
    'Arunachal Pradesh':   ('India',          'no regional proxy - All-India fallback'),
    'Mizoram':             ('India',          'no regional proxy - All-India fallback'),
    'Nagaland':            ('India',          'no regional proxy - All-India fallback'),
    'Keralam':             ('Kerala',         'alternate spelling in panel'),
}


def parse_state_columns(region_row, start_idx, end_idx):
    """Group column indices by state name, stripping the trailing
    ', <date range>' suffix (split-series columns share a state name)."""
    groups = {}
    for i in range(start_idx, end_idx + 1):
        label = str(region_row[i])
        state = 'India' if label == 'India' else label.split(',')[0].strip()
        groups.setdefault(state, []).append(i)
    return groups


def coalesce_split_series(data_block, cols):
    """Split-series states have >1 column (different date ranges); take the
    first non-null value across them per row, since ranges don't overlap."""
    return data_block[cols].bfill(axis=1).iloc[:, 0]


print('=' * 65)
print('SCRIPT 20: LAYER 5 — RURAL AGRICULTURAL WAGES (STATE-WISE)')
print('=' * 65)

print('[1] Parsing source file ...')
raw = pd.read_excel(SRC_FILE, sheet_name='Sheet1', header=None)
region_row = raw.iloc[12]
data_block = raw.iloc[13:].reset_index(drop=True)
data_block.columns = range(raw.shape[1])
dates = pd.to_datetime(data_block[0], format='%d-%b-%Y', errors='coerce')

men_groups   = parse_state_columns(region_row, *MEN_COL_RANGE)
women_groups = parse_state_columns(region_row, *WOMEN_COL_RANGE)
print(f'  Men states found:   {len(men_groups)}')
print(f'  Women states found: {len(women_groups)}')

men_series   = {s: coalesce_split_series(data_block, c) for s, c in men_groups.items()}
women_series = {s: coalesce_split_series(data_block, c) for s, c in women_groups.items()}

men_df   = pd.DataFrame(men_series);   men_df['date']   = dates.values
women_df = pd.DataFrame(women_series); women_df['date'] = dates.values

men_long = men_df.melt(id_vars='date', var_name='lb_state', value_name='wage_agri_men')
women_long = women_df.melt(id_vars='date', var_name='lb_state', value_name='wage_agri_women')
wages = men_long.merge(women_long, on=['date', 'lb_state'], how='outer')
wages['panel_state'] = wages['lb_state'].map(LB_TO_PANEL_STATE)
wages = wages.dropna(subset=['panel_state'])

wages = wages[(wages['date'] >= PANEL_START) & (wages['date'] <= PANEL_END)]
print(f'  Rows in project window (2017-2025): {len(wages):,}')

india_lookup = (wages[wages['panel_state'] == 'India']
                .set_index('date')[['wage_agri_men', 'wage_agri_women']])


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD FULL PANEL-STATE TABLE, APPLYING PROXIES WHERE NEEDED
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Loading panel to get the full state list, applying proxies for gaps ...')
panel_states = pd.read_csv(PANEL_FILE, usecols=['state', 'state_code']).drop_duplicates()

rows = []
for _, prow in panel_states.iterrows():
    pstate, pcode = prow['state'], prow['state_code']
    direct = wages[wages['panel_state'] == pstate]

    if not direct.empty:
        sub = direct[['date', 'wage_agri_men', 'wage_agri_women']].copy()
        sub['state'] = pstate
        sub['state_code'] = pcode
        sub['wage_source'] = 'direct'
        rows.append(sub)
    elif pstate in PROXY_MAP:
        proxy_state, reason = PROXY_MAP[pstate]
        if proxy_state == 'India':
            src = india_lookup.reset_index()
        else:
            src = wages[wages['panel_state'] == proxy_state][
                ['date', 'wage_agri_men', 'wage_agri_women']]
        sub = src.copy()
        sub['state'] = pstate
        sub['state_code'] = pcode
        sub['wage_source'] = f'{proxy_state} ({reason})'
        rows.append(sub)
    else:
        # Unmapped state with no proxy defined — fall back to India, flag clearly
        src = india_lookup.reset_index()
        sub = src.copy()
        sub['state'] = pstate
        sub['state_code'] = pcode
        sub['wage_source'] = 'India (UNMAPPED - add explicit proxy if this state has real market coverage)'
        rows.append(sub)

final = pd.concat(rows, ignore_index=True)
final['year']  = final['date'].dt.year
final['month'] = final['date'].dt.month
final = final[['state', 'state_code', 'year', 'month',
                'wage_agri_men', 'wage_agri_women', 'wage_source']]
final = final.sort_values(['state', 'year', 'month']).reset_index(drop=True)

out_path = os.path.join(OUT_DIR, 'wage_agri_state_monthly.csv')
final.to_csv(out_path, index=False, encoding='utf-8')
print(f'  Saved: {out_path}  ({len(final):,} rows)')

print('\n[3] Wage source summary (states using a proxy, not direct data):')
proxy_summary = (final[final['wage_source'] != 'direct']
                  .groupby(['state', 'wage_source']).size().reset_index(name='n_months'))
if not proxy_summary.empty:
    print(proxy_summary.to_string(index=False))
else:
    print('  (none — every panel state matched directly)')

n_direct = (final['wage_source'] == 'direct').sum()
n_total  = len(final)
print(f'\n  {n_direct:,}/{n_total:,} rows ({100*n_direct/n_total:.1f}%) are direct '
      f'state-level matches; the rest use a documented proxy.')

print('\n' + '=' * 65)
print('Script 20 complete.')
print('\nNext: join wage_agri_state_monthly.csv onto the main weekly panel')
print('on (state, year, month) — same join pattern as the CMIE/RBI/PPAC')
print('macro layer, but now state-varying instead of national-uniform.')
