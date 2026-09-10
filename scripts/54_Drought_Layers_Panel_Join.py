# -*- coding: utf-8 -*-
"""
Script 54 — Join Drought Layers (VEDAS Trigger-1, IDM CDI) onto the Panel
==============================================================
Builds two standalone, panel-joinable weekly drought feature files from the
raw acquisitions in Scripts 51-53, then joins both onto a copy of the base
market panel to demonstrate and measure real coverage -- following the same
checked-merge convention as Script 22 (row count must be preserved by every
left join; a non-unique join key on the right side would otherwise silently
fan out rows).

THIS SCRIPT DOES NOT TOUCH data/master_weekly_panel_all_layers.csv OR
SCRIPT 22. That file feeds the production ablation study and model retrain
(Scripts 15/23) -- wiring these two layers into it as a new M7 layer is a
separate, deliberate step for later, not a side effect of validating the
join here. This script's own output is a new, standalone file.

LAYERS AND THEIR JOIN LOGIC:

  VEDAS Trigger-1 (Script 51 + Script 52 crosswalk)
    Source rows: (state, vedas_district, date) -- fortnightly (1st/16th of
    month), Kharif-season only (mid-Jun to mid-Oct), 2022-2026.
    - Crosswalked to panel (state, district) via Script 52's output;
      'needs_review'/'unmatched' districts are dropped, not guessed.
    - Each fortnight date mapped to the Monday of its ISO week (same
      to_week_start() convention Script 14 uses for satellite composites).
    - Forward-filled up to 1 week within each (state, district) series --
      covers the ~15-day gap between fortnightly readings without ever
      bridging the ~8-month off-season gap (Oct-to-June), since a 1-week
      limit can't span 34 weeks regardless of season logic.
    - Encoded numeric (trigger1 = 1 for "Yes", 0 for "No") alongside the
      original Yes/No/None for readability.

  IDM CDI (Script 53)
    Source rows: (state, district, date) -- weekly, every Wednesday,
    2021-07-14 to present, already at panel-district granularity via the
    nearest-grid-point join.
    - Each Wednesday mapped to the Monday of its ISO week -- one reading
      per panel week already, no forward-fill needed.
    - Both the continuous `cdi` value and the documented `drought_category`
      (Normal/Abnormal/Moderate/Severe/Extreme/Exceptional) are carried.

Both layers join onto the panel on (state, district, week_start) -- district
is already a panel column, so no market-to-zone assignment step (like
Script 16's) is needed here; the drought signal genuinely is a district-
level quantity being broadcast to every market in that district, which is
the correct granularity for it (unlike price, which is market-specific).

Output:
  data/drought_vedas/trigger1_panel_weekly.csv   (state, district, week_start, trigger1_yn, trigger1)
  data/drought_idm/cdi_panel_weekly.csv          (state, district, week_start, cdi, drought_category)
  data/agmarknet_weekly/top_weekly_panel_with_drought.csv
    (base panel + both layers joined; NaN where a layer has no data for
    that district/week -- expected and documented per layer above, not a
    join bug)

Run: python scripts/54_Drought_Layers_Panel_Join.py
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANEL_FILE      = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CROSSWALK_FILE  = os.path.join(BASE, 'data', 'drought_vedas', 'district_crosswalk.csv')
TRIGGER1_FILE   = os.path.join(BASE, 'data', 'drought_vedas', 'trigger1_district_weekly.csv')
IDM_FILE        = os.path.join(BASE, 'data', 'drought_idm', 'cdi_district_weekly.csv')

OUT_TRIGGER1 = os.path.join(BASE, 'data', 'drought_vedas', 'trigger1_panel_weekly.csv')
OUT_IDM      = os.path.join(BASE, 'data', 'drought_idm', 'cdi_panel_weekly.csv')
OUT_PANEL    = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel_with_drought.csv')

TRIGGER1_FFILL_LIMIT = 1  # weeks; covers the ~15-day fortnight gap, can't bridge the off-season gap


def to_week_start(series):
    """Map any date to the Monday of its ISO week -- same convention as
    Script 14's to_week_start(), kept identical on purpose so a weekly
    feature built here lines up with every other weekly layer in the panel."""
    d = pd.to_datetime(series)
    return (d - pd.to_timedelta(d.dt.dayofweek, unit='D')).dt.normalize()


def checked_merge(left, right, on, how, label):
    """Left-join that asserts row count is preserved -- same guard Script
    22 uses, catching a non-unique join key on the right side before it
    silently fans out rows."""
    n_before = len(left)
    merged = left.merge(right, on=on, how=how)
    n_after = len(merged)
    status = 'OK' if n_after == n_before else 'ROW COUNT CHANGED'
    print(f'  [{label}] join on {on}: {n_before:,} -> {n_after:,} rows  [{status}]')
    if n_after != n_before:
        raise AssertionError(f'{label}: join key {on} is not unique on the right side -- '
                              f'fix the right-hand table before proceeding')
    return merged


print('=' * 65)
print('SCRIPT 54: JOIN DROUGHT LAYERS ONTO THE MARKET PANEL')
print('=' * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. VEDAS TRIGGER-1 -> panel-joinable weekly layer
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Building the Trigger-1 weekly layer ...')
crosswalk = pd.read_csv(CROSSWALK_FILE)
usable_cw = crosswalk[crosswalk['match_type'].isin(['exact', 'alias', 'fuzzy'])].copy()
print(f'  Crosswalk: {len(usable_cw)}/{len(crosswalk)} panel districts have a usable VEDAS mapping')

t1 = pd.read_csv(TRIGGER1_FILE, usecols=['state', 'district_norm', 'date', 'trigger1', 'fetch_status'])
t1 = t1[t1['fetch_status'] == 'ok'].dropna(subset=['trigger1'])
t1 = t1.rename(columns={'district_norm': 'vedas_district'})

# vedas_district in the crosswalk output already matches district_norm's
# original spelling (Script 52 preserved it via VNORM_TO_ORIGINAL) -- plain
# merge, no renormalization needed.
t1 = t1.merge(usable_cw[['state', 'panel_district', 'vedas_district']],
              on=['state', 'vedas_district'], how='inner')
t1['week_start'] = to_week_start(t1['date'])
t1 = t1.rename(columns={'panel_district': 'district'})
t1['trigger1_yn'] = t1['trigger1']
t1['trigger1'] = (t1['trigger1_yn'] == 'Yes').astype(int)
t1 = t1[['state', 'district', 'week_start', 'trigger1_yn', 'trigger1']].drop_duplicates(
    subset=['state', 'district', 'week_start'])

# Full weekly grid per (state, district) so ffill has somewhere to fill INTO,
# spanning only the district's own observed date range (not the whole panel
# history) -- keeps the ffill from ever reaching into years VEDAS has no data for.
frames = []
for (state, district), grp in t1.groupby(['state', 'district']):
    grp = grp.set_index('week_start').sort_index()
    full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq='W-MON')
    grp = grp.reindex(full_idx)
    grp[['trigger1_yn', 'trigger1']] = grp[['trigger1_yn', 'trigger1']].ffill(limit=TRIGGER1_FFILL_LIMIT)
    grp['state'] = state
    grp['district'] = district
    grp.index.name = 'week_start'
    frames.append(grp.reset_index())
t1_weekly = pd.concat(frames, ignore_index=True).dropna(subset=['trigger1'])
t1_weekly['trigger1'] = t1_weekly['trigger1'].astype(int)
t1_weekly = t1_weekly[['state', 'district', 'week_start', 'trigger1_yn', 'trigger1']]

t1_weekly.to_csv(OUT_TRIGGER1, index=False, encoding='utf-8')
print(f'  Saved: {OUT_TRIGGER1}  ({len(t1_weekly):,} district-weeks, '
      f'{t1_weekly[["state","district"]].drop_duplicates().shape[0]} districts)')

# ─────────────────────────────────────────────────────────────────────────────
# 2. IDM CDI -> panel-joinable weekly layer
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Building the IDM CDI weekly layer ...')
idm = pd.read_csv(IDM_FILE, usecols=['state', 'district', 'week_start', 'cdi', 'drought_category'])
idm['week_start'] = to_week_start(idm['week_start'])
idm = idm.dropna(subset=['cdi']).drop_duplicates(subset=['state', 'district', 'week_start'])
idm = idm[['state', 'district', 'week_start', 'cdi', 'drought_category']]

idm.to_csv(OUT_IDM, index=False, encoding='utf-8')
print(f'  Saved: {OUT_IDM}  ({len(idm):,} district-weeks, '
      f'{idm[["state","district"]].drop_duplicates().shape[0]} districts)')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Join both onto the market panel
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Joining onto the market panel ...')
panel = pd.read_csv(PANEL_FILE, parse_dates=['week_start'])
print(f'  Base panel: {len(panel):,} rows')

panel = checked_merge(panel, t1_weekly, on=['state', 'district', 'week_start'],
                       how='left', label='Trigger-1')
panel = checked_merge(panel, idm, on=['state', 'district', 'week_start'],
                       how='left', label='IDM CDI')

panel.to_csv(OUT_PANEL, index=False, encoding='utf-8')
print(f'\nSaved: {OUT_PANEL}  ({len(panel):,} rows)')

# ─────────────────────────────────────────────────────────────────────────────
# 4. Coverage report
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('COVERAGE REPORT')
print('=' * 65)
print('\nOverall row coverage (all panel rows, all dates 2017-2025):')
print(f'  Trigger-1 : {100*panel["trigger1"].notna().mean():5.1f}% of rows have a value '
      f'(expected to be low -- only Kharif weeks, 2022+, and only crosswalked districts)')
print(f'  IDM CDI   : {100*panel["cdi"].notna().mean():5.1f}% of rows have a value '
      f'(expected to be high for 2021-07-14 onward, zero before that)')

kharif_2022plus = panel[(panel['week_start'] >= '2022-06-16') &
                         (panel['week_start'].dt.month.isin([6, 7, 8, 9, 10]))]
print(f'\nWithin Trigger-1\'s actual valid window (Kharif weeks, 2022+, {len(kharif_2022plus):,} rows):')
print(f'  Trigger-1 : {100*kharif_2022plus["trigger1"].notna().mean():5.1f}% of rows have a value')

idm_era = panel[panel['week_start'] >= '2021-07-14']
print(f'\nWithin IDM\'s actual valid window (2021-07-14 onward, {len(idm_era):,} rows):')
print(f'  IDM CDI   : {100*idm_era["cdi"].notna().mean():5.1f}% of rows have a value')

print('\nVolume-weighted coverage within each layer\'s own valid window, per crop '
      '(non-imputed rows, arrivals-weighted):')
for label, sub in [('Trigger-1 (Kharif 2022+)', kharif_2022plus), ('IDM CDI (2021-07-14+)', idm_era)]:
    real = sub[sub['imputed'] == 0]
    col = 'trigger1' if 'Trigger-1' in label else 'cdi'
    print(f'  {label}:')
    for crop, g in real.groupby('crop'):
        total = g['arrivals_tonnes_week'].sum()
        cov = g.loc[g[col].notna(), 'arrivals_tonnes_week'].sum()
        pct = 100 * cov / total if total else float('nan')
        print(f'    {crop:8s}: {pct:5.1f}%')

print('\nScript 54 complete. Not wired into master_weekly_panel_all_layers.csv --')
print('that\'s a deliberate separate step (would touch the production ablation/model')
print('pipeline) and needs its own go-ahead.')
