# -*- coding: utf-8 -*-
"""
Script 34 — Two-Phase Baseline: Baseline-Phase Panel Join (Stage 2)
=====================================================================
Builds the long-window (2003-2026) panel for the two-phase architecture's
Phase 1 (baseline) model. Mirrors Script 22's join structure exactly --
same checked_merge discipline, same left-join-preserves-row-count
guarantee -- but swaps in the long-history sources built in Stage 1 and
deliberately DROPS two layers that Stage 1's audit found don't (or
shouldn't) span the full window:

  Infrastructure (wages, road density, cold storage): EXCLUDED by design
  decision (2026-08-02), not just missing data. Wages/road density do
  have real 2010+ history, but infrastructure's contribution has been
  modest throughout this project and backfilling 2003-2009 from the
  2010 value would be a real assumption injected into the "long, robust"
  baseline for a layer that matters this little. It belongs in Phase 2
  (the residual model), where its actual 2017+ window is already fully
  covered -- not stretched further back here.

  Policy (export ban/MEP/duty/market intervention): left-joined as-is
  from the EXISTING policy_weekly_features.csv, which still only spans
  2017-2026 (Script 19's PANEL_START was never extended). This is
  deliberate, not an oversight: the verified event log
  (TOP_policy_trade_verified_2017_2026.xlsx) has real, findable pre-2017
  events (a Dec-2010 onion export ban, 2014 MEP/ECA events, etc. --
  surfaced during this session's policy-document audit) that were never
  added with primary-source verification. Zero-filling policy for
  2003-2016 would ASSERT "no ban" for years that include a real one --
  worse than leaving it genuinely missing. Left as NaN pending that
  verification work; LightGBM handles per-row missing features natively.

Layers joined:
  Base      top_weekly_panel_longhistory.csv (Script 32)  key: (crop, state, market, week_start), 2003-2026
  M2 macro  CMIE (2000-2026) + RBI/PPAC longhistory (1992/2002-2026)  key: (year, month)
  M3/M4     crop_weekly_features.csv (climate/sat, now 2000-2026)     key: (crop, week_start)
  M6 policy policy_weekly_features.csv (still 2017-2026 -- see above) key: (crop, week_start), left as NaN before 2017

Output:
  data/baseline_phase_panel.csv

Run: python scripts/34_Baseline_Phase_Panel_Join.py
"""

import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANEL_FILE  = os.path.join(BASE, 'data', 'agmarknet_weekly', 'longhistory', 'top_weekly_panel_longhistory.csv')
CMIE_FILE   = os.path.join(BASE, 'data', 'cmie_macro', 'cmie_macro_2017_2025.csv')            # extended to 2000-2026 in Stage 1
RBI_FILE    = os.path.join(BASE, 'data', 'rbi_dbie', 'rbi_dbie_macro_longhistory.csv')        # 1992/2010/2011-2026, Stage 1
PPAC_FILE   = os.path.join(BASE, 'data', 'ppac_macro', 'ppac_diesel_lpg_longhistory.csv')     # 2002-2026, Stage 1
SAT_FILE    = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')     # extended to 2000-2026 in Stage 1
POLICY_FILE = os.path.join(BASE, 'data', 'policy_trade', 'policy_weekly_features.csv')        # still 2017-2026, see docstring

OUT_DIR  = os.path.join(BASE, 'data')
OUT_FILE = os.path.join(OUT_DIR, 'baseline_phase_panel.csv')

PANEL_START = '2003-01-01'
PANEL_END   = '2026-07-27'


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
print('SCRIPT 34: TWO-PHASE BASELINE -- STAGE 2 PANEL JOIN')
print('=' * 65)

print('\n[1] Loading base panel (Script 32 longhistory, 2003-2026) ...')
df = pd.read_csv(PANEL_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= PANEL_START) & (df['week_start'] <= PANEL_END)].copy()
df['year']  = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month
print(f'  Base panel: {len(df):,} rows  ({df["week_start"].min().date()} to {df["week_start"].max().date()})')
print(f'  Markets by crop: {df.groupby("crop")["market"].nunique().to_dict()}')

# ─────────────────────────────────────────────────────────────────────────────
# M2 — Macro (CMIE + RBI + PPAC longhistory), join key (year, month)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Joining macro (M2): CMIE + RBI/PPAC longhistory on (year, month) ...')
macro_dfs = []
for fpath in [CMIE_FILE, RBI_FILE, PPAC_FILE]:
    if os.path.exists(fpath):
        macro_dfs.append(pd.read_csv(fpath))
    else:
        raise FileNotFoundError(f'Expected Stage 1 output missing: {fpath}')
macro = macro_dfs[0]
for m in macro_dfs[1:]:
    macro = macro.merge(m, on=['year', 'month'], how='outer', suffixes=('', '_dup'))
    macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
macro = macro.drop(columns=[c for c in ['date', 'date_x', 'date_y'] if c in macro.columns])
assert macro[['year', 'month']].duplicated().sum() == 0, 'macro table has duplicate (year,month) rows'
print(f'  Combined macro table: {len(macro)} months, '
      f'{int(macro.year.min())}-{int(macro.loc[macro.year==macro.year.min(),"month"].min()):02d} to '
      f'{int(macro.year.max())}-{int(macro.loc[macro.year==macro.year.max(),"month"].max()):02d}')
df = checked_merge(df, macro, on=['year', 'month'], how='left', label='M2 macro')

# ─────────────────────────────────────────────────────────────────────────────
# M3/M4 — Climate + Satellite, join key (crop, week_start)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Joining climate/satellite (M3/M4) on (crop, week_start) ...')
sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'])
assert sat[['crop', 'week_start']].duplicated().sum() == 0, 'satellite table has duplicate (crop,week_start) rows'
df = checked_merge(df, sat, on=['crop', 'week_start'], how='left', label='M3/M4 climate/satellite')

# ─────────────────────────────────────────────────────────────────────────────
# M6 — Policy/trade events, join key (crop, week_start) -- see docstring:
# still only 2017-2026, left as NaN before that, deliberately not zero-filled.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Joining policy/trade (M6) on (crop, week_start) -- 2017-2026 only, '
      'pre-2017 rows left NaN by design (see docstring) ...')
policy = pd.read_csv(POLICY_FILE, parse_dates=['week_start'])
assert policy[['crop', 'week_start']].duplicated().sum() == 0, 'policy table has duplicate (crop,week_start) rows'
df = checked_merge(df, policy, on=['crop', 'week_start'], how='left', label='M6 policy')


# ─────────────────────────────────────────────────────────────────────────────
# 5. MISSING-VALUE DIAGNOSTICS -- per COLUMN, not "any column in the layer is
# missing". This layer mixes genuinely-long series (ERA5/CHIRPS/MODIS back to
# 2000, most CMIE series back to 2000-2002) with genuinely-short ones
# (Sentinel-2, mid-2015 floor; WPI/IIP, 2011 base-year floor; non-subsidized
# LPG, ~2013-14 dual-pricing-scheme floor) -- an "any missing" row-level
# aggregate would be dominated by whichever single column has the latest
# floor and falsely read as "this whole layer is mostly missing pre-2017",
# which is not true for most of its columns. Report each separately instead.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Missing-value check per COLUMN, split at 2017-01-01 (not "any '
      'column in the layer missing" -- see comment above for why) ...')
LAYER_COLS = {
    'M2 macro':          [c for c in macro.columns if c not in ('year', 'month')],
    'M3/M4 climate/sat': [c for c in sat.columns if c not in ('crop', 'week_start')],
    'M6 policy':         [c for c in policy.columns if c not in ('crop', 'week_start')],
}
pre_2017 = df['week_start'] < '2017-01-01'
for label, cols in LAYER_COLS.items():
    present = [c for c in cols if c in df.columns]
    if not present:
        continue
    print(f'  {label}:')
    for c in present:
        pct_pre  = df.loc[pre_2017, c].isna().mean() * 100
        pct_post = df.loc[~pre_2017, c].isna().mean() * 100
        print(f'    {c:<24s}: {pct_pre:5.1f}% missing pre-2017  |  {pct_post:5.1f}% missing 2017+')


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6] Saving baseline-phase panel ...')
df.to_csv(OUT_FILE, index=False, encoding='utf-8')
print(f'  Saved: {OUT_FILE}')
print(f'  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns')

print('\n' + '=' * 65)
print('Script 34 complete.')
print('\nNext: Stage 3 -- train the Phase 1 baseline model on this panel using')
print('the same 4 rolling-origin folds as Script 15, persisting per-row')
print('out-of-fold predictions (not just aggregate metrics) for Stage 4.')
