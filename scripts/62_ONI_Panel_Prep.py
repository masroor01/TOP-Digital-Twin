# -*- coding: utf-8 -*-
"""
Script 62 — Panel-Prep: NOAA ONI (M9 candidate)
==============================================================
Reshapes Script 56's raw ONI acquisition (3-month rolling seasons, e.g.
DJF/JFM/FMA, one new reading effectively released each calendar month)
into a plain (year, month) monthly series joinable the same way M2 macro
is, plus explicit lagged versions -- ENSO's effect on agricultural supply
doesn't show up the same month a season is measured, it shows up months
later as the anomaly works through planting/growing cycles. The original
review docs this project's layer list was built from specifically called
out oni_lag_3m/oni_lag_4m as the features worth testing, not the raw
contemporaneous value alone (see Script 56's docstring).

Convention: a season's value becomes "the ONI reading for month M" when
that season's season_end_month == M -- i.e. the most recently COMPLETED
season as of month M, the same one-month-behind convention every other
monthly macro series in this project already uses (WPI, CMIE releases,
etc. are joined directly on (year, month) with no extra artificial lag
beyond their own natural publication cadence).

Output:
  data/noaa_oni/oni_panel_monthly.csv
    (year, month, oni_anom, oni_lag_3m, oni_lag_4m)

Run: python scripts/62_ONI_Panel_Prep.py
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONI_SRC = os.path.join(BASE, 'data', 'noaa_oni', 'oni_seasonal.csv')
ONI_OUT = os.path.join(BASE, 'data', 'noaa_oni', 'oni_panel_monthly.csv')

print('=' * 65)
print('SCRIPT 62: PANEL-PREP FOR NOAA ONI (M9 CANDIDATE)')
print('=' * 65)

oni = pd.read_csv(ONI_SRC)

# One row per (year, month) = the season ending in that month. season_end_month
# already accounts for DJF's year-wrap (season_end_month=2, year=the DJF row's
# own YR column, i.e. the year of the Jan/Feb months) -- Script 56 built this
# column explicitly for this reason, use it as-is.
monthly = oni[['year', 'season_end_month', 'oni_anom', 'sst_total']].rename(
    columns={'season_end_month': 'month'})

dupes = monthly[monthly.duplicated(subset=['year', 'month'], keep=False)]
if len(dupes):
    raise ValueError(f'{len(dupes)} duplicate (year, month) rows after season->month '
                      f'mapping -- two seasons ending in the same calendar month, '
                      f'check Script 56\'s season_end_month construction.')

monthly = monthly.sort_values(['year', 'month']).reset_index(drop=True)

for lag in [3, 4]:
    monthly[f'oni_lag_{lag}m'] = monthly['oni_anom'].shift(lag)

monthly = monthly[['year', 'month', 'oni_anom', 'oni_lag_3m', 'oni_lag_4m']]
monthly.to_csv(ONI_OUT, index=False, encoding='utf-8')

first, last = monthly.dropna(subset=['oni_lag_4m']).iloc[0], monthly.iloc[-1]
print(f'\nSaved: {ONI_OUT}  ({len(monthly)} months, '
      f'{int(monthly.iloc[0]["year"])}-{int(monthly.iloc[0]["month"]):02d} -> '
      f'{int(last["year"])}-{int(last["month"]):02d}; '
      f'oni_lag_4m usable from {int(first["year"])}-{int(first["month"]):02d})')
print('\nScript 62 complete.')
