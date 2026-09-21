# -*- coding: utf-8 -*-
"""
Shared data-layer loading & join logic for Script 15 (ablation study) and
Script 22 (master panel join) -- both build the same M0-M7 layer stack
from the same source files, and until this module existed each
implemented the reading/uniqueness-checking/joining independently.

M8 (fertilizer MRP, CPI-AL/RL -- added 2026-09-25) is wired into Script 22
only, not Script 15's ablation stack: M7 went through a dedicated fold-level
statistical test (see Model_Output/MANIFEST.md) before a call was made on
whether to include or exclude it from production training; M8 hasn't had
that same scrutiny yet, so it's available in the joined panel file for
exploration but isn't silently added to what Script 23 trains on.

Extracted 2026-09-11 after a VS Code review flagged this project's
established risk with exactly this class of duplication: two independent
implementations of "the same operation" drifting apart (see the Script
20/21 PANEL_END divergence, caught and fixed 2026-09-02 -- one script's
fix to a hardcoded bound never propagated to the other's copy of the same
logic). Centralizing the literal join mechanics here -- file paths, join
keys, uniqueness assertions, row-count checks -- removes that risk for
every layer below.

What's deliberately NOT unified: Script 15 layers extra work on top of
several of these joins -- rolling climate/satellite features computed
before its own merge, forward-fill for M2/M5's tail-coverage gaps, and
(for M7 specifically) missingness flags instead of forward-fill, because
M7's gaps are core seasonal/historical coverage rather than a tail gap
(see Scripts 51-54). Script 22 wants the plain joined columns for
open-ended exploration instead. Those differences are real, deliberate
choices for different consumers, not accidental drift -- forcing them
into one shared code path would hide that distinction rather than remove
duplication. `join_drought()`'s `add_missing_flags` parameter is the one
place a shared function still branches, because both variants share
100% of their join mechanics and differ only in which extra columns they
attach.

Any script using this module computes its own BASE the same way it always
has (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`) --
this module's own path constants below use its own file location, which
lives in the same `scripts/` directory, so they resolve identically.
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CMIE_FILE     = os.path.join(BASE, 'data', 'cmie_macro', 'cmie_macro_2017_2025.csv')
RBI_FILE      = os.path.join(BASE, 'data', 'rbi_dbie', 'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE     = os.path.join(BASE, 'data', 'ppac_macro', 'ppac_diesel_lpg_2017_2025.csv')
SAT_FILE      = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
WAGE_FILE     = os.path.join(BASE, 'data', 'labour_wages', 'wage_agri_state_monthly.csv')
COLD_FILE     = os.path.join(BASE, 'data', 'infrastructure', 'cold_storage_by_state.csv')
ROAD_FILE     = os.path.join(BASE, 'data', 'infrastructure', 'road_density_state_annual.csv')
POLICY_FILE   = os.path.join(BASE, 'data', 'policy_trade', 'policy_weekly_features.csv')
TRIGGER1_FILE = os.path.join(BASE, 'data', 'drought_vedas', 'trigger1_panel_weekly.csv')
IDM_FILE      = os.path.join(BASE, 'data', 'drought_idm', 'cdi_panel_weekly.csv')
FERT_FILE     = os.path.join(BASE, 'data', 'fertilizer_mrp', 'fert_mrp_panel_monthly.csv')
CPI_ALRL_FILE = os.path.join(BASE, 'data', 'cpi_al_rl', 'cpi_alrl_panel_state_monthly.csv')
ONI_FILE = os.path.join(BASE, 'data', 'noaa_oni', 'oni_panel_monthly.csv')

POLICY_COLS = ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct',
               'market_intervention_flag', 'operation_greens_active']

# Raw Script-14 column groups -- shared reference so a caller that wants to
# engineer its own rolling features (Script 15) doesn't hardcode a second
# copy of this list.
ERA5_COLS   = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38']
CHIRPS_COLS = ['chirps_rain_mm', 'chirps_rain_max', 'chirps_excess']
S2_COLS     = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']
MODIS_COLS  = ['modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']


def assert_unique(frame, keys, label):
    n_dup = frame[keys].duplicated().sum()
    if n_dup:
        raise ValueError(f'{label}: {n_dup} duplicate rows on {keys} -- '
                          f'join would silently fan out the panel.')


def checked_merge(left, right, on, how, label, verbose=True):
    """Left-join that asserts the right side is unique on `on` (clear
    error before the merge even runs) AND that the row count is preserved
    after it (catches anything the uniqueness check missed)."""
    assert_unique(right, on, label)
    n_before = len(left)
    merged = left.merge(right, on=on, how=how)
    n_after = len(merged)
    if verbose:
        status = 'OK' if n_after == n_before else 'ROW COUNT CHANGED'
        print(f'  [{label}] join on {on}: {n_before:,} -> {n_after:,} rows  [{status}]')
    if n_after != n_before:
        raise ValueError(
            f'{label}: row count changed from {n_before:,} to {n_after:,} after '
            f'joining on {on} -- the right-hand table is not unique on that key. '
            f'Fix the source file or the join key before proceeding.')
    return merged


def load_macro():
    """Reads + outer-merges CMIE/RBI/PPAC on (year, month).
    Returns (macro_df, macro_cols); (None, []) if no macro file exists."""
    macro_dfs = []
    for fpath in [CMIE_FILE, RBI_FILE, PPAC_FILE]:
        if os.path.exists(fpath):
            macro_dfs.append(pd.read_csv(fpath))
    if not macro_dfs:
        return None, []
    macro = macro_dfs[0]
    for m in macro_dfs[1:]:
        macro = macro.merge(m, on=['year', 'month'], how='outer', suffixes=('', '_dup'))
        dup_cols = [c for c in macro.columns if c.endswith('_dup')]
        if dup_cols:
            print(f'  WARNING: dropping overlapping macro columns from a later source: {dup_cols}')
        macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
    macro = macro.drop(columns=[c for c in ['date', 'date_x', 'date_y'] if c in macro.columns])
    macro_cols = [c for c in macro.columns if c not in ('year', 'month')]
    return macro, macro_cols


def join_macro(df, verbose=True):
    """Join key: (year, month). Returns (df, macro_cols)."""
    macro, macro_cols = load_macro()
    if macro is None:
        return df, []
    df = checked_merge(df, macro, on=['year', 'month'], how='left', label='M2 macro', verbose=verbose)
    return df, macro_cols


def load_satellite_climate():
    """Raw read of Script 14's output -- no derived/rolling columns. Each
    caller decides whether to engineer rolling features (Script 15 does,
    before its own merge) and does its own checked_merge on (crop,
    week_start), since what gets merged genuinely differs by caller."""
    return pd.read_csv(SAT_FILE, parse_dates=['week_start'])


def join_wages(df, verbose=True):
    """Join key: (state, year, month). Returns (df, feature_cols)."""
    if not os.path.exists(WAGE_FILE):
        return df, []
    wages = pd.read_csv(WAGE_FILE)[['state', 'year', 'month', 'wage_agri_men', 'wage_agri_women']]
    df = checked_merge(df, wages, on=['state', 'year', 'month'], how='left',
                        label='M5a wages', verbose=verbose)
    return df, ['wage_agri_men', 'wage_agri_women']


def join_cold_storage(df, verbose=True):
    """Join key: (state) -- static. Returns (df, feature_cols)."""
    if not os.path.exists(COLD_FILE):
        return df, []
    cold = pd.read_csv(COLD_FILE)[['state', 'n_facilities', 'capacity_mt']]
    cold = cold.rename(columns={'n_facilities': 'cold_storage_n_facilities',
                                 'capacity_mt': 'cold_storage_capacity_mt'})
    df = checked_merge(df, cold, on=['state'], how='left', label='M5b cold storage', verbose=verbose)
    return df, ['cold_storage_n_facilities', 'cold_storage_capacity_mt']


def join_road_density(df, verbose=True):
    """Join key: (state, year). Returns (df, feature_cols)."""
    if not os.path.exists(ROAD_FILE):
        return df, []
    road = pd.read_csv(ROAD_FILE)[['state', 'year', 'road_density_per_100_sqkm']]
    df = checked_merge(df, road, on=['state', 'year'], how='left',
                        label='M5c road density', verbose=verbose)
    return df, ['road_density_per_100_sqkm']


def join_policy(df, verbose=True):
    """Join key: (crop, week_start). Returns (df, feature_cols)."""
    if not os.path.exists(POLICY_FILE):
        return df, []
    policy = pd.read_csv(POLICY_FILE, parse_dates=['week_start'])[['crop', 'week_start'] + POLICY_COLS]
    df = checked_merge(df, policy, on=['crop', 'week_start'], how='left',
                        label='M6 policy', verbose=verbose)
    return df, POLICY_COLS


def join_drought(df, add_missing_flags=True, verbose=True):
    """Join key: (state, district, week_start), both Trigger-1 and IDM
    CDI. Never forward-filled -- M7's gaps are core seasonal/historical
    coverage (whole off-season stretches, whole pre-2021/2022 years), not
    a tail gap safe to bridge; see Scripts 51-54.

    add_missing_flags=True (Script 15's need): adds trigger1_missing /
    cdi_missing and returns ['trigger1', 'trigger1_missing', 'cdi',
    'cdi_missing'] as the model-usable feature list, so a downstream
    fillna(0) can be told apart from a real 0/Normal reading.

    add_missing_flags=False (Script 22's need): keeps the raw
    `trigger1_yn` / `drought_category` columns instead, for open-ended
    exploration, and returns ['trigger1', 'cdi'] as the value columns.
    """
    feats = []
    if os.path.exists(TRIGGER1_FILE):
        extra = [] if add_missing_flags else ['trigger1_yn']
        t1 = pd.read_csv(TRIGGER1_FILE, parse_dates=['week_start'])[
            ['state', 'district', 'week_start', 'trigger1'] + extra]
        df = checked_merge(df, t1, on=['state', 'district', 'week_start'], how='left',
                            label='M7a Trigger-1', verbose=verbose)
        if add_missing_flags:
            df['trigger1_missing'] = df['trigger1'].isna().astype(int)
            feats += ['trigger1', 'trigger1_missing']
        else:
            feats += ['trigger1']

    if os.path.exists(IDM_FILE):
        extra = [] if add_missing_flags else ['drought_category']
        idm = pd.read_csv(IDM_FILE, parse_dates=['week_start'])[
            ['state', 'district', 'week_start', 'cdi'] + extra]
        df = checked_merge(df, idm, on=['state', 'district', 'week_start'], how='left',
                            label='M7b IDM CDI', verbose=verbose)
        if add_missing_flags:
            df['cdi_missing'] = df['cdi'].isna().astype(int)
            feats += ['cdi', 'cdi_missing']
        else:
            feats += ['cdi']

    return df, feats


def join_fertilizer(df, add_missing_flags=False, verbose=True):
    """M8a. Join key: (year, month) -- national, no state dimension, same
    shape as M2 macro. Source: Script 59's pivot of Script 57's raw MRP
    acquisition, narrowed to Urea/DAP/MOP (see Script 59 docstring for why
    those three).

    add_missing_flags=True (Script 15 ablation's need): the source starts
    Jan-2018 -- panel rows from 2017 have no fertilizer reading at all, a
    real ~1-year core gap at the front of the panel, not a recent-data lag
    like macro/wages' tail gaps. Adds fert_urea_mrp_missing/etc. and
    returns the flags alongside the values, same convention as
    join_drought(add_missing_flags=True), so a downstream fillna(0) can be
    told apart from a real zero reading (fertilizer MRP is never
    genuinely 0).

    add_missing_flags=False (Script 22's need): plain value columns only,
    for open-ended exploration. Returns (df, feature_cols)."""
    if not os.path.exists(FERT_FILE):
        return df, []
    cols = ['fert_urea_mrp', 'fert_dap_mrp', 'fert_mop_mrp']
    fert = pd.read_csv(FERT_FILE)
    df = checked_merge(df, fert, on=['year', 'month'], how='left',
                        label='M8a fertilizer MRP', verbose=verbose)
    if add_missing_flags:
        feats = []
        for c in cols:
            df[f'{c}_missing'] = df[c].isna().astype(int)
            feats += [c, f'{c}_missing']
        return df, feats
    return df, cols


def join_cpi_alrl(df, add_missing_flags=False, verbose=True):
    """M8b. Join key: (state, year, month) -- same shape as M5a wages.
    Source: Script 59's pivot + panel-state crosswalk of Script 58's raw
    CPI-AL/RL acquisition. NaN for panel states the source genuinely has no
    row for (Chandigarh -- see Script 59), not a join failure.

    add_missing_flags=True (Script 15 ablation's need): unlike M5's wages,
    this series only exists Jun-2025 onward (the Base-2019 rebase) with a
    documented gap (May-2026) inside even that short window -- this is
    core coverage, not a tail lag safe to forward-fill or zero-fill (same
    reasoning as join_drought). Adds cpi_al_missing/cpi_rl_missing.

    add_missing_flags=False (Script 22's need): plain value columns only.
    Returns (df, feature_cols)."""
    if not os.path.exists(CPI_ALRL_FILE):
        return df, []
    cpi = pd.read_csv(CPI_ALRL_FILE)[['state', 'year', 'month', 'cpi_al', 'cpi_rl']]
    df = checked_merge(df, cpi, on=['state', 'year', 'month'], how='left',
                        label='M8b CPI-AL/RL', verbose=verbose)
    if add_missing_flags:
        feats = []
        for c in ['cpi_al', 'cpi_rl']:
            df[f'{c}_missing'] = df[c].isna().astype(int)
            feats += [c, f'{c}_missing']
        return df, feats
    return df, ['cpi_al', 'cpi_rl']


def join_oni(df, verbose=True):
    """M9 candidate. Join key: (year, month) -- national, no state
    dimension, same shape as M2 macro. Source: Script 62's pivot of
    Script 56's raw ONI acquisition, reshaped from overlapping 3-month
    seasons into a plain (year, month) series plus lags. Includes
    oni_lag_3m/oni_lag_4m alongside the contemporaneous oni_anom -- the
    original review docs this layer's inclusion was based on specifically
    named the lagged versions as the ones worth testing (ENSO's effect on
    supply shows up months after the anomaly, not the same month). No
    missingness-flag treatment needed: ONI has continuous monthly coverage
    back to 1950, decades before the panel starts, so there's no core
    coverage gap the way drought/CPI-AL/RL have -- ordinary NaN (only at
    the very start of the panel, if ever) is fine to fillna(0) like any
    other complete series. Returns (df, feature_cols)."""
    if not os.path.exists(ONI_FILE):
        return df, []
    oni = pd.read_csv(ONI_FILE)
    df = checked_merge(df, oni, on=['year', 'month'], how='left',
                        label='M9 ONI', verbose=verbose)
    return df, ['oni_anom', 'oni_lag_3m', 'oni_lag_4m']
