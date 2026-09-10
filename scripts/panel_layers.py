# -*- coding: utf-8 -*-
"""
Shared data-layer loading & join logic for Script 15 (ablation study) and
Script 22 (master panel join) -- both build the same M0-M7 layer stack
from the same source files, and until this module existed each
implemented the reading/uniqueness-checking/joining independently.

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
