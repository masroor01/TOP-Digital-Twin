# -*- coding: utf-8 -*-
"""Shared helpers for the manuscript Section 2 scripts (78, 84, 85): the current panel and the latest data point of each layer."""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS = ['tomato', 'onion', 'potato']


def load_panel():
    """The three tracked per-crop weekly panels (refreshed by the weekly workflow), Kerala and Keralam as one state."""
    fr = []
    for c in CROPS:
        d = pd.read_csv(os.path.join(BASE, 'data', 'agmarknet_weekly', f'{c}_weekly_panel.csv'), parse_dates=['week_start'])
        d['crop'] = c
        fr.append(d)
    p = pd.concat(fr, ignore_index=True)
    p['state'] = p['state'].replace({'Keralam': 'Kerala'})
    p['market_id'] = p['market_id'].astype(float).astype(int)
    return p


def _month_end(df):
    return pd.to_datetime(dict(year=df['year'], month=df['month'], day=1)) + pd.offsets.MonthEnd(0)


def layer_latest():
    """{layer label: latest data point (Timestamp)} read from the files themselves, not typed in."""
    out = {}
    p = os.path.join(BASE, 'data', 'agmarknet_weekly')
    out['Market prices and arrivals (Agmarknet)'] = max(pd.read_csv(os.path.join(p, f'{c}_weekly_panel.csv'), usecols=['week_start'], parse_dates=['week_start'])['week_start'].max() for c in CROPS)
    cw = pd.read_csv(os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv'), parse_dates=['week_start'])
    for lab, col in (('ERA5-Land temperature', 'era5_tmax'), ('CHIRPS rainfall', 'chirps_rain_mm'), ('Sentinel-2 vegetation', 's2_ndvi'), ('MODIS vegetation and land temperature', 'modis_ndvi')):
        out[lab] = cw.loc[cw[col].notna(), 'week_start'].max()
    st = json.load(open(os.path.join(BASE, 'Model_Output', 'production_models', 'macro_climate_staleness.json')))['onion']
    asof = lambda keys: max(pd.Timestamp(st[k]['as_of']) for k in keys if k in st)
    out['Macro (CMIE: credit, trade, WPI, wages, crude, IIP, USD/INR)'] = asof(['bank_credit_agri_cr', 'export_veg_usd_mn', 'import_veg_usd_mn', 'crude_oil_usd_bbl', 'iip_food_proc', 'agri_wages_rs_day', 'usdinr_monthly_avg', 'wpi_onion'])
    out['Policy rates (RBI)'] = asof(['repo_rate_pct', 'reverse_repo_pct'])
    out['Fuel prices (PPAC)'] = asof(['diesel_4city_rs_litre', 'lpg_nonsub_4city_rs_cyl', 'diesel_delhi_per_L'])
    out['State agricultural wages (Labour Bureau)'] = asof(['wage_agri_men', 'wage_agri_women'])
    ev = pd.read_csv(os.path.join(BASE, 'data', 'policy_trade', 'export_policy_events.csv'), parse_dates=['effective_date'])
    out['Policy and trade log'] = ev['effective_date'].max()
    d = BASE + os.sep + 'data' + os.sep
    out['Drought (VEDAS / India Drought Monitor)'] = max(pd.read_csv(d + 'drought_idm/cdi_panel_weekly.csv', usecols=['week_start'], parse_dates=['week_start'])['week_start'].max(),
                                                      pd.read_csv(d + 'drought_vedas/trigger1_panel_weekly.csv', usecols=['week_start'], parse_dates=['week_start'])['week_start'].max())
    for lab, f in (('Fertiliser prices (fert.gov.in)', 'fertilizer_mrp/fert_mrp_panel_monthly.csv'), ('Rural price index (CPI-AL/RL)', 'cpi_al_rl/cpi_alrl_panel_state_monthly.csv'),
                   ('Oceanic Nino Index (NOAA)', 'noaa_oni/oni_panel_monthly.csv')):
        x = pd.read_csv(d + f)
        if lab.startswith('Rural'):
            x = x[x['cpi_al'].notna() | x['cpi_rl'].notna()]
        elif lab.startswith('Fert'):
            x = x[x[['fert_urea_mrp', 'fert_dap_mrp', 'fert_mop_mrp']].notna().any(axis=1)]
        else:
            x = x[x['oni_anom'].notna()]
        out[lab] = _month_end(x).max()
    return out
