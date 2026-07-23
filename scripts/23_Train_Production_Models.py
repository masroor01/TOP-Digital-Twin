# -*- coding: utf-8 -*-
"""
Script 23 — Train Production Models (for the Simulation Dashboard)
========================================================================
Script 15 trains and discards models on every CV fold — none are ever
saved as a reusable artifact. This script trains ONE final, deployable
LightGBM model per (crop, horizon) — 3 crops x 4 horizons = 12 models —
using the M6 feature set (the full pipeline: price, arrivals, macro,
climate, satellite, infrastructure, policy) so the dashboard can let a
user toggle any of those inputs and see the price response.

Feature engineering is identical to Script 15's M6 variant (same lags,
rolling stats, seasonality, and all 6 data-layer joins) — duplicated here
rather than imported because Script 15 is a top-level script that runs
its whole ablation study on import.

Training procedure per (crop, horizon):
  1. Fit with early stopping on a held-out tail (last 26 weeks) to find
     the right number of trees (best_iteration).
  2. Refit on the FULL 2017-2025 history using that fixed tree count —
     standard "validate then refit on everything" practice for a
     production model, so the final model isn't wasting the most recent
     6 months of data on validation alone.

Outputs (Model_Output/production_models/):
  {crop}_{h}w.joblib        the trained LGBMRegressor
  feature_columns.json      exact feature list + dtype per model (dashboard
                             must build the input vector in this order)
  reference_rows.csv        latest available feature vector per (crop,
                             market) — dashboard's baseline "current state"
                             for the scenario simulator
  feature_ranges.json       observed min/median/max per raw input feature,
                             for setting sensible dashboard slider bounds

Run: python scripts/23_Train_Production_Models.py
"""

import io, os, sys, json, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE     = r'C:\Users\masro\Documents\TOP_Digital_Twin'
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CMIE_FILE= os.path.join(BASE, 'data', 'cmie_macro',      'cmie_macro_2017_2025.csv')
RBI_FILE = os.path.join(BASE, 'data', 'rbi_dbie',        'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE= os.path.join(BASE, 'data', 'ppac_macro',      'ppac_diesel_lpg_2017_2025.csv')
SAT_FILE = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
WAGE_FILE  = os.path.join(BASE, 'data', 'labour_wages',   'wage_agri_state_monthly.csv')
COLD_FILE  = os.path.join(BASE, 'data', 'infrastructure', 'cold_storage_by_state.csv')
ROAD_FILE  = os.path.join(BASE, 'data', 'infrastructure', 'road_density_state_annual.csv')
POLICY_FILE= os.path.join(BASE, 'data', 'policy_trade',   'policy_weekly_features.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output', 'production_models')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED     = 42
VAL_WEEKS = 26   # held-out tail for early-stopping tree-count selection

LGBM_PARAMS = dict(
    objective='regression', metric='rmse', n_estimators=1000,
    learning_rate=0.05, num_leaves=127, max_depth=-1,
    min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
    random_state=SEED, verbose=-1,
)

LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS = [4, 8, 13]

print('=' * 65)
print('SCRIPT 23: TRAIN PRODUCTION MODELS (M6, for dashboard)')
print('=' * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD + JOIN ALL LAYERS (identical to Script 15)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Loading panel + all layers ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2025-12-31')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)
df['year']  = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month

macro_dfs = [pd.read_csv(f) for f in [CMIE_FILE, RBI_FILE, PPAC_FILE] if os.path.exists(f)]
macro = macro_dfs[0]
for m in macro_dfs[1:]:
    macro = macro.merge(m, on=['year', 'month'], how='outer', suffixes=('', '_dup'))
    macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
macro = macro.drop(columns=[c for c in ['date', 'date_x', 'date_y'] if c in macro.columns])
MACRO_COLS = [c for c in macro.columns if c not in ('date', 'year', 'month')]
df = df.merge(macro, on=['year', 'month'], how='left')

sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'])
ERA5_COLS   = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38']
CHIRPS_COLS = ['chirps_rain_mm', 'chirps_rain_max', 'chirps_excess']
S2_COLS     = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']
MODIS_COLS  = ['modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']
roll_specs = [
    ('era5_heat_35', 'sum', [4, 8]), ('chirps_rain_mm', 'sum', [4, 8]),
    ('s2_ndvi', 'mean', [4]), ('s2_ndvi_anom', 'mean', [4]), ('modis_lst_mean', 'mean', [4]),
]
roll_cols = []
for col, func, windows in roll_specs:
    if col not in sat.columns:
        continue
    for w in windows:
        new_col = f'{col}_roll{w}'
        agg = 'sum' if func == 'sum' else 'mean'
        sat[new_col] = (sat.groupby('crop')[col]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=2).agg(agg)))
        roll_cols.append(new_col)
CLIMATE_FEATS = [c for c in ERA5_COLS + CHIRPS_COLS if c in sat.columns]
CLIMATE_FEATS += [c for c in roll_cols if any(s in c for s in ['era5_heat', 'chirps_rain'])]
SAT_FEATS = [c for c in S2_COLS + MODIS_COLS if c in sat.columns]
SAT_FEATS += [c for c in roll_cols if any(s in c for s in ['s2_', 'modis_'])]
df = df.merge(sat[['week_start', 'crop'] + CLIMATE_FEATS + SAT_FEATS], on=['crop', 'week_start'], how='left')

INFRA_FEATS, POLICY_FEATS = [], []
if os.path.exists(WAGE_FILE):
    wages = pd.read_csv(WAGE_FILE)[['state', 'year', 'month', 'wage_agri_men', 'wage_agri_women']]
    df = df.merge(wages, on=['state', 'year', 'month'], how='left')
    INFRA_FEATS += ['wage_agri_men', 'wage_agri_women']
if os.path.exists(COLD_FILE):
    cold = pd.read_csv(COLD_FILE)[['state', 'n_facilities', 'capacity_mt']]
    cold = cold.rename(columns={'n_facilities': 'cold_storage_n_facilities', 'capacity_mt': 'cold_storage_capacity_mt'})
    df = df.merge(cold, on=['state'], how='left')
    INFRA_FEATS += ['cold_storage_n_facilities', 'cold_storage_capacity_mt']
if os.path.exists(ROAD_FILE):
    road = pd.read_csv(ROAD_FILE)[['state', 'year', 'road_density_per_100_sqkm']]
    df = df.merge(road, on=['state', 'year'], how='left')
    INFRA_FEATS += ['road_density_per_100_sqkm']
if os.path.exists(POLICY_FILE):
    policy_cols = ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct',
                   'market_intervention_flag', 'operation_greens_active']
    policy = pd.read_csv(POLICY_FILE, parse_dates=['week_start'])[['crop', 'week_start'] + policy_cols]
    df = df.merge(policy, on=['crop', 'week_start'], how='left')
    POLICY_FEATS += policy_cols

print(f'  Panel joined: {len(df):,} rows x {df.shape[1]} columns')


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING (identical to Script 15)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Engineering features ...')

def build_features(df_in):
    out = {}
    for crop in CROPS:
        sub = df_in[df_in['crop'] == crop].copy()
        sub = sub.sort_values(['market', 'week_start'])
        sub['log_price'] = np.log1p(sub['modal_price_weighted'])
        for lag in LAG_WEEKS:
            sub[f'price_lag_{lag}'] = sub.groupby('market')['log_price'].shift(lag)
        for w in ROLL_WINS:
            g = sub.groupby('market')['log_price']
            sub[f'price_roll_mean_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).mean())
            sub[f'price_roll_std_{w}']  = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
        if 'arrivals_tonnes_week' in sub.columns:
            sub['log_arr'] = np.log1p(sub['arrivals_tonnes_week'].clip(lower=0))
            for lag in [1, 2, 4]:
                sub[f'arr_lag_{lag}'] = sub.groupby('market')['log_arr'].shift(lag)
            for w in [4, 8]:
                sub[f'arr_roll_mean_{w}'] = sub.groupby('market')['log_arr'].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        sub['price_yoy'] = sub.groupby('market')['log_price'].shift(52)
        sub['week_num'] = sub['week_start'].dt.isocalendar().week.astype(int)
        sub['sin_week'] = np.sin(2 * np.pi * sub['week_num'] / 52)
        sub['cos_week'] = np.cos(2 * np.pi * sub['week_num'] / 52)
        sub['sin2_week'] = np.sin(4 * np.pi * sub['week_num'] / 52)
        sub['cos2_week'] = np.cos(4 * np.pi * sub['week_num'] / 52)
        m = sub['week_start'].dt.month
        if crop == 'tomato':
            sub['season_peak_arrival'] = m.isin([11, 12, 1, 2]).astype(int)
            sub['season_lean']         = m.isin([5, 6, 7]).astype(int)
            sub['season_kharif']       = m.isin([8, 9, 10]).astype(int)
        elif crop == 'onion':
            sub['season_rabi_arrival'] = m.isin([2, 3, 4, 5]).astype(int)
            sub['season_lean']         = m.isin([9, 10, 11]).astype(int)
            sub['season_kharif']       = m.isin([8, 9]).astype(int)
        elif crop == 'potato':
            sub['season_harvest']      = m.isin([2, 3, 4]).astype(int)
            sub['season_storage']      = m.isin([5, 6, 7, 8, 9]).astype(int)
            sub['season_lean']         = m.isin([10, 11]).astype(int)
        for col in ['state', 'market']:
            if col in sub.columns:
                sub[f'{col}_enc'] = pd.Categorical(sub[col]).codes
        sub['year_trend'] = sub['week_start'].dt.year - 2017
        out[crop] = sub
    return out

feat = build_features(df)

PRICE_FEATS = (
    [f'price_lag_{lag}' for lag in LAG_WEEKS] +
    [f'price_roll_mean_{w}' for w in ROLL_WINS] +
    [f'price_roll_std_{w}' for w in ROLL_WINS] +
    ['price_yoy', 'sin_week', 'cos_week', 'sin2_week', 'cos2_week',
     'week_num', 'year_trend', 'market_enc', 'state_enc',
     'season_peak_arrival', 'season_lean', 'season_kharif',
     'season_rabi_arrival', 'season_harvest', 'season_storage']
)
ARR_FEATS = (
    ['log_arr'] + [f'arr_lag_{lag}' for lag in [1, 2, 4]] +
    [f'arr_roll_mean_{w}' for w in [4, 8]]
)
M6_FEATS = PRICE_FEATS + ARR_FEATS + MACRO_COLS + CLIMATE_FEATS + SAT_FEATS + INFRA_FEATS + POLICY_FEATS

print(f'  M6 feature list: {len(M6_FEATS)} candidate columns')


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN + SAVE ONE MODEL PER (crop, horizon)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Training production models (M6, full history) ...\n')

feature_columns = {}
reference_rows = []

for crop in CROPS:
    df_crop = feat[crop]
    fcols = [c for c in M6_FEATS if c in df_crop.columns]

    for h in HORIZONS:
        df_h = df_crop.copy()
        df_h['target'] = df_h.groupby('market')['log_price'].shift(-h)
        required = ['target', 'price_lag_1']
        df_h = df_h.dropna(subset=[c for c in required if c in df_h.columns])

        max_date = df_h['week_start'].max()
        val_cutoff = max_date - pd.Timedelta(weeks=VAL_WEEKS)

        train = df_h[df_h['week_start'] <= val_cutoff]
        val   = df_h[df_h['week_start'] > val_cutoff]

        X_tr, y_tr = train[fcols].fillna(0), train['target']
        X_va, y_va = val[fcols].fillna(0), val['target']

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        best_iter = model.best_iteration_ or LGBM_PARAMS['n_estimators']

        # Refit on ALL data (train + val) using the tree count found above
        final_params = dict(LGBM_PARAMS)
        final_params['n_estimators'] = best_iter
        final_model = lgb.LGBMRegressor(**final_params)
        X_all, y_all = df_h[fcols].fillna(0), df_h['target']
        final_model.fit(X_all, y_all)

        model_path = os.path.join(OUT_DIR, f'{crop}_{h}w.joblib')
        joblib.dump(final_model, model_path)
        feature_columns[f'{crop}_{h}w'] = fcols

        print(f'  {crop:7s} h={h:>2}w | trees={best_iter:>4} (from val) | '
              f'refit on {len(df_h):,} rows | saved {os.path.basename(model_path)}')

    # Reference rows: latest available feature vector per market, for the
    # dashboard's baseline "current state" (built once per crop, reused
    # across all 4 horizon models since features are the same up to target)
    latest_idx = df_crop.sort_values('week_start').groupby('market').tail(1).index
    latest = df_crop.loc[latest_idx].copy()
    latest['crop'] = crop
    reference_rows.append(latest)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE METADATA FOR THE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Saving dashboard metadata ...')

fcols_path = os.path.join(OUT_DIR, 'feature_columns.json')
with open(fcols_path, 'w', encoding='utf-8') as f:
    json.dump(feature_columns, f, indent=2)
print(f'  Saved: {fcols_path}')

ref_df = pd.concat(reference_rows, ignore_index=True)
# log_price isn't an M6 feature (it's the target's source column) but the
# dashboard needs it to display "last observed price" — include explicitly
keep_cols = (['crop', 'market', 'state', 'week_start', 'log_price'] +
             sorted(set(M6_FEATS) & set(ref_df.columns)))
ref_df = ref_df[keep_cols]
ref_path = os.path.join(OUT_DIR, 'reference_rows.csv')
ref_df.to_csv(ref_path, index=False, encoding='utf-8')
print(f'  Saved: {ref_path}  ({len(ref_df):,} market baselines)')

# Slider bounds: min/median/max for the "what-if" input variables.
# IMPORTANT: computed from the FULL historical panel (df, all weeks
# 2017-2025), not from ref_df (latest week only) — found in dashboard
# testing that national-level macro variables (diesel, repo rate, USD/INR)
# are identical across all markets within any single week, so their
# cross-sectional range in ref_df is degenerate (min==max), which crashes
# st.slider(). The full time series has real historical variation.
SIMULATABLE = (['export_banned', 'mep_usd_per_tonne', 'export_duty_pct',
                 'market_intervention_flag', 'operation_greens_active'] +
                CLIMATE_FEATS[:6] + SAT_FEATS[:4] +
                ['diesel_4city_rs_litre', 'repo_rate_pct', 'usdinr_monthly_avg',
                 'wage_agri_men', 'wage_agri_women'])
ranges = {}
for col in SIMULATABLE:
    if col in df.columns:
        s = df[col].dropna()
        if len(s) > 0:
            ranges[col] = {'min': float(s.min()), 'median': float(s.median()), 'max': float(s.max())}
ranges_path = os.path.join(OUT_DIR, 'feature_ranges.json')
with open(ranges_path, 'w', encoding='utf-8') as f:
    json.dump(ranges, f, indent=2)
print(f'  Saved: {ranges_path}  ({len(ranges)} simulatable features)')

print('\n' + '=' * 65)
print('Script 23 complete. 12 production models saved to')
print(f'  {OUT_DIR}')
print('\nNext: scripts/24_Simulation_Dashboard.py (Streamlit app)')
