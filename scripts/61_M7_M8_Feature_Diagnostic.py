# -*- coding: utf-8 -*-
"""
Script 61 — M7/M8 Feature Diagnostic: SHAP Usage + Redundancy Check
================================================================
Track A (forecast-accuracy diagnosis) of the 2026-09-21 strategic review:
the fold-level DM tests (Scripts 55/60) found neither M7 (drought) nor M8
(fertilizer MRP) -- nor, more importantly, the CORE M0-vs-M6 comparison --
clears a defensible significance bar. Before concluding "these layers
carry no signal" (or committing to an expensive CV redesign to get more
statistical power), this script checks two cheaper, more diagnostic
questions on data already in hand:

  1. USAGE: do the drought/fertilizer features get real split volume in
     the trained trees at all, and if so, does their SHAP direction match
     the economic prior (e.g. higher fertilizer cost -> higher price)?
     A feature with near-zero SHAP weight was never given a chance to
     move predictions, regardless of what a fold-level test finds.
  2. REDUNDANCY: are fertilizer/drought signals substantially collinear
     with information the model already has (year_trend, WPI, price
     lags/seasonality)? If so, weak marginal contribution is expected
     and doesn't mean the underlying economic relationship is false --
     it means the tree-boosting setup can't tell the two apart.

Trains ONE full-history model per (crop, horizon) for M7 and M8 -- same
train/validate/refit recipe as Script 23's actual production models (early
stop on the last 26 weeks, refit at that tree count on everything) -- since
neither M7 nor M8 has a saved joblib artifact (never wired into Script 23).
This is intentionally NOT the 5-fold CV setup Scripts 15/55/60 use; it's a
single richest-available-data fit, for feature-level diagnosis only, not
another significance test.

Outputs (Model_Output/):
  table_shap_m7_m8_layer_features.csv   per (crop, horizon, variant):
                                         SHAP rank/mean|SHAP|/mean-signed-SHAP
                                         for every M7/M8-specific feature
  table_fert_drought_redundancy.csv     correlation of each M7/M8 feature
                                         against existing macro/WPI/trend
                                         features

Run: python scripts/61_M7_M8_Feature_Diagnostic.py
"""

import io, os, sys, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import panel_layers as pl
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED     = 42
VAL_WEEKS = 26
SHAP_SAMPLE = 2000

LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS = [4, 8, 13]

LGBM_PARAMS = dict(
    objective='regression', metric='rmse', n_estimators=1000,
    learning_rate=0.05, num_leaves=127, max_depth=-1,
    min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
    random_state=SEED, verbose=-1,
)

print('=' * 65)
print('SCRIPT 61: M7/M8 FEATURE DIAGNOSTIC -- USAGE + REDUNDANCY')
print('=' * 65)

print('\n[1] Loading panel + all layers ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2030-12-31')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)
df['year'] = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month

df, macro_cols = pl.join_macro(df, verbose=False)
sat = pl.load_satellite_climate()
ERA5_COLS = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38']
CHIRPS_COLS = ['chirps_rain_mm', 'chirps_rain_max', 'chirps_excess']
S2_COLS = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']
MODIS_COLS = ['modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']
climate_feats = [c for c in ERA5_COLS + CHIRPS_COLS if c in sat.columns]
sat_feats = [c for c in S2_COLS + MODIS_COLS if c in sat.columns]
df = pl.checked_merge(df, sat[['crop', 'week_start'] + climate_feats + sat_feats],
                       on=['crop', 'week_start'], how='left', label='M3/M4', verbose=False)

df, wage_feats = pl.join_wages(df, verbose=False)
df, cold_feats = pl.join_cold_storage(df, verbose=False)
df, road_feats = pl.join_road_density(df, verbose=False)
infra_feats = wage_feats + cold_feats + road_feats
df, policy_feats = pl.join_policy(df, verbose=False)
df, drought_feats = pl.join_drought(df, add_missing_flags=True, verbose=False)
df, fert_feats = pl.join_fertilizer(df, add_missing_flags=True, verbose=False)
print(f'  Drought features (M7): {drought_feats}')
print(f'  Fertilizer features (M8): {fert_feats}')

# Same forward-fill treatment as Script 15/23 for tail-lagging sources
_ffill_cols = [c for c in ['wage_agri_men', 'wage_agri_women', 'road_density_per_100_sqkm'] if c in df.columns]
if _ffill_cols:
    df = df.sort_values(['state', 'week_start'])
    df[_ffill_cols] = df.groupby('state')[_ffill_cols].ffill()
_ffill_macro = [c for c in macro_cols if c in df.columns]
if _ffill_macro:
    df = df.sort_values('week_start')
    df[_ffill_macro] = df[_ffill_macro].ffill()
_ffill_sat = [c for c in (climate_feats + sat_feats) if c in df.columns]
if _ffill_sat:
    df = df.sort_values(['crop', 'week_start'])
    df[_ffill_sat] = df.groupby('crop')[_ffill_sat].ffill()
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)

print(f'  Panel joined: {len(df):,} rows x {df.shape[1]} columns')

print('\n[2] Engineering features ...')
def build_features(df_in):
    out = {}
    for crop in CROPS:
        sub = df_in[df_in['crop'] == crop].copy()
        sub = sub.sort_values(['market_id', 'week_start'])
        sub['log_price'] = np.log1p(sub['modal_price_weighted'])
        for lag in LAG_WEEKS:
            sub[f'price_lag_{lag}'] = sub.groupby('market_id')['log_price'].shift(lag)
        for w in ROLL_WINS:
            g = sub.groupby('market_id')['log_price']
            sub[f'price_roll_mean_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).mean())
            sub[f'price_roll_std_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())
        if 'arrivals_tonnes_week' in sub.columns:
            sub['log_arr'] = np.log1p(sub['arrivals_tonnes_week'].clip(lower=0))
            for lag in [1, 2, 4]:
                sub[f'arr_lag_{lag}'] = sub.groupby('market_id')['log_arr'].shift(lag)
            for w in [4, 8]:
                sub[f'arr_roll_mean_{w}'] = sub.groupby('market_id')['log_arr'].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        sub['price_yoy'] = sub.groupby('market_id')['log_price'].shift(52)
        sub['week_num'] = sub['week_start'].dt.isocalendar().week.astype(int)
        sub['sin_week'] = np.sin(2 * np.pi * sub['week_num'] / 52)
        sub['cos_week'] = np.cos(2 * np.pi * sub['week_num'] / 52)
        sub['sin2_week'] = np.sin(4 * np.pi * sub['week_num'] / 52)
        sub['cos2_week'] = np.cos(4 * np.pi * sub['week_num'] / 52)
        m = sub['week_start'].dt.month
        if crop == 'tomato':
            sub['season_peak_arrival'] = m.isin([11, 12, 1, 2]).astype(int)
            sub['season_lean'] = m.isin([5, 6, 7]).astype(int)
            sub['season_kharif'] = m.isin([8, 9, 10]).astype(int)
        elif crop == 'onion':
            sub['season_rabi_arrival'] = m.isin([2, 3, 4, 5]).astype(int)
            sub['season_lean'] = m.isin([9, 10, 11]).astype(int)
            sub['season_kharif'] = m.isin([8, 9]).astype(int)
        elif crop == 'potato':
            sub['season_harvest'] = m.isin([2, 3, 4]).astype(int)
            sub['season_storage'] = m.isin([5, 6, 7, 8, 9]).astype(int)
            sub['season_lean'] = m.isin([10, 11]).astype(int)
        sub['market_enc'] = pd.Categorical(sub['market_id']).codes
        sub['state_enc'] = pd.Categorical(sub['state']).codes
        sub['year_trend'] = sub['week_start'].dt.year - 2017
        out[crop] = sub
    return out

feat = build_features(df)

PRICE_FEATS = (
    [f'price_lag_{lag}' for lag in LAG_WEEKS] + [f'price_roll_mean_{w}' for w in ROLL_WINS] +
    [f'price_roll_std_{w}' for w in ROLL_WINS] +
    ['price_yoy', 'sin_week', 'cos_week', 'sin2_week', 'cos2_week', 'week_num', 'year_trend',
     'market_enc', 'state_enc', 'season_peak_arrival', 'season_lean', 'season_kharif',
     'season_rabi_arrival', 'season_harvest', 'season_storage']
)
ARR_FEATS = ['log_arr'] + [f'arr_lag_{lag}' for lag in [1, 2, 4]] + [f'arr_roll_mean_{w}' for w in [4, 8]]
M6_FEATS = PRICE_FEATS + ARR_FEATS + macro_cols + climate_feats + sat_feats + infra_feats + policy_feats
M7_FEATS = M6_FEATS + drought_feats
M8_FEATS = M6_FEATS + fert_feats

print(f'  M6 base features: {len(M6_FEATS)}  |  M7 adds {len(drought_feats)}  |  M8 adds {len(fert_feats)}')

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN + SHAP PER (crop, horizon), FOR BOTH M7 AND M8
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Training full-history models + computing SHAP ...\n')
shap_rows = []

for variant, feat_list_all, new_feats in [('M7', M7_FEATS, drought_feats), ('M8', M8_FEATS, fert_feats)]:
    for crop in CROPS:
        df_crop = feat[crop]
        fcols = [c for c in feat_list_all if c in df_crop.columns]
        for h in HORIZONS:
            df_h = df_crop.copy()
            df_h['target'] = df_h.groupby('market_id')['log_price'].shift(-h)
            df_h = df_h.dropna(subset=[c for c in ['target', 'price_lag_1'] if c in df_h.columns])

            max_date = df_h['week_start'].max()
            val_cutoff = max_date - pd.Timedelta(weeks=VAL_WEEKS)
            train = df_h[df_h['week_start'] <= val_cutoff]
            val = df_h[df_h['week_start'] > val_cutoff]

            X_tr, y_tr = train[fcols].fillna(0), train['target']
            X_va, y_va = val[fcols].fillna(0), val['target']

            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            best_iter = model.best_iteration_ or LGBM_PARAMS['n_estimators']

            final_params = dict(LGBM_PARAMS)
            final_params['n_estimators'] = best_iter
            final_model = lgb.LGBMRegressor(**final_params)
            X_all, y_all = df_h[fcols].fillna(0), df_h['target']
            final_model.fit(X_all, y_all)

            # SHAP on a sample of real rows (not synthetic), matching Script 25's convention
            sample = df_h[fcols].fillna(0).sample(min(SHAP_SAMPLE, len(df_h)), random_state=SEED)
            explainer = shap.TreeExplainer(final_model)
            shap_vals = explainer.shap_values(sample)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            mean_signed_shap = shap_vals.mean(axis=0)
            rank = pd.Series(mean_abs_shap, index=fcols).rank(ascending=False)

            # Also grab LightGBM's own split-count importance as a second, cheaper signal
            split_importance = dict(zip(fcols, final_model.feature_importances_))

            for nf in new_feats:
                if nf not in fcols:
                    continue
                idx = fcols.index(nf)
                shap_rows.append({
                    'variant': variant, 'crop': crop, 'horizon_weeks': h, 'feature': nf,
                    'mean_abs_shap': round(float(mean_abs_shap[idx]), 6),
                    'mean_signed_shap': round(float(mean_signed_shap[idx]), 6),
                    'shap_rank_of_total': int(rank.iloc[idx]),
                    'n_features_total': len(fcols),
                    'split_count': int(split_importance[nf]),
                })
            print(f'  {variant} {crop:7s} h={h:>2}w done (n_features={len(fcols)}, best_iter={best_iter})')

shap_df = pd.DataFrame(shap_rows)
shap_path = os.path.join(OUT_DIR, 'table_shap_m7_m8_layer_features.csv')
shap_df.to_csv(shap_path, index=False, encoding='utf-8')
print(f'\n[4] Saved: {shap_path}  ({len(shap_df)} rows)')

print('\n  Summary: median SHAP rank (of ~total features) per variant/feature, across all 12 crop x horizon cells\n')
for variant in ['M7', 'M8']:
    vsub = shap_df[shap_df['variant'] == variant]
    for feature in vsub['feature'].unique():
        fsub = vsub[vsub['feature'] == feature]
        med_rank = fsub['shap_rank_of_total'].median()
        n_total = fsub['n_features_total'].iloc[0]
        med_split = fsub['split_count'].median()
        sign = 'mixed' if (fsub['mean_signed_shap'] > 0).any() and (fsub['mean_signed_shap'] < 0).any() else \
               ('positive' if (fsub['mean_signed_shap'] > 0).all() else 'negative')
        print(f'  {variant} {feature:28s}: median rank {med_rank:.0f}/{n_total}  '
              f'median split_count={med_split:.0f}  SHAP sign across cells: {sign}')

# ─────────────────────────────────────────────────────────────────────────────
# 5. REDUNDANCY CHECK
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Redundancy check: correlation of M7/M8 features against existing signal ...\n')
EXISTING_REF_COLS = [c for c in [
    'year_trend', 'wpi_fruits_vegetables', 'wpi_vegetables_total',
    'wpi_potato', 'wpi_onion', 'wpi_tomato', 'diesel_4city_rs_litre',
    'crude_oil_usd_bbl', 'agri_wages_rs_day', 'price_lag_52',
] if c in feat['tomato'].columns]

redun_rows = []
for crop in CROPS:
    sub = feat[crop]
    for nf in drought_feats + fert_feats:
        if nf not in sub.columns or sub[nf].dropna().nunique() < 2:
            continue
        for ref in EXISTING_REF_COLS:
            if ref not in sub.columns:
                continue
            pair = sub[[nf, ref]].dropna()
            if len(pair) < 30:
                continue
            corr = pair[nf].corr(pair[ref])
            redun_rows.append({'crop': crop, 'feature': nf, 'reference': ref,
                                'pearson_r': round(float(corr), 3), 'n': len(pair)})

redun_df = pd.DataFrame(redun_rows)
redun_path = os.path.join(OUT_DIR, 'table_fert_drought_redundancy.csv')
redun_df.to_csv(redun_path, index=False, encoding='utf-8')
print(f'  Saved: {redun_path}  ({len(redun_df)} rows)')

print('\n  Strongest redundancies found (|r| >= 0.5):')
strong = redun_df[redun_df['pearson_r'].abs() >= 0.5].sort_values('pearson_r', key=abs, ascending=False)
if strong.empty:
    print('    None -- no M7/M8 feature is strongly collinear with an existing macro/trend feature.')
else:
    for _, r in strong.iterrows():
        print(f'    {r["crop"]:8s} {r["feature"]:22s} vs {r["reference"]:24s}: r={r["pearson_r"]:+.3f}  (n={r["n"]})')

print('\n' + '=' * 65)
print('Script 61 complete.')
