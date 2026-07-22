# -*- coding: utf-8 -*-
"""
Script 15 — Ablation Study: M0 → M4
=====================================
Trains 5 progressively richer LightGBM variants on the same rolling-origin
CV framework as Script 12, adding one data layer at a time:

  M0  Price features only (lags, rolling stats, seasonality, market encoding)
  M1  + Arrivals (log arrivals, lags, rolling means)
  M2  + Macro-logistics (CMIE credit, RBI repo/WPI/USDINR, PPAC diesel/LPG)
  M3  + Climate stress (ERA5 temperature + CHIRPS rainfall, incl. 4/8-week rolls)
  M4  + Satellite vegetation (Sentinel-2 NDVI/EVI + MODIS NDVI/LST + rolling)

Each variant × 4 folds × 4 horizons × 3 crops = 240 LightGBM model fits.

Compare against B1 Naive Persistence from Script 13.

Outputs (Model_Output/)
-----------------------
  ablation_raw_results.csv          all 180 rows: variant × crop × fold × horizon
  ablation_predictions.csv          crop-level weekly avg y_true/y_pred per
                                     variant×crop×fold×horizon (for Script 18 DM tests)
  table_ablation.csv                paper Table — mean across folds
  fig_ablation_r2.png               R² by model variant (h=1, 4 panels)
  fig_ablation_improvement.png      delta R² over M0 by layer addition
  fig_ablation_heatmap.png          full heatmap of all metric × variant combinations

Run: python scripts/15_Ablation_Study_M0_M4.py
Estimated runtime: 60-120 min (early-stop at 50 rounds)
"""

import io, os, sys, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE     = r'C:\Users\masro\Documents\TOP_Digital_Twin'
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CMIE_FILE= os.path.join(BASE, 'data', 'cmie_macro',      'cmie_macro_2017_2025.csv')
RBI_FILE = os.path.join(BASE, 'data', 'rbi_dbie',        'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE= os.path.join(BASE, 'data', 'ppac_macro',      'ppac_diesel_lpg_2017_2025.csv')
SAT_FILE = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
BENCH_FILE = os.path.join(BASE, 'Model_Output', 'table_benchmarks.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED     = 42

# Set FAST_MODE = True to run h=1 and h=4 only with reduced trees (for testing)
FAST_MODE = False
HORIZONS_RUN = [1, 4] if FAST_MODE else HORIZONS

# Diagnostic: retrain only M0 and M4 (headline pair) on the FULL market
# panel and save per-market (not crop-averaged) predictions, so Script 18b
# can run a higher-power, market-level Diebold-Mariano test — checking
# whether crop-level weekly averaging was hiding a real per-market effect.
# Cheap: only 2 of 5 variants, so ~2/5 the runtime of a full ablation pass.
MARKET_LEVEL_DIAGNOSTIC = True

FOLDS = [
    {'fold': 1, 'train_end': '2021-06-30',
     'val_start': '2021-07-01', 'val_end': '2021-12-31',
     'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'fold': 2, 'train_end': '2022-06-30',
     'val_start': '2022-07-01', 'val_end': '2022-12-31',
     'test_start': '2023-01-01', 'test_end': '2023-12-31'},
    {'fold': 3, 'train_end': '2023-06-30',
     'val_start': '2023-07-01', 'val_end': '2023-12-31',
     'test_start': '2024-01-01', 'test_end': '2024-12-31'},
    {'fold': 4, 'train_end': '2024-06-30',
     'val_start': '2024-07-01', 'val_end': '2024-12-31',
     'test_start': '2025-01-01', 'test_end': '2025-12-31'},
]

LGBM_PARAMS = dict(
    objective        = 'regression',
    metric           = 'rmse',
    n_estimators     = 300 if FAST_MODE else 1000,
    learning_rate    = 0.05,
    num_leaves       = 127,
    max_depth        = -1,
    min_child_samples= 20,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    n_jobs           = -1,
    random_state     = SEED,
    verbose          = -1,
)

CROP_COLORS = {'tomato': '#E63946', 'onion': '#F4A261', 'potato': '#457B9D'}
VARIANT_COLORS = {
    'M0': '#adb5bd', 'M1': '#74c0fc', 'M2': '#51cf66',
    'M3': '#ff922b', 'M4': '#cc5de8'
}
VARIANT_LABELS = {
    'M0': 'M0 Price only',
    'M1': 'M1 + Arrivals',
    'M2': 'M2 + Macro',
    'M3': 'M3 + Climate',
    'M4': 'M4 + Satellite',
}

LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS  = [4, 8, 13]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD PANEL + MACRO
# ─────────────────────────────────────────────────────────────────────────────
print('='*65)
print('SCRIPT 15: ABLATION STUDY  M0 → M4')
print('='*65)
print(f'  Fast mode : {FAST_MODE}')
print(f'  Horizons  : {HORIZONS_RUN}')
print(f'  Total fits: {5 * len(FOLDS) * len(HORIZONS_RUN) * len(CROPS)}\n')

print('[1] Loading panel ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2025-12-31')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)
df['year']  = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month
print(f'   Panel: {len(df):,} rows')

# Macro join
macro_dfs = []
for fpath in [CMIE_FILE, RBI_FILE, PPAC_FILE]:
    if os.path.exists(fpath):
        m = pd.read_csv(fpath)
        macro_dfs.append(m)

MACRO_COLS = []
if macro_dfs:
    macro = macro_dfs[0]
    for m in macro_dfs[1:]:
        macro = macro.merge(m, on=['year', 'month'], how='outer',
                            suffixes=('', '_dup'))
        macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
    drop_cols = [c for c in ['date', 'date_x', 'date_y'] if c in macro.columns]
    df = df.merge(macro.drop(columns=drop_cols, errors='ignore'),
                  on=['year', 'month'], how='left')
    MACRO_COLS = [c for c in macro.columns if c not in ('date', 'year', 'month')]
    print(f'   Macro joined: {len(MACRO_COLS)} series → {MACRO_COLS}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. SATELLITE / CLIMATE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Loading satellite/climate features (Script 14 output) ...')

sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'])
sat = sat.sort_values(['crop', 'week_start']).reset_index(drop=True)

# Raw climate/satellite columns from Script 14
ERA5_COLS   = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr',
               'era5_heat_35', 'era5_heat_38']
CHIRPS_COLS = ['chirps_rain_mm', 'chirps_rain_max', 'chirps_excess']
S2_COLS     = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']
MODIS_COLS  = ['modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max',
               'modis_lst_frac35']

# Rolling aggregates (computed at crop × time level, shift(1) to avoid leakage)
roll_specs = [
    ('era5_heat_35',   'sum', [4, 8]),   # cumulative heat stress
    ('chirps_rain_mm', 'sum', [4, 8]),   # cumulative rainfall
    ('s2_ndvi',        'mean', [4]),     # vegetation momentum
    ('s2_ndvi_anom',   'mean', [4]),     # anomaly trend
    ('modis_lst_mean', 'mean', [4]),     # LST trend
]

roll_cols = []
for col, func, windows in roll_specs:
    if col not in sat.columns:
        continue
    for w in windows:
        new_col = f'{col}_roll{w}'
        if func == 'sum':
            sat[new_col] = (sat.groupby('crop')[col]
                               .transform(lambda x: x.shift(1).rolling(w, min_periods=2).sum()))
        else:
            sat[new_col] = (sat.groupby('crop')[col]
                               .transform(lambda x: x.shift(1).rolling(w, min_periods=2).mean()))
        roll_cols.append(new_col)

# Feature lists for each layer
CLIMATE_FEATS  = [c for c in ERA5_COLS + CHIRPS_COLS if c in sat.columns]
CLIMATE_FEATS += [c for c in roll_cols if any(s in c for s in ['era5_heat', 'chirps_rain'])]
SAT_FEATS      = [c for c in S2_COLS + MODIS_COLS if c in sat.columns]
SAT_FEATS     += [c for c in roll_cols if any(s in c for s in ['s2_', 'modis_'])]

# Join to panel
join_cols = ['week_start', 'crop'] + CLIMATE_FEATS + SAT_FEATS
df = df.merge(sat[join_cols], on=['crop', 'week_start'], how='left')
print(f'   Climate features  : {len(CLIMATE_FEATS)} → {CLIMATE_FEATS}')
print(f'   Satellite features: {len(SAT_FEATS)} → {SAT_FEATS}')
print(f'   Panel after join  : {len(df):,} rows  |  {df.shape[1]} columns')
print(f'   ERA5 coverage     : {df["era5_tmax"].notna().mean():.1%}')
print(f'   S2 NDVI coverage  : {df["s2_ndvi"].notna().mean():.1%}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Engineering features ...')

_EXCLUDE = {
    'crop', 'market', 'state', 'district', 'state_code', 'market_id',
    'arrivals_tonnes_week', 'modal_price_weighted', 'log_price', 'log_arr',
    'week_start', 'year_month', 'year', 'month', 'date',
    'next_price', 'target', 'spike', 'iso_year', 'iso_week', 'trading_days',
}


def build_features(df_in):
    """Build all features; returns per-crop dict of DataFrames."""
    out = {}
    for crop in CROPS:
        sub = df_in[df_in['crop'] == crop].copy()
        sub = sub.sort_values(['market', 'week_start'])

        sub['log_price'] = np.log1p(sub['modal_price_weighted'])

        # Price lags
        for lag in LAG_WEEKS:
            sub[f'price_lag_{lag}'] = sub.groupby('market')['log_price'].shift(lag)

        # Rolling price stats (shift-1 to avoid leakage)
        for w in ROLL_WINS:
            g = sub.groupby('market')['log_price']
            sub[f'price_roll_mean_{w}'] = g.transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).mean())
            sub[f'price_roll_std_{w}'] = g.transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).std())

        # Arrivals
        if 'arrivals_tonnes_week' in sub.columns:
            sub['log_arr'] = np.log1p(sub['arrivals_tonnes_week'].clip(lower=0))
            for lag in [1, 2, 4]:
                sub[f'arr_lag_{lag}'] = sub.groupby('market')['log_arr'].shift(lag)
            for w in [4, 8]:
                sub[f'arr_roll_mean_{w}'] = sub.groupby('market')['log_arr'].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=2).mean())

        # YoY price
        sub['price_yoy'] = sub.groupby('market')['log_price'].shift(52)

        # Sinusoidal seasonality
        sub['week_num'] = sub['week_start'].dt.isocalendar().week.astype(int)
        sub['sin_week'] = np.sin(2 * np.pi * sub['week_num'] / 52)
        sub['cos_week'] = np.cos(2 * np.pi * sub['week_num'] / 52)
        sub['sin2_week'] = np.sin(4 * np.pi * sub['week_num'] / 52)
        sub['cos2_week'] = np.cos(4 * np.pi * sub['week_num'] / 52)

        # Crop-specific agronomic season flags
        m = sub['week_start'].dt.month
        if crop == 'tomato':
            sub['season_peak_arrival'] = m.isin([11, 12, 1, 2]).astype(int)   # Rabi harvest; max supply, lowest prices
            sub['season_lean']         = m.isin([5, 6, 7]).astype(int)         # inter-crop gap; price spike window
            sub['season_kharif']       = m.isin([8, 9, 10]).astype(int)        # rainy-season crop; reduced quality
        elif crop == 'onion':
            sub['season_rabi_arrival'] = m.isin([2, 3, 4, 5]).astype(int)     # peak rabi supply; prices lowest
            sub['season_lean']         = m.isin([9, 10, 11]).astype(int)       # rabi storage depleted; crisis window
            sub['season_kharif']       = m.isin([8, 9]).astype(int)            # small kharif crop arrives
        elif crop == 'potato':
            sub['season_harvest']      = m.isin([2, 3, 4]).astype(int)         # rabi harvest; peak arrivals, low prices
            sub['season_storage']      = m.isin([5, 6, 7, 8, 9]).astype(int)  # cold-storage release sustains supply
            sub['season_lean']         = m.isin([10, 11]).astype(int)          # storage tail; highest price risk

        # Market / state encodings
        for col in ['state', 'market']:
            if col in sub.columns:
                sub[f'{col}_enc'] = pd.Categorical(sub[col]).codes

        # Time trend
        sub['year_trend'] = sub['week_start'].dt.year - 2017

        out[crop] = sub
    return out


feat = build_features(df)

# Define feature groups (built column names must exist in at least one crop's df)
PRICE_FEATS = (
    [f'price_lag_{lag}' for lag in LAG_WEEKS] +
    [f'price_roll_mean_{w}' for w in ROLL_WINS] +
    [f'price_roll_std_{w}' for w in ROLL_WINS] +
    ['price_yoy', 'sin_week', 'cos_week', 'sin2_week', 'cos2_week',
     'week_num', 'year_trend', 'market_enc', 'state_enc',
     # crop-specific agronomic season flags (crop-conditional; absent columns ignored at fit time)
     'season_peak_arrival', 'season_lean', 'season_kharif',
     'season_rabi_arrival', 'season_harvest', 'season_storage']
)

ARR_FEATS = (
    ['log_arr'] +
    [f'arr_lag_{lag}' for lag in [1, 2, 4]] +
    [f'arr_roll_mean_{w}' for w in [4, 8]]
)

MODEL_FEATURE_SETS = {
    'M0': PRICE_FEATS,
    'M1': PRICE_FEATS + ARR_FEATS,
    'M2': PRICE_FEATS + ARR_FEATS + MACRO_COLS,
    'M3': PRICE_FEATS + ARR_FEATS + MACRO_COLS + CLIMATE_FEATS,
    'M4': PRICE_FEATS + ARR_FEATS + MACRO_COLS + CLIMATE_FEATS + SAT_FEATS,
}
if MARKET_LEVEL_DIAGNOSTIC:
    MODEL_FEATURE_SETS = {'M0': MODEL_FEATURE_SETS['M0'],
                           'M4': MODEL_FEATURE_SETS['M4']}

for crop in CROPS:
    df_crop = feat[crop]
    all_possible = MODEL_FEATURE_SETS['M4']
    available = [c for c in all_possible if c in df_crop.columns]
    print(f'   {crop:8s}: {len(df_crop):>8,} rows  | '
          f'M4 features available: {len(available)}/{len(all_possible)}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask   = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, R2=np.nan, N=0)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(RMSE=round(rmse, 1), MAE=round(mae, 1),
                MAPE=round(mape, 2), R2=round(r2, 4), N=len(yt))


# ─────────────────────────────────────────────────────────────────────────────
# 6. ABLATION TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Running ablation: M0 → M4 ...')
print(f'    {len(MODEL_FEATURE_SETS)} variants × {len(FOLDS)} folds × '
      f'{len(HORIZONS_RUN)} horizons × {len(CROPS)} crops = '
      f'{len(MODEL_FEATURE_SETS)*len(FOLDS)*len(HORIZONS_RUN)*len(CROPS)} model fits\n')

all_rows = []
all_pred_rows = []   # crop-level weekly avg predictions, for Script 18 DM tests
t0_total = time.time()

for variant, feat_list_all in MODEL_FEATURE_SETS.items():
    print(f'\n  ── {variant}: {VARIANT_LABELS[variant]} '
          f'(up to {len(feat_list_all)} features) ──')
    v_t0 = time.time()

    for crop in CROPS:
        df_crop = feat[crop].copy()
        # Only keep columns that actually exist in this crop's dataframe
        fcols = [c for c in feat_list_all if c in df_crop.columns]

        for fold_info in FOLDS:
            fold     = fold_info['fold']
            t_end    = pd.Timestamp(fold_info['train_end'])
            v_start  = pd.Timestamp(fold_info['val_start'])
            v_end    = pd.Timestamp(fold_info['val_end'])
            te_start = pd.Timestamp(fold_info['test_start'])
            te_end   = pd.Timestamp(fold_info['test_end'])

            for h in HORIZONS_RUN:
                t0 = time.time()

                df_h = df_crop.copy()
                df_h['target'] = df_h.groupby('market')['log_price'].shift(-h)

                # Filter: must have target and enough price lags
                required = ['target', 'price_lag_1']
                df_h = df_h.dropna(subset=[c for c in required if c in df_h.columns])

                train = df_h[df_h['week_start'] <= t_end]
                val   = df_h[(df_h['week_start'] > v_start) & (df_h['week_start'] <= v_end)]
                test  = df_h[(df_h['week_start'] >= te_start) & (df_h['week_start'] <= te_end)]

                if len(train) < 100 or len(test) < 10:
                    continue

                X_tr = train[fcols].fillna(0)
                y_tr = train['target']
                X_va = val[fcols].fillna(0)
                y_va = val['target']
                X_te = test[fcols].fillna(0)
                y_te = test['target']

                model = lgb.LGBMRegressor(**LGBM_PARAMS)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)]
                )

                y_pred  = model.predict(X_te)
                m       = compute_metrics(y_te.values, y_pred)
                elapsed = round(time.time() - t0, 1)
                trees   = model.best_iteration_ if model.best_iteration_ else LGBM_PARAMS['n_estimators']

                pred_df = pd.DataFrame({
                    'market':     test['market'].values,
                    'week_start': test['week_start'].values,
                    'y_true': np.expm1(y_te.values),
                    'y_pred': np.expm1(y_pred),
                })

                if MARKET_LEVEL_DIAGNOSTIC:
                    # Full per-market rows — only 2 variants (M0, M4), so
                    # this stays a manageable file size
                    mkt = pred_df.copy()
                    mkt['variant']       = variant
                    mkt['crop']          = crop
                    mkt['fold']          = fold
                    mkt['horizon_weeks'] = h
                    all_pred_rows.append(mkt)
                else:
                    # Crop-level weekly aggregate predictions (avg across
                    # markets) for Script 18 DM tests — market-level rows
                    # would produce tens of millions of rows across all fits
                    weekly = pred_df.groupby('week_start').agg(
                        y_true=('y_true', 'mean'),
                        y_pred=('y_pred', 'mean'),
                        n_markets=('y_true', 'size'),
                    ).reset_index()
                    weekly['variant']       = variant
                    weekly['crop']          = crop
                    weekly['fold']          = fold
                    weekly['horizon_weeks'] = h
                    all_pred_rows.append(weekly)

                row = {
                    'variant': variant, 'crop': crop,
                    'fold': fold, 'horizon_weeks': h,
                    'test_year': te_end.year,
                    'n_features': len(fcols),
                    'n_trees': trees,
                    **m, 'fit_sec': elapsed
                }
                all_rows.append(row)

                flag = ''
                if m['R2'] is not None and not np.isnan(m['R2']):
                    if m['R2'] < 0:
                        flag = ' ← NEGATIVE R²'

                print(f'   {variant} {crop:7s} fold{fold} h={h:>2}w | '
                      f'RMSE={m["RMSE"]:>7.1f}  MAPE={m["MAPE"]:>5.1f}%  '
                      f'R²={m["R2"]:>6.3f}  trees={trees:>4}  [{elapsed}s]{flag}')

    v_elapsed = (time.time() - v_t0) / 60
    print(f'  {variant} done in {v_elapsed:.1f} min')

total_min = (time.time() - t0_total) / 60
print(f'\n  Total ablation time: {total_min:.1f} min')

results = pd.DataFrame(all_rows)

if MARKET_LEVEL_DIAGNOSTIC:
    # Diagnostic-only filenames — never touch the production M0-M4
    # ablation_raw_results.csv / ablation_predictions.csv that Script 18
    # and the paper tables depend on
    diag_raw_path = os.path.join(OUT_DIR, 'dm_market_level_raw_results.csv')
    results.to_csv(diag_raw_path, index=False)
    print(f'  Saved: {diag_raw_path}')

    predictions = pd.concat(all_pred_rows, ignore_index=True)
    predictions = predictions[['variant', 'crop', 'fold', 'horizon_weeks',
                                'market', 'week_start', 'y_true', 'y_pred']]
    diag_pred_path = os.path.join(OUT_DIR, 'dm_market_level_predictions.csv')
    predictions.to_csv(diag_pred_path, index=False)
    print(f'  Saved: {diag_pred_path}  ({len(predictions):,} rows)')
    print('\nMARKET_LEVEL_DIAGNOSTIC run complete — skipping summary table/figures')
    print('(those assume the full M0-M4 + B1_Naive production run).')
    sys.exit(0)

raw_path = os.path.join(OUT_DIR, 'ablation_raw_results.csv')
results.to_csv(raw_path, index=False)
print(f'  Saved: {raw_path}')

predictions = pd.concat(all_pred_rows, ignore_index=True)
predictions = predictions[['variant', 'crop', 'fold', 'horizon_weeks',
                            'week_start', 'y_true', 'y_pred', 'n_markets']]
pred_path = os.path.join(OUT_DIR, 'ablation_predictions.csv')
predictions.to_csv(pred_path, index=False)
print(f'  Saved: {pred_path}  ({len(predictions):,} rows)')


# ─────────────────────────────────────────────────────────────────────────────
# 7. ABLATION SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Building ablation summary table ...')

summary = (results
           .groupby(['variant', 'crop', 'horizon_weeks'])[['RMSE', 'MAE', 'MAPE', 'R2']]
           .mean()
           .round({'RMSE': 1, 'MAE': 1, 'MAPE': 2, 'R2': 4})
           .reset_index())

# Load Naive Persistence from Script 13
VARIANTS_ALL = list(MODEL_FEATURE_SETS.keys())
if os.path.exists(BENCH_FILE):
    bench = pd.read_csv(BENCH_FILE)
    naive = (bench[bench['model'] == 'B1_Naive']
             .groupby(['crop', 'horizon_weeks'])[['RMSE', 'MAE', 'MAPE', 'R2']]
             .mean()
             .round({'RMSE': 1, 'MAE': 1, 'MAPE': 2, 'R2': 4})
             .reset_index())
    naive['variant'] = 'B1_Naive'
    summary = pd.concat([summary, naive], ignore_index=True)
    VARIANTS_ALL = list(MODEL_FEATURE_SETS.keys()) + ['B1_Naive']
    print('   B1_Naive benchmark appended from table_benchmarks.csv')
else:
    print('   WARNING: table_benchmarks.csv not found — omitting benchmark row')

# Print console table
print()
for crop in CROPS:
    print(f'  {crop.upper()}')
    hdr = f"  {'Variant':<12}"
    for h in HORIZONS_RUN:
        hdr += f"  h={h}w RMSE   MAPE    R²  "
    print(hdr)
    print('  ' + '-' * (12 + len(HORIZONS_RUN) * 26))
    for v in VARIANTS_ALL:
        row_str = f'  {v:<12}'
        for h in HORIZONS_RUN:
            sub = summary[(summary['variant'] == v) & (summary['crop'] == crop)
                          & (summary['horizon_weeks'] == h)]
            if sub.empty:
                row_str += f"  {'—':>8} {'—':>6} {'—':>6}  "
            else:
                r = sub.iloc[0]
                row_str += f"  {r.RMSE:>8.1f} {r.MAPE:>5.1f}% {r.R2:>6.3f}  "
        print(row_str)
    print()

table_path = os.path.join(OUT_DIR, 'table_ablation.csv')
summary.to_csv(table_path, index=False)
print(f'  Saved: {table_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6] Generating figures ...')

VARIANTS_PLOT = list(MODEL_FEATURE_SETS.keys())   # M0-M4 only (no naive in comparison figs)

# ── Figure A: R² by variant × crop (h=1 and h=4, side-by-side panels) ────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey='row')

for row_i, h in enumerate([1, 4]):
    for col_i, crop in enumerate(CROPS):
        ax = axes[row_i][col_i]
        sub = summary[(summary['crop'] == crop) & (summary['horizon_weeks'] == h)]

        x = np.arange(len(VARIANTS_PLOT))
        vals = []
        for v in VARIANTS_PLOT:
            rs = sub[sub['variant'] == v]['R2']
            vals.append(rs.values[0] if not rs.empty else np.nan)

        colors = [VARIANT_COLORS.get(v, '#888') for v in VARIANTS_PLOT]
        bars = ax.bar(x, vals, color=colors, alpha=0.87, edgecolor='white', width=0.65)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ypos = max(val, 0) + 0.015
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Naive Persistence reference line
        naive_r2 = sub[sub['variant'] == 'B1_Naive']['R2']
        if not naive_r2.empty:
            ax.axhline(naive_r2.values[0], color='#555', linestyle='--',
                       linewidth=1.2, label=f'Naive (R²={naive_r2.values[0]:.3f})')
            ax.legend(fontsize=7, loc='lower right')

        ax.axhline(0, color='black', linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANTS_PLOT, fontsize=9)
        ax.set_ylim(-0.2, 1.02)
        if col_i == 0:
            ax.set_ylabel(f'R²  (h={h}w)', fontsize=10)
        if row_i == 0:
            ax.set_title(crop.capitalize(), fontsize=11, fontweight='bold',
                         color=CROP_COLORS[crop])
        ax.grid(axis='y', alpha=0.3)

plt.suptitle('Ablation Study: R² by Data Layer Addition (LightGBM, mean across 3 folds)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_ablation_r2.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ── Figure B: Delta R² over M0 (improvement from adding each layer) ───────────
fig, axes = plt.subplots(1, len(HORIZONS_RUN), figsize=(5 * len(HORIZONS_RUN), 6), sharey=False)
if len(HORIZONS_RUN) == 1:
    axes = [axes]

for ax, h in zip(axes, HORIZONS_RUN):
    for crop in CROPS:
        sub = summary[(summary['crop'] == crop) & (summary['horizon_weeks'] == h)]
        m0_r2 = sub[sub['variant'] == 'M0']['R2']
        if m0_r2.empty:
            continue
        base = m0_r2.values[0]

        delta_vals, delta_labels = [], []
        for v in VARIANTS_PLOT[1:]:    # M1 through M4
            rs = sub[sub['variant'] == v]['R2']
            if not rs.empty:
                delta_vals.append(rs.values[0] - base)
                delta_labels.append(v)

        x = np.arange(len(delta_labels))
        bars = ax.bar(x + CROPS.index(crop) * 0.25 - 0.25,
                      delta_vals, width=0.22, alpha=0.85,
                      label=crop.capitalize(), color=CROP_COLORS[crop],
                      edgecolor='white')

    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks(np.arange(len(VARIANTS_PLOT) - 1))
    ax.set_xticklabels([f'M0→{v}' for v in VARIANTS_PLOT[1:]], fontsize=9)
    ax.set_ylabel('ΔR² over M0 (price-only)', fontsize=10)
    ax.set_title(f'h = {h} week(s)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Ablation Study: Incremental R² Gain from Each Data Layer (ΔR² over M0)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_ablation_improvement.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ── Figure C: Full RMSE heatmap — variant × (crop × horizon) ──────────────────
pivot_rows = []
for v in VARIANTS_PLOT:
    for crop in CROPS:
        for h in HORIZONS_RUN:
            sub = summary[(summary['variant'] == v) & (summary['crop'] == crop)
                          & (summary['horizon_weeks'] == h)]
            val = sub['RMSE'].values[0] if not sub.empty else np.nan
            pivot_rows.append({'variant': v, 'crop': crop, 'horizon_weeks': h, 'RMSE': val})
piv_df = pd.DataFrame(pivot_rows)
piv_df['col_label'] = piv_df['crop'].str.capitalize() + ' h=' + piv_df['horizon_weeks'].astype(str) + 'w'

pivot = piv_df.pivot_table(index='variant', columns='col_label', values='RMSE')
col_order = []
for crop in CROPS:
    for h in HORIZONS_RUN:
        col_order.append(f'{crop.capitalize()} h={h}w')
pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.4), 5))
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd_r')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in pivot.index], fontsize=9)
plt.colorbar(im, ax=ax, label='RMSE (Rs/quintal)', fraction=0.04, pad=0.02)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=8, color='black' if val < pivot.values.max() * 0.7 else 'white')
ax.set_title('RMSE Heatmap: Model Variant × Crop × Horizon (Rs/quintal)\n(Lower is better — mean across 3 folds)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_ablation_heatmap.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ── Figure D: MAPE line chart across variants (h=1, all crops) ────────────────
fig, axes = plt.subplots(1, len(HORIZONS_RUN), figsize=(6 * len(HORIZONS_RUN), 5))
if len(HORIZONS_RUN) == 1:
    axes = [axes]

for ax, h in zip(axes, HORIZONS_RUN):
    for crop in CROPS:
        sub = summary[(summary['crop'] == crop) & (summary['horizon_weeks'] == h)]
        x_labels, y_vals = [], []
        for v in VARIANTS_PLOT:
            rs = sub[sub['variant'] == v]['MAPE']
            if not rs.empty:
                x_labels.append(v)
                y_vals.append(rs.values[0])
        ax.plot(x_labels, y_vals, 'o-', color=CROP_COLORS[crop],
                lw=2, ms=7, label=crop.capitalize())

        # Add Naive reference point
        naive_mape = sub[sub['variant'] == 'B1_Naive']['MAPE']
        if not naive_mape.empty:
            ax.axhline(naive_mape.values[0], color=CROP_COLORS[crop],
                       linestyle=':', linewidth=1, alpha=0.7)

    ax.set_xlabel('Model Variant', fontsize=10)
    ax.set_ylabel('MAPE (%) — mean across folds', fontsize=10)
    ax.set_title(f'h = {h} week(s)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Ablation Study: MAPE Trend by Data Layer Addition\n(Dotted line = Naive Persistence reference)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_ablation_mape.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# 9. PAPER-READY TABLE (LaTeX-style format)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[7] Paper-ready ablation table (h=1 and h=4) ...')
print()

VARIANTS_PAPER = VARIANTS_PLOT + (['B1_Naive'] if 'B1_Naive' in summary['variant'].values else [])
HORIZONS_PAPER = [h for h in [1, 4] if h in HORIZONS_RUN]

for crop in CROPS:
    print(f'  {crop.upper()}')
    print(f"  {'Model':<24}  " +
          "  ".join([f'h={h}w RMSE  MAPE    R²' for h in HORIZONS_PAPER]))
    print('  ' + '─' * (24 + len(HORIZONS_PAPER) * 26))
    for v in VARIANTS_PAPER:
        row_str = f'  {VARIANT_LABELS.get(v, v):<24}  '
        for h in HORIZONS_PAPER:
            sub = summary[(summary['variant'] == v) & (summary['crop'] == crop)
                          & (summary['horizon_weeks'] == h)]
            if sub.empty:
                row_str += f"{'—':>10} {'—':>6} {'—':>7}  "
            else:
                r = sub.iloc[0]
                row_str += f"{r.RMSE:>10.1f} {r.MAPE:>5.1f}% {r.R2:>7.4f}  "
        print(row_str)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10. DONE
# ─────────────────────────────────────────────────────────────────────────────
print('='*65)
print(f'Script 15 complete. Total elapsed: {total_min:.1f} min')
print()
print('Key outputs:')
for fname in ['ablation_raw_results.csv', 'table_ablation.csv',
              'fig_ablation_r2.png', 'fig_ablation_improvement.png',
              'fig_ablation_heatmap.png', 'fig_ablation_mape.png']:
    fp = os.path.join(OUT_DIR, fname)
    if os.path.exists(fp):
        print(f'  {fname:<42} {os.path.getsize(fp)/1024:>7.1f} KB')
print()
print('Next: Script 16 — Temporal Fusion Transformer (TFT) model')
