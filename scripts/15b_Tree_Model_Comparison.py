# -*- coding: utf-8 -*-
"""
Script 15b — Tree-Based Model Comparison at M6 (Full Feature Set)
====================================================================
Script 15's M0-M6 ablation always uses LightGBM — it ablates FEATURE
LAYERS, not model choice. This script asks the orthogonal question:
at the full M6 feature set, does the choice of tree-based algorithm
matter? Compares LightGBM (refit here for an exact apples-to-apples
run, not reused from Script 15's numbers), XGBoost, CatBoost, and
RandomForest — same rolling-origin CV folds, horizons, and crops as
Script 15, M6 feature set only (not a full re-ablation across M0-M6
per model — see the project's memory for why that narrower scope was
chosen: it answers "does model choice matter" without ~4x the runtime
of redoing the whole heterogeneity study per model).

4 models x 4 folds x 4 horizons x 3 crops = 192 fits.

Outputs (Model_Output/):
  table_tree_model_comparison.csv   all 192 rows: model x crop x fold x horizon
  table_tree_model_comparison_mean.csv   mean across folds, paper-ready table
  fig_tree_model_comparison.png     R2/MAPE by model, per crop x horizon

Run: python scripts/15b_Tree_Model_Comparison.py
Estimated runtime: 20-40 min
"""

import io, os, sys, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG (identical join to Script 15 — see that script for the
# per-source rationale; duplicated here rather than imported since Script 15
# executes its own ablation on import, same pattern as Scripts 15/23/25)
# ─────────────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CMIE_FILE= os.path.join(BASE, 'data', 'cmie_macro',      'cmie_macro_2017_2025.csv')
RBI_FILE = os.path.join(BASE, 'data', 'rbi_dbie',        'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE= os.path.join(BASE, 'data', 'ppac_macro',      'ppac_diesel_lpg_2017_2025.csv')
SAT_FILE = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
WAGE_FILE  = os.path.join(BASE, 'data', 'labour_wages',   'wage_agri_state_monthly.csv')
COLD_FILE  = os.path.join(BASE, 'data', 'infrastructure', 'cold_storage_by_state.csv')
ROAD_FILE  = os.path.join(BASE, 'data', 'infrastructure', 'road_density_state_annual.csv')
POLICY_FILE= os.path.join(BASE, 'data', 'policy_trade',   'policy_weekly_features.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED     = 42
LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS = [4, 8, 13]

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

MODEL_COLORS = {'LightGBM': '#e64980', 'XGBoost': '#5c7cfa',
                'CatBoost': '#f59f00', 'RandomForest': '#37b24d'}

print('=' * 65)
print('SCRIPT 15b: TREE-MODEL COMPARISON AT M6')
print('=' * 65)
print(f'  4 models x {len(FOLDS)} folds x {len(HORIZONS)} horizons x '
      f'{len(CROPS)} crops = {4*len(FOLDS)*len(HORIZONS)*len(CROPS)} fits\n')

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD + JOIN ALL LAYERS (identical to Script 15)
# ─────────────────────────────────────────────────────────────────────────────
print('[1] Loading panel ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2026-07-27')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)
df['year']  = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month
print(f'   Panel: {len(df):,} rows')

macro_dfs = []
for fpath in [CMIE_FILE, RBI_FILE, PPAC_FILE]:
    if os.path.exists(fpath):
        macro_dfs.append(pd.read_csv(fpath))
MACRO_COLS = []
if macro_dfs:
    macro = macro_dfs[0]
    for m in macro_dfs[1:]:
        macro = macro.merge(m, on=['year', 'month'], how='outer', suffixes=('', '_dup'))
        macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
    drop_cols = [c for c in ['date', 'date_x', 'date_y'] if c in macro.columns]
    df = df.merge(macro.drop(columns=drop_cols, errors='ignore'), on=['year', 'month'], how='left')
    MACRO_COLS = [c for c in macro.columns if c not in ('date', 'year', 'month')]
print(f'   Macro joined: {len(MACRO_COLS)} series')

print('[2] Loading satellite/climate features ...')
sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'])
sat = sat.sort_values(['crop', 'week_start']).reset_index(drop=True)

ERA5_COLS   = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38']
CHIRPS_COLS = ['chirps_rain_mm', 'chirps_rain_max', 'chirps_excess']
S2_COLS     = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']
MODIS_COLS  = ['modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']

roll_specs = [
    ('era5_heat_35',   'sum', [4, 8]),
    ('chirps_rain_mm', 'sum', [4, 8]),
    ('s2_ndvi',        'mean', [4]),
    ('s2_ndvi_anom',   'mean', [4]),
    ('modis_lst_mean', 'mean', [4]),
]
roll_cols = []
for col, func, windows in roll_specs:
    if col not in sat.columns:
        continue
    for w in windows:
        new_col = f'{col}_roll{w}'
        agg = 'sum' if func == 'sum' else 'mean'
        sat[new_col] = (sat.groupby('crop')[col]
                           .transform(lambda x: getattr(x.shift(1).rolling(w, min_periods=2), agg)()))
        roll_cols.append(new_col)

CLIMATE_FEATS  = [c for c in ERA5_COLS + CHIRPS_COLS if c in sat.columns]
CLIMATE_FEATS += [c for c in roll_cols if any(s in c for s in ['era5_heat', 'chirps_rain'])]
SAT_FEATS      = [c for c in S2_COLS + MODIS_COLS if c in sat.columns]
SAT_FEATS     += [c for c in roll_cols if any(s in c for s in ['s2_', 'modis_'])]

join_cols = ['week_start', 'crop'] + CLIMATE_FEATS + SAT_FEATS
df = df.merge(sat[join_cols], on=['crop', 'week_start'], how='left')
print(f'   Climate features  : {len(CLIMATE_FEATS)}')
print(f'   Satellite features: {len(SAT_FEATS)}')

print('[2b] Loading infrastructure + policy/trade layers ...')


def assert_unique(frame, keys, label):
    n_dup = frame[keys].duplicated().sum()
    if n_dup:
        raise ValueError(f'{label}: {n_dup} duplicate rows on {keys}')


INFRA_FEATS, POLICY_FEATS = [], []

if os.path.exists(WAGE_FILE):
    wages = pd.read_csv(WAGE_FILE)[['state', 'year', 'month', 'wage_agri_men', 'wage_agri_women']]
    assert_unique(wages, ['state', 'year', 'month'], 'wages')
    n0 = len(df)
    df = df.merge(wages, on=['state', 'year', 'month'], how='left')
    assert len(df) == n0
    INFRA_FEATS += ['wage_agri_men', 'wage_agri_women']

if os.path.exists(COLD_FILE):
    cold = pd.read_csv(COLD_FILE)[['state', 'n_facilities', 'capacity_mt']]
    cold = cold.rename(columns={'n_facilities': 'cold_storage_n_facilities',
                                 'capacity_mt': 'cold_storage_capacity_mt'})
    assert_unique(cold, ['state'], 'cold storage')
    n0 = len(df)
    df = df.merge(cold, on=['state'], how='left')
    assert len(df) == n0
    INFRA_FEATS += ['cold_storage_n_facilities', 'cold_storage_capacity_mt']

if os.path.exists(ROAD_FILE):
    road = pd.read_csv(ROAD_FILE)[['state', 'year', 'road_density_per_100_sqkm']]
    assert_unique(road, ['state', 'year'], 'road density')
    n0 = len(df)
    df = df.merge(road, on=['state', 'year'], how='left')
    assert len(df) == n0
    INFRA_FEATS += ['road_density_per_100_sqkm']

if os.path.exists(POLICY_FILE):
    policy = pd.read_csv(POLICY_FILE, parse_dates=['week_start'])
    policy_cols = ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct',
                   'market_intervention_flag', 'operation_greens_active']
    policy = policy[['crop', 'week_start'] + policy_cols]
    assert_unique(policy, ['crop', 'week_start'], 'policy')
    n0 = len(df)
    df = df.merge(policy, on=['crop', 'week_start'], how='left')
    assert len(df) == n0
    POLICY_FEATS += policy_cols

print(f'   Infrastructure features: {len(INFRA_FEATS)}')
print(f'   Policy features        : {len(POLICY_FEATS)}')

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING (identical to Script 15)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Engineering features ...')


def build_features(df_in):
    # FIXED 2026-08-14 (full-layer audit): grouping/sorting used to key on
    # 'market' NAME, not market_id -- see Script 23/15's commits for the
    # full discovery story (a few market names repeat across different
    # states, e.g. "Fatehabad APMC" in both Haryana and Uttar Pradesh;
    # grouping by name interleaves two different markets' series together).
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

        if 'market_id' in sub.columns:
            sub['market_enc'] = pd.Categorical(sub['market_id']).codes
        for col in ['state']:
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
    ['log_arr'] + [f'arr_lag_{lag}' for lag in [1, 2, 4]] + [f'arr_roll_mean_{w}' for w in [4, 8]]
)
M6_FEATS = PRICE_FEATS + ARR_FEATS + MACRO_COLS + CLIMATE_FEATS + SAT_FEATS + INFRA_FEATS + POLICY_FEATS

for crop in CROPS:
    available = [c for c in M6_FEATS if c in feat[crop].columns]
    print(f'   {crop:8s}: {len(feat[crop]):>8,} rows  |  M6 features available: {len(available)}/{len(M6_FEATS)}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, R2=np.nan, N=0)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(RMSE=round(rmse, 1), MAE=round(mae, 1), MAPE=round(mape, 2), R2=round(r2, 4), N=len(yt))


def fit_predict(model_name, X_tr, y_tr, X_va, y_va, X_te):
    if model_name == 'LightGBM':
        m = lgb.LGBMRegressor(
            objective='regression', metric='rmse', n_estimators=1000, learning_rate=0.05,
            num_leaves=127, min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=SEED, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        trees = m.best_iteration_ or 1000
    elif model_name == 'XGBoost':
        m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
            random_state=SEED, early_stopping_rounds=50, eval_metric='rmse', verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        trees = m.best_iteration or 1000
    elif model_name == 'CatBoost':
        m = cb.CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
            random_state=SEED, early_stopping_rounds=50, verbose=False, thread_count=-1)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        trees = m.get_best_iteration() or 1000
    elif model_name == 'RandomForest':
        # No native early-stopping concept, so instead we grow the forest
        # incrementally via warm_start and pick whichever n_estimators
        # checkpoint scores best on the validation split -- same spirit as
        # early stopping for the boosted models above. warm_start means each
        # step only builds the *new* trees, so the total work across the grid
        # is ~800 trees (vs. the old fixed 300), a small constant factor.
        n_estimators_grid = [100, 300, 500, 800]
        m = RandomForestRegressor(
            n_estimators=n_estimators_grid[0], max_depth=20, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED, warm_start=True)
        best_rmse, best_n, best_estimators = np.inf, n_estimators_grid[0], None
        for n in n_estimators_grid:
            m.n_estimators = n
            m.fit(X_tr, y_tr)
            va_rmse = np.sqrt(mean_squared_error(y_va, m.predict(X_va)))
            if va_rmse < best_rmse:
                best_rmse, best_n, best_estimators = va_rmse, n, list(m.estimators_)
        m.estimators_ = best_estimators
        trees = best_n
    else:
        raise ValueError(model_name)
    return m.predict(X_te), trees


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPARISON LOOP
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Running tree-model comparison at M6 ...')
MODELS = ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest']
all_rows = []
t0_total = time.time()

for model_name in MODELS:
    print(f'\n  == {model_name} ==')
    m_t0 = time.time()
    for crop in CROPS:
        df_crop = feat[crop].copy()
        fcols = [c for c in M6_FEATS if c in df_crop.columns]

        for fold_info in FOLDS:
            fold = fold_info['fold']
            t_end = pd.Timestamp(fold_info['train_end'])
            v_start = pd.Timestamp(fold_info['val_start'])
            v_end = pd.Timestamp(fold_info['val_end'])
            te_start = pd.Timestamp(fold_info['test_start'])
            te_end = pd.Timestamp(fold_info['test_end'])

            for h in HORIZONS:
                t0 = time.time()
                df_h = df_crop.copy()
                df_h['target'] = df_h.groupby('market_id')['log_price'].shift(-h)
                required = ['target', 'price_lag_1']
                df_h = df_h.dropna(subset=[c for c in required if c in df_h.columns])

                train = df_h[df_h['week_start'] <= t_end]
                val = df_h[(df_h['week_start'] > v_start) & (df_h['week_start'] <= v_end)]
                test = df_h[(df_h['week_start'] >= te_start) & (df_h['week_start'] <= te_end)]

                if len(train) < 100 or len(test) < 10:
                    continue

                X_tr, y_tr = train[fcols].fillna(0), train['target']
                X_va, y_va = val[fcols].fillna(0), val['target']
                X_te, y_te = test[fcols].fillna(0), test['target']

                y_pred, trees = fit_predict(model_name, X_tr, y_tr, X_va, y_va, X_te)
                m = compute_metrics(y_te.values, y_pred)
                elapsed = round(time.time() - t0, 1)

                all_rows.append(dict(model=model_name, crop=crop, fold=fold, horizon_weeks=h,
                                      trees=trees, elapsed_s=elapsed,
                                      n_train=len(train), n_test=len(test), **m))
                print(f'    {crop:8s} fold{fold} h={h:>2d}w  '
                      f'RMSE={m["RMSE"]:>8.1f}  MAPE={m["MAPE"]:>6.2f}%  R2={m["R2"]:>7.4f}  '
                      f'({elapsed:.1f}s)')
    print(f'  {model_name} total time: {(time.time()-m_t0)/60:.1f} min')

print(f'\nTotal runtime: {(time.time()-t0_total)/60:.1f} min')

results = pd.DataFrame(all_rows)
results.to_csv(os.path.join(OUT_DIR, 'table_tree_model_comparison.csv'), index=False)

mean_tbl = (results.groupby(['model', 'crop', 'horizon_weeks'])[['RMSE', 'MAE', 'MAPE', 'R2']]
            .mean().round(3).reset_index())
mean_tbl.to_csv(os.path.join(OUT_DIR, 'table_tree_model_comparison_mean.csv'), index=False)

print('\n=== Mean R2 by model x crop x horizon ===')
print(mean_tbl.pivot_table(index=['crop', 'horizon_weeks'], columns='model', values='R2').to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Generating comparison figure ...')
fig, axes = plt.subplots(len(CROPS), 2, figsize=(13, 11))
for i, crop in enumerate(CROPS):
    sub = mean_tbl[mean_tbl['crop'] == crop]
    ax_r2, ax_mape = axes[i, 0], axes[i, 1]
    for model_name in MODELS:
        s = sub[sub['model'] == model_name].sort_values('horizon_weeks')
        if s.empty:
            continue
        ax_r2.plot(s['horizon_weeks'], s['R2'], 'o-', label=model_name, color=MODEL_COLORS[model_name])
        ax_mape.plot(s['horizon_weeks'], s['MAPE'], 'o-', label=model_name, color=MODEL_COLORS[model_name])
    ax_r2.set_title(f'{crop.capitalize()} — R² by horizon')
    ax_r2.set_xlabel('Horizon (weeks)')
    ax_r2.set_ylabel('R²')
    ax_r2.axhline(0, color='gray', lw=0.5, ls='--')
    ax_r2.legend(fontsize=8)
    ax_mape.set_title(f'{crop.capitalize()} — MAPE by horizon')
    ax_mape.set_xlabel('Horizon (weeks)')
    ax_mape.set_ylabel('MAPE (%)')
    ax_mape.legend(fontsize=8)

plt.suptitle('Tree-Model Comparison at M6 (Full Feature Set)', fontsize=14, y=1.0)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_tree_model_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'  Saved: {fig_path}')

print('\nScript 15b complete.')
