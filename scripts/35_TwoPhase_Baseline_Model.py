# -*- coding: utf-8 -*-
"""
Script 35 — Two-Phase Baseline: Phase 1 Model Training (Stage 3)
=====================================================================
Trains the long-window BASELINE model: price/arrivals/climate(non-S2)/
macro/policy, on the full 2003-2026 window built in Stage 2
(data/baseline_phase_panel.csv), using each market's own earliest real
data as the training floor (no shorthistory cutoff -- that comparison
already happened in Script 33; this is the actual Phase 1 model, not
another validation experiment).

Fold structure -- WHY THIS DIFFERS FROM THE PROJECT'S USUAL 4 FOLDS:
Every other script in this project (15, 18, 27, 28, 30, 33...) uses 4
folds with test years 2022-2025, because that's what's needed to
EVALUATE a model. This script's job is different: its output isn't a
final accuracy number, it's a per-row PREDICTION that Stage 4's residual
model will subtract from the actual price to build ITS OWN training
target. That means Phase 1 needs genuine out-of-fold predictions
covering every week Phase 2 might train OR test on -- not just 4
disjoint test years. Using only the usual 4 folds (predictions for 2022,
2023, 2024, 2025 alone) would leave Phase 2 with no valid residual
target for 2017-2021, which is most of its likely training window.

Resolved by extending to 9 annual expanding-window folds, test years
2017-2025 (train up to 31-Dec of year Y-1, test all of year Y, each
fold's training floor is still each market's own earliest real data,
not shorthistory-capped) -- same annual-boundary convention as the
usual 4 folds, just finer-grained so Phase 2 gets continuous walk-
forward coverage across its whole 2017-2026 window instead of 4 gaps.

Fold 0 (test 2017) added 2026-08-03 as a follow-up fix: the original
8-fold version (test years 2018-2025) let all of 2017 fall straight
into fold 1's training set with no out-of-fold counterpart, silently
wasting the one calendar year Stage 4 could have used but couldn't --
Stage 4's own features (satellite, infrastructure) aren't real before
2017, so every recoverable year at that floor matters more than
elsewhere. This alone doesn't fix the deeper data-scarcity issue Stage
4 has at its earliest usable folds (still bounded by how much calendar
time has actually elapsed since 2017), but it recovers what was a pure,
avoidable gap at no cost -- fold 0 is genuine walk-forward like every
other fold, not a compromise.

Every fold's test-period prediction is out-of-sample by construction
(the fold's model is fit only on data strictly before that test year)
-- this is the property Stage 4 depends on. Getting it wrong here
would leak information into the residual model with no visible symptom
until results looked suspiciously good.

Feature set (price/arrivals recipe identical to Script 33's, which is
Script 15's M1 recipe -- kept identical for continuity):
  PRICE_FEATS + ARR_FEATS   (lags, rolling stats, seasonality, market/
                              state encoding -- see build_features())
  CLIMATE_FEATS             era5_*, chirps_*, modis_ndvi/evi/lst_*
                             (Sentinel-2 EXCLUDED -- mid-2015 mission
                             floor, belongs to Phase 2 only)
  MACRO_FEATS                all CMIE + RBI + PPAC longhistory columns
  INFRA_FEATS                 wages, cold storage, road density -- added
                              2026-08-04, reversing the original
                              exclusion, now that Phase 2 narrows to
                              Sentinel-2 only (see Script 36). Real
                              coverage 2017-2025, NaN outside.
  POLICY_FEATS               all 5 (NaN before 2017 by Stage 2's design
                              -- LightGBM handles missing features
                              natively, no imputation applied)

Inputs:
  data/baseline_phase_panel.csv (Script 34)

Outputs (Model_Output/experiments/two_phase/):
  table_baseline_phase_oof_predictions.csv   per-row out-of-fold predictions,
                                              the input Stage 4 needs
  table_baseline_phase_metrics.csv           RMSE/MAE/MAPE/R2 by crop x
                                              horizon x fold, for sanity-
                                              checking Phase 1 alone

Run: python scripts/35_TwoPhase_Baseline_Model.py
"""

import io, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(BASE, 'data', 'baseline_phase_panel.csv')
OUT_DIR = os.path.join(BASE, 'Model_Output', 'experiments', 'two_phase')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED = 42

# 9 annual expanding-window folds, test years 2017-2025 -- see docstring for
# why this differs from the project's usual 4 (test years 2022-2025).
# Fold 0 (test 2017) added 2026-08-03: the original range started at 2018,
# which meant all of 2017 went straight into fold 1's training set and was
# NEVER surfaced as an out-of-fold residual -- a real, avoidable gap in
# Stage 4's available residual-training data (Stage 4 can only use 2017+
# anyway, since that's where its own satellite/infrastructure features
# start being real, so every recoverable year in that window matters).
# Folds 1-8 keep their original numbers/test years unchanged -- purely
# additive, no renumbering.
FOLDS = [
    {'fold': y - 2017, 'train_end': f'{y-1}-12-31', 'val_start': f'{y-1}-07-01', 'val_end': f'{y-1}-12-31',
     'test_start': f'{y}-01-01', 'test_end': f'{y}-12-31'}
    for y in range(2017, 2026)
]

LGBM_PARAMS = dict(
    objective='regression', metric='rmse', n_estimators=1000, learning_rate=0.05,
    num_leaves=127, max_depth=-1, min_child_samples=20, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=SEED, verbose=-1,
)

LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS = [4, 8, 13]

print('=' * 65)
print('SCRIPT 35: TWO-PHASE BASELINE -- STAGE 3 PHASE 1 MODEL')
print('=' * 65)

if not os.path.exists(PANEL_FILE):
    print(f'ERROR: {PANEL_FILE} not found. Run scripts/34_Baseline_Phase_Panel_Join.py first.')
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Loading baseline-phase panel ...')
df = pd.read_csv(PANEL_FILE, parse_dates=['week_start'])
print(f'  {len(df):,} rows, {df["market"].nunique()} markets, '
      f'{df["week_start"].min().date()} to {df["week_start"].max().date()}')

CLIMATE_FEATS = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38',
                  'chirps_rain_mm', 'chirps_rain_max', 'chirps_excess',
                  'modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']
MACRO_FEATS = ['bank_credit_agri_cr', 'export_veg_usd_mn', 'import_veg_usd_mn', 'crude_oil_usd_bbl',
                'iip_food_proc', 'agri_wages_rs_day', 'repo_rate_pct', 'reverse_repo_pct',
                'usdinr_monthly_avg', 'wpi_fruits_vegetables', 'wpi_vegetables_total', 'wpi_potato',
                'wpi_onion', 'wpi_tomato', 'diesel_4city_rs_litre', 'lpg_nonsub_4city_rs_cyl',
                'diesel_delhi_per_L', 'lpg_nonsub_delhi_per14kg']
INFRA_FEATS = ['wage_agri_men', 'wage_agri_women', 'cold_storage_n_facilities',
                'cold_storage_capacity_mt', 'road_density_per_100_sqkm']  # added 2026-08-04,
# see Script 34's docstring -- real coverage 2017-2025, genuine NaN outside (not backfilled)
POLICY_FEATS = ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct',
                 'market_intervention_flag', 'operation_greens_active']
assert not any(c.startswith('s2_') for c in CLIMATE_FEATS), 'Sentinel-2 must not leak into Phase 1'


def build_features(df_in):
    """Price/arrivals recipe identical to Script 33 (= Script 15's M1)."""
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
            sub[f'price_roll_std_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())

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

        for col in ['state', 'market']:
            sub[f'{col}_enc'] = pd.Categorical(sub[col]).codes
        sub['year_trend'] = sub['week_start'].dt.year - 2017

        out[crop] = sub
    return out


print('[2] Engineering features ...')
feat = build_features(df)
for crop in CROPS:
    print(f'  {crop:7s}: {len(feat[crop]):,} rows')

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
BASELINE_FEATS = PRICE_FEATS + ARR_FEATS + CLIMATE_FEATS + MACRO_FEATS + INFRA_FEATS + POLICY_FEATS


def compute_metrics(y_true_log, y_pred_log):
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, R2=np.nan, N=0)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    ss_res, ss_tot = np.sum((yt - yp) ** 2), np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(RMSE=round(rmse, 1), MAE=round(mae, 1), MAPE=round(mape, 2), R2=round(r2, 4), N=len(yt))


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOOP -- 8 folds x 4 horizons x 3 crops, expanding window,
# each market's own earliest real data as the training floor (no shorthistory
# cap -- that's the whole point of Phase 1).
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[3] Training Phase 1 baseline: {len(FOLDS)} folds x {len(HORIZONS)} horizons x '
      f'{len(CROPS)} crops = {len(FOLDS)*len(HORIZONS)*len(CROPS)} fits ...\n')

oof_frames = []
metric_rows = []
t0_total = time.time()

for crop in CROPS:
    df_crop = feat[crop].copy()
    fcols = [c for c in BASELINE_FEATS if c in df_crop.columns]

    for fold_info in FOLDS:
        fold = fold_info['fold']
        t_end = pd.Timestamp(fold_info['train_end'])
        v_start, v_end = pd.Timestamp(fold_info['val_start']), pd.Timestamp(fold_info['val_end'])
        te_start, te_end = pd.Timestamp(fold_info['test_start']), pd.Timestamp(fold_info['test_end'])

        for h in HORIZONS:
            t0 = time.time()
            df_h = df_crop.copy()
            df_h['target'] = df_h.groupby('market')['log_price'].shift(-h)
            df_h = df_h.dropna(subset=['target', 'price_lag_1'])

            train = df_h[df_h['week_start'] <= t_end]
            val = df_h[(df_h['week_start'] > v_start) & (df_h['week_start'] <= v_end)]
            test = df_h[(df_h['week_start'] >= te_start) & (df_h['week_start'] <= te_end)]
            if len(train) < 100 or len(test) < 10:
                continue

            X_tr, y_tr = train[fcols], train['target']
            X_va, y_va = val[fcols], val['target']
            X_te, y_te = test[fcols], test['target']

            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            y_pred = model.predict(X_te)

            m = compute_metrics(y_te.values, y_pred)
            metric_rows.append({'crop': crop, 'fold': fold, 'horizon_weeks': h,
                                 'n_train': len(train), 'train_start': train['week_start'].min(),
                                 'test_start': te_start, 'test_end': te_end, **m})

            oof_frames.append(pd.DataFrame({
                'crop': crop, 'market': test['market'].values, 'week_start': test['week_start'].values,
                'horizon_weeks': h, 'fold': fold,
                'log_price_actual': y_te.values, 'log_price_baseline_pred': y_pred,
            }))

            elapsed = round(time.time() - t0, 1)
            print(f'  {crop:7s} fold{fold} (test {fold_info["test_start"][:4]}) h={h:>2}w  '
                  f'n_train={len(train):>7,}  MAPE={m["MAPE"]:>6.2f}%  [{elapsed}s]')

print(f'\n  Total time: {(time.time()-t0_total)/60:.1f} min')

oof = pd.concat(oof_frames, ignore_index=True)
oof_path = os.path.join(OUT_DIR, 'table_baseline_phase_oof_predictions.csv')
oof.to_csv(oof_path, index=False)
print(f'\n  Saved: {oof_path}  ({len(oof):,} rows)')
print(f'  Coverage: {oof["week_start"].min()} to {oof["week_start"].max()}')

metrics = pd.DataFrame(metric_rows)
metrics_path = os.path.join(OUT_DIR, 'table_baseline_phase_metrics.csv')
metrics.to_csv(metrics_path, index=False)
print(f'  Saved: {metrics_path}')

print(f'\n[4] Mean MAPE by crop x horizon, across all {len(FOLDS)} folds ...\n')
piv = metrics.groupby(['crop', 'horizon_weeks'])['MAPE'].mean().round(2).unstack()
print(piv.to_string())

print('\n' + '=' * 65)
print('Script 35 complete.')
print('\nNext: Stage 4 -- train the Phase 2 residual model on')
print('table_baseline_phase_oof_predictions.csv, restricted to where satellite')
print('(Sentinel-2) and infrastructure are actually available (~2017-2026).')
