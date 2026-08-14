# -*- coding: utf-8 -*-
"""
Script 33 — Long-History Baseline Validation Experiment
=============================================================================
Tests the core premise behind the "tiered/multi-phase residual modeling"
idea (2026-08-01 discussion): does extending the price/arrivals-only
training window from 9 years (2017-2026, current production) to 23 years
(2003-2026, Script 32's panel) measurably improve forecast accuracy, on
IDENTICAL test folds and an IDENTICAL model recipe?

This is a controlled, single-variable experiment. Both variants below:
  - use the SAME market set (Script 32's long-history panel's own coverage-
    filtered markets, so "longhistory" and "shorthistory" aren't accidentally
    comparing different markets)
  - use the SAME feature engineering (Script 15's exact M1 recipe: price
    lags/rolling stats/seasonality/market encoding + arrivals)
  - use the SAME LightGBM config, same 4 rolling-origin folds, same 4
    horizons, same 3 crops, same test periods (2022/2023/2024/2025)
  - differ ONLY in how far back the TRAINING window is allowed to start:
      "shorthistory": train >= 2017-01-01 (mimics current production)
      "longhistory":  train >= each market's own earliest real data (up to 2003)

If longhistory doesn't measurably beat shorthistory here, the core premise
of the tiered-residual architecture doesn't hold and the bigger rebuild
isn't worth doing. If it does, that justifies the larger investment.

B1_Naive is also recomputed on this panel (naive persistence needs no
training history, but the market set differs slightly from production's
independently-filtered panel, so it's recomputed rather than assumed).

Inputs:
  data/agmarknet_weekly/longhistory/top_weekly_panel_longhistory.csv (Script 32)

Outputs (Model_Output/):
  table_longhistory_validation.csv   RMSE/MAE/MAPE/R2 by crop x horizon x variant

Run: python scripts/33_LongHistory_Validation_Experiment.py
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
# 1. CONFIG (identical to Script 15 where applicable)
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'longhistory', 'top_weekly_panel_longhistory.csv')
PROD_ABLATION_FILE = os.path.join(BASE, 'Model_Output', 'table_ablation.csv')
OUT_DIR = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED = 42
SHORTHISTORY_FLOOR = pd.Timestamp('2017-01-01')   # mimics production's Script 09 START_DATE

FOLDS = [
    {'fold': 1, 'train_end': '2021-06-30', 'val_start': '2021-07-01', 'val_end': '2021-12-31',
     'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'fold': 2, 'train_end': '2022-06-30', 'val_start': '2022-07-01', 'val_end': '2022-12-31',
     'test_start': '2023-01-01', 'test_end': '2023-12-31'},
    {'fold': 3, 'train_end': '2023-06-30', 'val_start': '2023-07-01', 'val_end': '2023-12-31',
     'test_start': '2024-01-01', 'test_end': '2024-12-31'},
    {'fold': 4, 'train_end': '2024-06-30', 'val_start': '2024-07-01', 'val_end': '2024-12-31',
     'test_start': '2025-01-01', 'test_end': '2025-12-31'},
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
print('SCRIPT 33: LONG-HISTORY BASELINE VALIDATION EXPERIMENT')
print('=' * 65)

if not os.path.exists(PANEL_FILE):
    print(f'ERROR: {PANEL_FILE} not found. Run scripts/32_LongHistory_Panel_Builder.py first.')
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD + FEATURE ENGINEERING (identical recipe to Script 15's M1 variant)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Loading long-history panel ...')
df = pd.read_csv(PANEL_FILE, parse_dates=['week_start'])
print(f'  {len(df):,} rows, {df["market"].nunique()} markets, '
      f'{df["week_start"].min().date()} to {df["week_start"].max().date()}')

def build_features(df_in):
    # FIXED 2026-08-14 (full-layer audit): grouping/sorting used to key on
    # 'market' NAME, not market_id -- see Script 23/15's commits for the
    # full discovery story.
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
M_TIER0_FEATS = PRICE_FEATS + ARR_FEATS   # identical to Script 15's M1 feature set


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
# 3. TRAINING LOOP — shorthistory vs longhistory, same market set, same folds
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Running validation: shorthistory (2017+) vs longhistory (2003+) ...')
print(f'    2 variants x {len(FOLDS)} folds x {len(HORIZONS)} horizons x {len(CROPS)} crops = '
      f'{2*len(FOLDS)*len(HORIZONS)*len(CROPS)} fits, plus naive persistence\n')

all_rows = []
t0_total = time.time()

for crop in CROPS:
    df_crop = feat[crop].copy()
    fcols = [c for c in M_TIER0_FEATS if c in df_crop.columns]

    for fold_info in FOLDS:
        fold = fold_info['fold']
        t_end = pd.Timestamp(fold_info['train_end'])
        v_start, v_end = pd.Timestamp(fold_info['val_start']), pd.Timestamp(fold_info['val_end'])
        te_start, te_end = pd.Timestamp(fold_info['test_start']), pd.Timestamp(fold_info['test_end'])

        for h in HORIZONS:
            t0 = time.time()
            df_h = df_crop.copy()
            df_h['target'] = df_h.groupby('market_id')['log_price'].shift(-h)
            df_h = df_h.dropna(subset=['target', 'price_lag_1'])

            val = df_h[(df_h['week_start'] > v_start) & (df_h['week_start'] <= v_end)]
            test = df_h[(df_h['week_start'] >= te_start) & (df_h['week_start'] <= te_end)]
            if len(test) < 10:
                continue

            X_va, y_va = val[fcols].fillna(0), val['target']
            X_te, y_te = test[fcols].fillna(0), test['target']

            # naive persistence: today's real price carried forward, same test rows
            naive_pred = test['log_price'].values
            m_naive = compute_metrics(y_te.values, naive_pred)
            all_rows.append({'crop': crop, 'fold': fold, 'horizon_weeks': h, 'variant': 'B1_Naive', **m_naive})

            for variant, floor in [('shorthistory', SHORTHISTORY_FLOOR), ('longhistory', None)]:
                train = df_h[df_h['week_start'] <= t_end]
                if floor is not None:
                    train = train[train['week_start'] >= floor]
                if len(train) < 100:
                    continue
                X_tr, y_tr = train[fcols].fillna(0), train['target']

                model = lgb.LGBMRegressor(**LGBM_PARAMS)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                y_pred = model.predict(X_te)
                m = compute_metrics(y_te.values, y_pred)
                all_rows.append({'crop': crop, 'fold': fold, 'horizon_weeks': h, 'variant': variant,
                                  'n_train': len(train), 'train_start': train['week_start'].min(),
                                  **m})

            elapsed = round(time.time() - t0, 1)
            print(f'  {crop:7s} fold{fold} h={h:>2}w  done  [{elapsed}s]')

print(f'\n  Total time: {(time.time()-t0_total)/60:.1f} min')

raw = pd.DataFrame(all_rows)
raw_path = os.path.join(OUT_DIR, 'table_longhistory_validation_raw.csv')
raw.to_csv(raw_path, index=False)
print(f'  Saved: {raw_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY — mean across folds, by crop x horizon x variant
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Summary (mean across 4 folds) ...\n')

summary = raw.groupby(['crop', 'horizon_weeks', 'variant']).agg(
    RMSE=('RMSE', 'mean'), MAE=('MAE', 'mean'), MAPE=('MAPE', 'mean'), R2=('R2', 'mean'),
).round(3).reset_index()
summary_path = os.path.join(OUT_DIR, 'table_longhistory_validation.csv')
summary.to_csv(summary_path, index=False)
print(f'Saved: {summary_path}')

for crop in CROPS:
    print(f'\n  {crop.upper()}')
    piv = summary[summary['crop'] == crop].pivot(index='variant', columns='horizon_weeks', values='MAPE')
    piv = piv.reindex(index=['B1_Naive', 'shorthistory', 'longhistory'], columns=HORIZONS)
    print('  MAPE (%):')
    print(piv.to_string())
    piv_r2 = summary[summary['crop'] == crop].pivot(index='variant', columns='horizon_weeks', values='R2')
    piv_r2 = piv_r2.reindex(index=['B1_Naive', 'shorthistory', 'longhistory'], columns=HORIZONS)
    print('  R2:')
    print(piv_r2.to_string())

# Verdict: does longhistory beat shorthistory, and by how much, per crop/horizon?
print('\n[5] Verdict: longhistory vs shorthistory (negative MAPE delta = longhistory better) ...\n')
wide = summary.pivot_table(index=['crop', 'horizon_weeks'], columns='variant', values='MAPE')
wide['MAPE_delta_longhistory_minus_short'] = wide['longhistory'] - wide['shorthistory']
wide['R2_short'] = summary.pivot_table(index=['crop', 'horizon_weeks'], columns='variant', values='R2')['shorthistory']
wide['R2_long'] = summary.pivot_table(index=['crop', 'horizon_weeks'], columns='variant', values='R2')['longhistory']
print(wide.round(3).to_string())

n_improved = (wide['MAPE_delta_longhistory_minus_short'] < -0.5).sum()
n_worse = (wide['MAPE_delta_longhistory_minus_short'] > 0.5).sum()
n_flat = len(wide) - n_improved - n_worse
print(f'\n  Across {len(wide)} crop x horizon cells: longhistory improves MAPE by >0.5pp in '
      f'{n_improved}, worsens by >0.5pp in {n_worse}, roughly flat in {n_flat}.')

print('\n' + '=' * 65)
print('Script 33 complete.')
