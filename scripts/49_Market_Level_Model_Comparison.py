# -*- coding: utf-8 -*-
"""
Script 49 -- Per-MARKET Model vs. Shared Model: A Head-to-Head Test
=============================================================================
Direct follow-up to Script 48 (state-restricted models lost to the shared
model, 15/24 significant comparisons vs. 5/24, even in the best-data
states). User asked to push the same question one level further: does
fitting one model per (crop, state, MARKET) -- not just per state -- beat
the shared model?

Same rationale as Script 48, more acute here: LightGBM's M6 feature set
(~58 lag/rolling/macro/climate/satellite/infra/policy columns once
market_enc/state_enc are dropped as constant for a single market) needs
real data volume. A single market has at most ~504 weeks of history ever
(2017-2026); after the fold-1 train cutoff (2017-01 to 2021-06) that's
roughly 230 rows to fit ~58 features on -- far less than even the losing
state-restricted fits in Script 48 (tens of thousands of rows). If
state-restricted already lost on a variance basis, market-restricted
should lose more decisively still -- this script checks that rather than
assuming it.

Scope: the 3 markets with the FULLEST possible history (504/504 weeks, no
coverage gaps) per crop, all of which are also the project's own
identified "market leader" markets (Script 42) -- deliberately the
best-case scenario for the per-market argument, same principle as Script
48's best-case state selection:
  tomato: Shadnagar APMC (Telangana), Kolar APMC (Karnataka), Kota (F&V) APMC (Rajasthan)
  onion:  Chandvad APMC (Maharashtra), Pimpalgaon APMC (Maharashtra), Bharuch APMC (Gujarat)
  potato: Chakdah APMC (West Bengal), Dehradoon APMC (Uttarakhand), Kashipur APMC (Uttarakhand)
(Dehradoon/Kashipur chosen for potato specifically to test Uttarakhand
further -- Script 48 found Uttarakhand's STATE-restricted model was the
one genuine win, at both long horizons.)

Method: identical rolling-origin CV (Script 15's 5 folds) and LightGBM
hyperparameters as the shared model's own recorded backtest
(`dm_market_level_predictions.csv`), training data restricted to ONLY
that one market's own rows. DM-tested against the shared model's
predictions for the exact same (market_id, week_start) cells.

Outputs:
  Model_Output/table_market_restricted_vs_shared_model_dm_test.csv
  Model_Output/market_restricted_model_predictions.csv   (audit trail)

Run: python scripts/49_Market_Level_Model_Comparison.py
"""
import io, os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE  = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CMIE_FILE = os.path.join(BASE, 'data', 'cmie_macro',      'cmie_macro_2017_2025.csv')
RBI_FILE  = os.path.join(BASE, 'data', 'rbi_dbie',        'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE = os.path.join(BASE, 'data', 'ppac_macro',      'ppac_diesel_lpg_2017_2025.csv')
SAT_FILE  = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
WAGE_FILE = os.path.join(BASE, 'data', 'labour_wages',    'wage_agri_state_monthly.csv')
COLD_FILE = os.path.join(BASE, 'data', 'infrastructure',  'cold_storage_by_state.csv')
ROAD_FILE = os.path.join(BASE, 'data', 'infrastructure',  'road_density_state_annual.csv')
POLICY_FILE = os.path.join(BASE, 'data', 'policy_trade',  'policy_weekly_features.csv')
OUT_DIR   = os.path.join(BASE, 'Model_Output')

SEED = 42
HORIZONS = [1, 4, 13, 26]
LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS = [4, 8, 13]

# Top-3 fullest-history markets per crop -- see docstring for rationale.
TARGET_MARKETS = {
    'tomato': [9.0, 112.0, 2481.0],      # Shadnagar, Kolar, Kota (F&V)
    'onion':  [54.0, 162.0, 550.0],      # Bharuch, Pimpalgaon, Chandvad
    'potato': [303.0, 319.0, 3350.0],    # Dehradoon, Kashipur, Chakdah
}

# Identical rolling-origin folds as Scripts 15/18b/48.
FOLDS = [
    {'fold': 1, 'train_end': '2021-06-30', 'val_start': '2021-07-01', 'val_end': '2021-12-31',
     'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'fold': 2, 'train_end': '2022-06-30', 'val_start': '2022-07-01', 'val_end': '2022-12-31',
     'test_start': '2023-01-01', 'test_end': '2023-12-31'},
    {'fold': 3, 'train_end': '2023-06-30', 'val_start': '2023-07-01', 'val_end': '2023-12-31',
     'test_start': '2024-01-01', 'test_end': '2024-12-31'},
    {'fold': 4, 'train_end': '2024-06-30', 'val_start': '2024-07-01', 'val_end': '2024-12-31',
     'test_start': '2025-01-01', 'test_end': '2025-12-31'},
    {'fold': 5, 'train_end': '2025-06-30', 'val_start': '2025-07-01', 'val_end': '2025-12-31',
     'test_start': '2026-01-01', 'test_end': '2026-12-31'},
]

LGBM_PARAMS = dict(
    objective='regression', metric='rmse', n_estimators=1000,
    learning_rate=0.05, num_leaves=127, max_depth=-1,
    min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
    random_state=SEED, verbose=-1,
)
POLICY_MONOTONE = {'export_banned': -1, 'export_duty_pct': -1, 'mep_usd_per_tonne': -1}


def build_monotone(fcols):
    if not any(c in POLICY_MONOTONE for c in fcols):
        return None
    return [POLICY_MONOTONE.get(c, 0) for c in fcols]


print('=' * 78)
print('SCRIPT 49: PER-MARKET MODEL vs. SHARED MODEL -- HEAD-TO-HEAD DM TEST')
print('=' * 78)

# ── 1. Load + join all layers (identical to Scripts 15/23/48) ──
print('\n[1] Loading panel + all layers ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2030-12-31')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)
df['year'] = df['week_start'].dt.year
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
ERA5_COLS = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38']
CHIRPS_COLS = ['chirps_rain_mm', 'chirps_rain_max', 'chirps_excess']
S2_COLS = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']
MODIS_COLS = ['modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']
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
        sat[new_col] = (sat.groupby('crop')[col].transform(lambda x: x.shift(1).rolling(w, min_periods=2).agg(agg)))
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

# ── 2. Feature engineering (identical to Scripts 15/23/48) ──
print('\n[2] Engineering features ...')


def build_features(df_in, crops):
    out = {}
    for crop in crops:
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
        if 'market_id' in sub.columns:
            sub['market_enc'] = pd.Categorical(sub['market_id']).codes
        for col in ['state']:
            if col in sub.columns:
                sub[f'{col}_enc'] = pd.Categorical(sub[col]).codes
        sub['year_trend'] = sub['week_start'].dt.year - 2017
        out[crop] = sub
    return out


CROPS = list(TARGET_MARKETS.keys())
feat = build_features(df, CROPS)

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
# market_enc AND state_enc are both constant (single market -> single
# state) once training data is restricted to one market -- drop both, same
# principle as Script 48 dropping state_enc for state-restricted fits.
M6_FEATS_MARKET = [c for c in M6_FEATS if c not in ('market_enc', 'state_enc')]

print(f'  M6 feature list: {len(M6_FEATS)} candidate columns '
      f'({len(M6_FEATS_MARKET)} for market-restricted fits, market_enc/state_enc dropped as constant)')

# ── 3. Train market-restricted M6 models, same folds/horizons as the shared model ──
print('\n[3] Training market-restricted M6 models ...\n')

pred_rows = []
t0_total = time.time()

for crop in CROPS:
    df_crop = feat[crop]
    fcols_all = [c for c in M6_FEATS_MARKET if c in df_crop.columns]

    for market_id in TARGET_MARKETS[crop]:
        df_mkt = df_crop[df_crop['market_id'] == market_id].copy()
        if df_mkt.empty:
            print(f'  {crop} / market_id={market_id}: NOT FOUND in panel, skipping')
            continue
        market_name = df_mkt['market'].iloc[0]
        state_name = df_mkt['state'].iloc[0]
        print(f'  {crop} / {market_name} ({state_name}, market_id={market_id}): {len(df_mkt):,} rows')

        for fold_info in FOLDS:
            fold = fold_info['fold']
            t_end = pd.Timestamp(fold_info['train_end'])
            v_start = pd.Timestamp(fold_info['val_start'])
            v_end = pd.Timestamp(fold_info['val_end'])
            te_start = pd.Timestamp(fold_info['test_start'])
            te_end = pd.Timestamp(fold_info['test_end'])

            for h in HORIZONS:
                t0 = time.time()
                df_h = df_mkt.copy()
                df_h['target'] = df_h['log_price'].shift(-h)
                df_h = df_h.dropna(subset=[c for c in ['target', 'price_lag_1'] if c in df_h.columns])

                train = df_h[df_h['week_start'] <= t_end]
                val = df_h[(df_h['week_start'] > v_start) & (df_h['week_start'] <= v_end)]
                test = df_h[(df_h['week_start'] >= te_start) & (df_h['week_start'] <= te_end)]

                if len(train) < 50 or len(val) < 5 or len(test) < 5:
                    print(f'    fold{fold} h={h:>2}w  SKIPPED (train={len(train)}, val={len(val)}, '
                          f'test={len(test)} -- too little data)')
                    continue

                X_tr, y_tr = train[fcols_all].fillna(0), train['target']
                X_va, y_va = val[fcols_all].fillna(0), val['target']
                X_te, y_te = test[fcols_all].fillna(0), test['target']

                fit_params = dict(LGBM_PARAMS)
                mono = build_monotone(fcols_all)
                if mono is not None:
                    fit_params['monotone_constraints'] = mono

                model = lgb.LGBMRegressor(**fit_params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                y_pred = model.predict(X_te)
                trees = model.best_iteration_ or LGBM_PARAMS['n_estimators']

                out = pd.DataFrame({
                    'crop': crop, 'market_id': market_id, 'market': market_name, 'state': state_name,
                    'fold': fold, 'horizon_weeks': h,
                    'week_start': test['week_start'].values,
                    'y_true': np.expm1(y_te.values),
                    'y_pred_market': np.expm1(y_pred),
                })
                pred_rows.append(out)
                elapsed = round(time.time() - t0, 1)
                print(f'    fold{fold} h={h:>2}w | train={len(train):>4,} test={len(test):>3,} '
                      f'trees={trees:>4} [{elapsed}s]')

print(f'\n  Total training time: {(time.time() - t0_total) / 60:.1f} min')

market_preds = pd.concat(pred_rows, ignore_index=True)
market_pred_path = os.path.join(OUT_DIR, 'market_restricted_model_predictions.csv')
market_preds.to_csv(market_pred_path, index=False, encoding='utf-8')
print(f'\n  Saved: {market_pred_path} ({len(market_preds):,} rows)')

# ── 4. Load the shared model's own recorded backtest, restrict to the same cells ──
print('\n[4] Loading shared-model (M6) backtest for the same markets/weeks ...')
shared = pd.read_csv(os.path.join(OUT_DIR, 'dm_market_level_predictions.csv'), parse_dates=['week_start'])
shared = shared[shared['variant'] == 'M6'][['crop', 'market_id', 'fold', 'horizon_weeks', 'week_start', 'y_true', 'y_pred']]
shared = shared.rename(columns={'y_pred': 'y_pred_shared', 'y_true': 'y_true_shared'})

merged = market_preds.merge(
    shared, on=['crop', 'market_id', 'fold', 'horizon_weeks', 'week_start'], how='inner'
)
print(f'  {len(merged):,} matched (market, week) cells between market-restricted model and shared model')
mismatch = (merged['y_true'] - merged['y_true_shared']).abs()
print(f'  y_true sanity check: max |diff| = {mismatch.max():.6f} (should be ~0, same ground truth)')

# Guard the MAPE division below against a zero (or non-finite) actual price
# -- same filter as Script 47's market-level accuracy table.
n_before = len(merged)
merged = merged[(merged['y_true'] > 0) & np.isfinite(merged['y_true']) &
                 np.isfinite(merged['y_pred_shared']) & np.isfinite(merged['y_pred_market'])].copy()
if len(merged) < n_before:
    print(f'  Dropped {n_before - len(merged):,} rows with y_true<=0 or non-finite prices before MAPE calc')


# ── 5. DM test: market-restricted model vs. shared model, per (crop, market, horizon) ──
def diebold_mariano_test(y_true, pred_a, pred_b, h):
    """Same HLN-corrected DM test as Scripts 18/37/48. pred_a=shared (baseline), pred_b=market (richer)."""
    e_a = y_true - pred_a
    e_b = y_true - pred_b
    d = e_a ** 2 - e_b ** 2
    T = len(d)
    if T < 10:
        return dict(DM_stat=np.nan, p_value=np.nan, mean_d=np.nan, n=T)
    d_mean = d.mean()
    max_lag = max(h - 1, 0)
    gamma0 = np.mean((d - d_mean) ** 2)
    var_d = gamma0
    for k in range(1, min(max_lag, T - 1) + 1):
        cov_k = np.mean((d[:-k] - d_mean) * (d[k:] - d_mean))
        var_d += 2 * cov_k
    var_mean_d = var_d / T
    if not np.isfinite(var_mean_d) or var_mean_d <= 0:
        return dict(DM_stat=np.nan, p_value=np.nan, mean_d=round(d_mean, 6), n=T)
    dm_raw = d_mean / np.sqrt(var_mean_d)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_adj = dm_raw * hln_factor if np.isfinite(hln_factor) and hln_factor > 0 else dm_raw
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_adj), df=max(T - 1, 1)))
    return dict(DM_stat=round(float(dm_adj), 4), p_value=round(float(p_value), 4),
                mean_d=round(float(d_mean), 6), n=T)


def sig_stars(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


print('\n[5] Running DM tests (shared = baseline, market-restricted = richer) ...\n')
rows = []
for crop in CROPS:
    for market_id in TARGET_MARKETS[crop]:
        cell_mkt = merged[(merged['crop'] == crop) & (merged['market_id'] == market_id)]
        if cell_mkt.empty:
            continue
        market_name = cell_mkt['market'].iloc[0]
        for h in HORIZONS:
            cell = cell_mkt[cell_mkt['horizon_weeks'] == h]
            if len(cell) < 10:
                print(f'  {crop:7s} {market_name:30s} h={h:>2}w  SKIPPED (only {len(cell)} matched cells)')
                continue
            y_true = cell['y_true'].values
            mape_shared = np.mean(np.abs((y_true - cell['y_pred_shared'].values) / y_true)) * 100
            mape_market = np.mean(np.abs((y_true - cell['y_pred_market'].values) / y_true)) * 100
            result = diebold_mariano_test(y_true, cell['y_pred_shared'].values, cell['y_pred_market'].values, h)
            better = 'market-restricted' if (not pd.isna(result['mean_d']) and result['mean_d'] > 0) \
                else ('shared' if not pd.isna(result['mean_d']) else 'n/a')
            row = {
                'crop': crop, 'market_id': market_id, 'market': market_name, 'horizon_weeks': h,
                'mape_shared': round(mape_shared, 2), 'mape_market_restricted': round(mape_market, 2),
                **result, 'better_model': better,
                'significant_5pct': (not pd.isna(result['p_value'])) and result['p_value'] < 0.05,
            }
            rows.append(row)
            sig = sig_stars(result['p_value'])
            print(f'  {crop:7s} {market_name:30s} h={h:>2}w  MAPE shared={mape_shared:>5.1f}%  '
                  f'market={mape_market:>5.1f}%  DM={result["DM_stat"]:>7.3f}  p={result["p_value"]:.4f}{sig:<4s} '
                  f'n={result["n"]:>3}  better={better}')

dm_results = pd.DataFrame(rows)
dm_path = os.path.join(OUT_DIR, 'table_market_restricted_vs_shared_model_dm_test.csv')
dm_results.to_csv(dm_path, index=False, encoding='utf-8')
print(f'\n[6] Saved: {dm_path}')

n_sig = dm_results['significant_5pct'].sum()
n_market_wins = ((dm_results['better_model'] == 'market-restricted') & dm_results['significant_5pct']).sum()
n_shared_wins = ((dm_results['better_model'] == 'shared') & dm_results['significant_5pct']).sum()
print(f'\n  {n_sig}/{len(dm_results)} comparisons significant at p<0.05')
print(f'  Significant market-restricted-model wins: {n_market_wins}')
print(f'  Significant shared-model wins: {n_shared_wins}')
print(f'  Not significant (no real difference detected): {len(dm_results) - n_sig}')

print('\n' + '=' * 78)
print('Script 49 complete.')
print('=' * 78)
