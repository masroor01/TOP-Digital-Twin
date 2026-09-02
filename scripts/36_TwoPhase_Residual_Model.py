# -*- coding: utf-8 -*-
"""
Script 36 — Two-Phase Baseline: Phase 2 Residual Model Training (Stage 4)
=============================================================================
STATUS (2026-08-04): REJECTED. Both this narrowed version and the earlier
full-feature version produce a combined (Phase1 + residual) forecast that
is worse than Phase 1 alone and worse than production M6 at nearly every
crop/horizon cell -- see Model_Output/experiments/two_phase/table_twophase_combined_metrics.csv.
Kept in the repo as a documented negative result, not part of the active
pipeline. Production model remains M6 (scripts/15, 23).

Trains the RESIDUAL model: predicts residual = actual_log_price -
phase1_baseline_pred. NARROWED as of 2026-08-04 (was the full M6 feature
set through 2026-08-03's runs): Phase 2 now sees ONLY Sentinel-2 vegetation
indices (s2_ndvi, s2_evi, s2_valid_frac, s2_ndvi_anom -- NDWI is not yet
computed by Script 14, substituted with valid-fraction and NDVI-anomaly,
which are already-available, related derived metrics; see the session
discussion for why this is a substitution, not the literal NDWI). No
price/arrivals recipe, no market/state encoding, no macro/infra/policy --
everything else moved into Phase 1 as of the same date (see Script 34/35).

Why the narrowing: the full-feature version (Scripts 36 runs through
2026-08-03) gave Phase 2 ~90 features to fit a residual target on as few
as ~19,000 training rows in the sparsest fold -- high capacity, thin
noisy target, a textbook overfitting setup, and plausibly why that
version produced wild instability (one fold/horizon cell hit 232% MAPE).
Restricting to 4 raw vegetation columns removes nearly all of that
capacity. The real, acknowledged risk in the other direction: those 4
columns alone may not carry much residual signal beyond what arrivals
and climate (both already in Phase 1) already provide -- vegetation
indices are a slow-moving proxy for crop condition that likely overlaps
heavily with what rainfall/temperature and actual market arrivals
already explain. This run is the test of that specific trade-off, not
a foregone conclusion either way.

Uses the 2017-2026 window where satellite is actually real
(data/master_weekly_panel_all_layers.csv -- the same panel Script 15/23
use for M6, reused as-is, not rebuilt; only its S2 columns are read here).

Final combined prediction = phase1_baseline_pred + phase2_residual_pred.
This is the actual two-phase architecture's output -- the number Stage 5
compares against M6.

WALK-FORWARD DISCIPLINE -- the single most important correctness property
in this whole two-phase build, worth restating precisely:
For Phase-2 fold k (reusing Script 35's exact 9 annual folds and their
numbering, test years 2017-2025), the residual TRAINING target for any
week must come from a Phase-1 OOF prediction whose own model was fit on
data strictly earlier than that week -- already guaranteed by Script 35's
walk-forward construction. This script adds a second requirement on top:
Phase-2 fold k's residual-training rows must themselves only be drawn
from EARLIER Phase-1 folds (fold number < k), i.e. calendar years
strictly before fold k's test year. This is NOT redundant with Script
35's own guarantee -- it's what stops Phase 2 from training on, say,
2024's residuals to help predict 2022 for a "fold 5" run; without this
rule the fold number would be decorative rather than a real training
cutoff. Fold 0 (test 2017) has NO earlier folds to train on and is
therefore skipped entirely -- there is no valid residual training data
before the very first Phase-1 test year. Fold 1 (test 2018) becomes
usable for the first time as of this revision, training on fold 0's
2017 residuals -- previously skipped when Script 35 only went back to
2018, wasting a full year of otherwise-usable residual training data
at Stage 4's data-scarcest end (see Script 35's docstring for the fix).

Inputs:
  Model_Output/experiments/two_phase/table_baseline_phase_oof_predictions.csv  (Script 35)
  data/master_weekly_panel_all_layers.csv                (Script 22, M6 panel)

Outputs (Model_Output/experiments/two_phase/):
  table_twophase_combined_predictions.csv   per-row: actual, baseline_pred,
                                             residual_pred, combined_pred
  table_twophase_combined_metrics.csv       RMSE/MAE/MAPE/R2 by crop x
                                             horizon x fold, comparable
                                             structure to table_ablation.csv

Run: python scripts/36_TwoPhase_Residual_Model.py
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
OOF_FILE = os.path.join(BASE, 'Model_Output', 'experiments', 'two_phase', 'table_baseline_phase_oof_predictions.csv')
PANEL_FILE = os.path.join(BASE, 'data', 'master_weekly_panel_all_layers.csv')
OUT_DIR = os.path.join(BASE, 'Model_Output', 'experiments', 'two_phase')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED = 42

# Identical fold definitions/numbering to Script 35 -- must match exactly,
# since this script joins on Script 35's 'fold' column. Includes fold 0
# (test 2017, added to Script 35 as a follow-up fix) -- folds 1-8 keep
# their original numbers/test years, this is purely additive.
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
print('SCRIPT 36: TWO-PHASE BASELINE -- STAGE 4 PHASE 2 RESIDUAL MODEL')
print('=' * 65)

for f in [OOF_FILE, PANEL_FILE]:
    if not os.path.exists(f):
        print(f'ERROR: {f} not found.')
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD + FEATURE ENGINEERING -- full M6 feature set (price/arrivals recipe
# identical to Script 35, PLUS Sentinel-2 and infrastructure, both real here
# since this panel is the 2017-2026 window).
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Loading M6 panel (2017-2026, all layers) ...')
df = pd.read_csv(PANEL_FILE, parse_dates=['week_start'])
print(f'  {len(df):,} rows, {df["market"].nunique()} markets, '
      f'{df["week_start"].min().date()} to {df["week_start"].max().date()}')

# NARROWED 2026-08-04: Phase 2's ONLY inputs are Sentinel-2 vegetation
# indices -- no price/arrivals recipe, no market/state encoding, no
# macro/infra/policy (all now exclusively in Phase 1, see Script 35).
# NDWI substituted with valid-fraction + NDVI-anomaly (see module docstring).
S2_FEATS = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']


def build_features(df_in):
    """Identical price/arrivals recipe to Script 35/33 (= Script 15's M1)."""
    out = {}
    for crop in CROPS:
        sub = df_in[df_in['crop'] == crop].copy()
        # Keyed on market_id, not market NAME -- a few market names repeat
        # across different states (e.g. "Fatehabad APMC" in both Haryana and
        # Uttar Pradesh), which would interleave two physically different
        # markets' price series into one shift/rolling computation. See
        # Script 15's build_features() for the reference fix.
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

        # market_enc keyed on market_id, not market NAME -- two same-named
        # markets in different states would otherwise get the identical code.
        sub['market_enc'] = pd.Categorical(sub['market_id']).codes
        for col in ['state']:
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
# PRICE_FEATS/ARR_FEATS are still computed by build_features() above (needed
# for the 'target' shift(-h) construction) but deliberately NOT included in
# Phase 2's model inputs below -- narrowed to S2_FEATS only, see docstring.

print('\n[3] Loading Phase 1 out-of-fold predictions ...')
oof = pd.read_csv(OOF_FILE, parse_dates=['week_start'])
print(f'  {len(oof):,} rows, folds {sorted(oof["fold"].unique())}, '
      f'{oof["week_start"].min().date()} to {oof["week_start"].max().date()}')


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
# 4. TRAINING LOOP -- Phase-2 fold k trains ONLY on residuals from Phase-1
# folds < k (see docstring for why this second walk-forward rule matters).
# Fold 0 has no earlier folds and is skipped; fold 1 (test 2018) is now
# usable for the first time, training on fold 0's 2017 residuals.
# ─────────────────────────────────────────────────────────────────────────────
usable_folds = [f for f in FOLDS if f['fold'] > 0]
print(f'\n[4] Training Phase 2 residual model: {len(usable_folds)} folds x {len(HORIZONS)} horizons x '
      f'{len(CROPS)} crops = {len(usable_folds)*len(HORIZONS)*len(CROPS)} fits ...\n')

combined_frames = []
metric_rows = []
t0_total = time.time()

for crop in CROPS:
    df_crop = feat[crop].copy()
    fcols = [c for c in S2_FEATS if c in df_crop.columns]
    oof_crop = oof[oof['crop'] == crop]

    for h in HORIZONS:
        t0 = time.time()
        df_h = df_crop.copy()
        df_h['target'] = df_h.groupby('market_id')['log_price'].shift(-h)
        df_h = df_h.dropna(subset=['target', 'price_lag_1'])
        oof_ch = oof_crop[oof_crop['horizon_weeks'] == h]

        # Attach residual target: join Phase-1 OOF predictions onto this
        # horizon's feature rows by (market_id, week_start) -- week_start
        # here is the reference/feature week, same convention Script 35
        # used. Joining on market NAME instead of market_id (as this used
        # to) is unsafe: a few market names repeat across different states
        # (e.g. "Fatehabad APMC" in both Haryana and Uttar Pradesh), which
        # would fan the join out and cross-wire one market's Phase-1
        # residual target onto a different market's features.
        _n_before = len(df_h)
        merged = df_h.merge(
            oof_ch[['market_id', 'week_start', 'fold', 'log_price_actual', 'log_price_baseline_pred']],
            on=['market_id', 'week_start'], how='inner'
        )
        # An inner join legitimately dropping unmatched rows is fine; row-
        # count GROWTH means the join fanned out on a duplicate key, which
        # would silently mix predictions across markets/folds.
        assert len(merged) <= _n_before, (
            f'Unexpected row-count growth after OOF join: {_n_before} -> {len(merged)} '
            '-- check for duplicate (market_id, week_start) keys'
        )
        merged['residual_target'] = merged['log_price_actual'] - merged['log_price_baseline_pred']

        for fold_info in usable_folds:
            fold = fold_info['fold']
            train = merged[merged['fold'] < fold]
            test = merged[merged['fold'] == fold]
            if len(train) < 100 or len(test) < 10:
                continue

            X_tr, y_tr = train[fcols], train['residual_target']
            X_te = test[fcols]

            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X_tr, y_tr)
            residual_pred = model.predict(X_te)
            combined_pred = test['log_price_baseline_pred'].values + residual_pred

            m = compute_metrics(test['log_price_actual'].values, combined_pred)
            metric_rows.append({'crop': crop, 'fold': fold, 'horizon_weeks': h,
                                 'n_train': len(train), 'test_year': fold_info['test_start'][:4], **m})

            combined_frames.append(pd.DataFrame({
                'crop': crop, 'market': test['market'].values, 'week_start': test['week_start'].values,
                'horizon_weeks': h, 'fold': fold,
                'log_price_actual': test['log_price_actual'].values,
                'log_price_baseline_pred': test['log_price_baseline_pred'].values,
                'log_price_residual_pred': residual_pred,
                'log_price_combined_pred': combined_pred,
            }))

        elapsed = round(time.time() - t0, 1)
        print(f'  {crop:7s} h={h:>2}w  {len(usable_folds)} folds done  [{elapsed}s]')

print(f'\n  Total time: {(time.time()-t0_total)/60:.1f} min')

combined = pd.concat(combined_frames, ignore_index=True)
combined_path = os.path.join(OUT_DIR, 'table_twophase_combined_predictions.csv')
combined.to_csv(combined_path, index=False)
print(f'\n  Saved: {combined_path}  ({len(combined):,} rows)')

metrics = pd.DataFrame(metric_rows)
metrics_path = os.path.join(OUT_DIR, 'table_twophase_combined_metrics.csv')
metrics.to_csv(metrics_path, index=False)
print(f'  Saved: {metrics_path}')

print('\n[5] Mean MAPE by crop x horizon, combined (Phase 1 + Phase 2), across all usable folds ...\n')
piv = metrics.groupby(['crop', 'horizon_weeks'])['MAPE'].mean().round(2).unstack()
print(piv.to_string())

print('\n[6] Mean MAPE by crop x horizon, RESTRICTED to folds 5-8 (test 2022-2025) -- '
      'the same test years M6 is evaluated on ...\n')
m_recent = metrics[metrics['fold'] >= 5]
piv_recent = m_recent.groupby(['crop', 'horizon_weeks'])['MAPE'].mean().round(2).unstack()
print(piv_recent.to_string())

print('\n' + '=' * 65)
print('Script 36 complete.')
print('\nNext: Stage 5 -- benchmark table_twophase_combined_metrics.csv (folds 5-8)')
print('against M6 (table_ablation.csv) on the identical 2022-2025 test years,')
print('with a Diebold-Mariano test. This is the actual gate.')
