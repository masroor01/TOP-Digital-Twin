# -*- coding: utf-8 -*-
"""
Script 12 — Module B Upgraded: Rolling-Origin CV + Multi-Horizon + Spike Probability
======================================================================================
Upgrades from simple temporal split → rolling-origin validation.
Trains separate LightGBM models per (crop × horizon × fold).

Design
------
Rolling-origin folds (expanding window):
  Fold 1 : Train 2017-2021  |  Val 2021 (last 6 months)  |  Test 2022
  Fold 2 : Train 2017-2022  |  Val 2022 (last 6 months)  |  Test 2023
  Fold 3 : Train 2017-2023  |  Val 2023 (last 6 months)  |  Test 2024

Forecast horizons:  h = 1, 4, 13, 26 weeks
Target            : log1p(modal_price_weighted) → back-transformed for metrics
Secondary target  : price-spike probability (>90th pct of train prices)

Ablation hook
-------------
Call run_model_variant(feat, variant_name, feature_cols) from Script 13
to slot into the ablation table without rewriting this file.

Outputs (Model_Output/)
-----------------------
  table_rolling_origin_metrics.csv    — RMSE/MAE/MAPE/R² per crop×horizon×fold
  table_spike_auc.csv                 — AUC-ROC per crop×fold
  fig_horizon_r2.png                  — R² vs horizon (3-crop panel)
  fig_rolling_origin_rmse.png         — RMSE per fold × horizon
  fig_spike_roc.png                   — ROC curves for spike model

Run: python scripts/12_ModuleB_RollingOrigin_MultiHorizon.py
"""

import io, os, sys, time, warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             roc_auc_score, roc_curve)
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = r'C:\Users\masro\Documents\TOP_Digital_Twin'
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CMIE_FILE= os.path.join(BASE, 'data', 'cmie_macro',      'cmie_macro_2017_2024.csv')
RBI_FILE = os.path.join(BASE, 'data', 'rbi_dbie',        'rbi_dbie_macro_2017_2024.csv')
PPAC_FILE= os.path.join(BASE, 'data', 'ppac_macro',      'ppac_diesel_lpg_2017_2024.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]          # weeks ahead
SPIKE_PCT= 90                       # percentile threshold for spike label
SEED     = 42

# Rolling-origin folds: (train_end, val_start, val_end, test_start, test_end)
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
]

LGBM_PARAMS = dict(
    objective        = 'regression',
    metric           = 'rmse',
    n_estimators     = 800,
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

LGBM_CLF_PARAMS = dict(
    objective        = 'binary',
    metric           = 'auc',
    n_estimators     = 500,
    learning_rate    = 0.05,
    num_leaves       = 63,
    min_child_samples= 20,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    n_jobs           = -1,
    random_state     = SEED,
    verbose          = -1,
)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})
CROP_COLORS = {'tomato': '#E63946', 'onion': '#F4A261', 'potato': '#457B9D'}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print('='*65)
print('MODULE B UPGRADED: Rolling-Origin + Multi-Horizon')
print('='*65)

print('\n[1] Loading data ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2024-12-31')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)

# Macro tables — all share columns: date, year, month, <series...>
macro_dfs = []
for fpath in [CMIE_FILE, RBI_FILE, PPAC_FILE]:
    if os.path.exists(fpath):
        m = pd.read_csv(fpath)
        macro_dfs.append(m)

if macro_dfs:
    macro = macro_dfs[0]
    for m in macro_dfs[1:]:
        macro = macro.merge(m, on=['year', 'month'], how='outer',
                            suffixes=('', '_dup'))
        macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
    df['year']  = df['week_start'].dt.year
    df['month'] = df['week_start'].dt.month
    drop_cols = [c for c in ['date','date_x','date_y'] if c in macro.columns]
    df = df.merge(macro.drop(columns=drop_cols, errors='ignore'),
                  on=['year','month'], how='left')
    print(f'   Macro joined: {macro.shape[1]-3} series')
    MACRO_COLS = [c for c in macro.columns if c not in ('date','year','month')]
else:
    MACRO_COLS = []
    print('   No macro files found — running market-only features')

print(f'   Panel: {len(df):,} rows | crops: {df["crop"].unique().tolist()}')


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (same as Module B + log transform)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[2] Engineering features ...')

LAG_WEEKS  = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS  = [4, 8, 13]

def build_features(df):
    """Build feature matrix on full df. Returns dict of per-crop DataFrames."""
    feat = {}
    for crop in CROPS:
        sub = df[df['crop'] == crop].copy()
        sub = sub.sort_values(['market', 'week_start'])

        # Log price (primary target base)
        sub['log_price'] = np.log1p(sub['modal_price_weighted'])

        # Lag features (price)
        for lag in LAG_WEEKS:
            sub[f'price_lag_{lag}'] = (sub.groupby('market')['log_price']
                                          .shift(lag))
        # Rolling stats on log price
        for w in ROLL_WINS:
            g = sub.groupby('market')['log_price']
            sub[f'price_roll_mean_{w}'] = g.transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).mean())
            sub[f'price_roll_std_{w}']  = g.transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).std())

        # Arrivals lags + rolling
        if 'arrivals_tonnes_week' in sub.columns:
            sub['log_arr'] = np.log1p(sub['arrivals_tonnes_week'].clip(lower=0))
            for lag in [1, 2, 4]:
                sub[f'arr_lag_{lag}'] = (sub.groupby('market')['log_arr']
                                             .shift(lag))
            for w in [4, 8]:
                sub[f'arr_roll_mean_{w}'] = (sub.groupby('market')['log_arr']
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=2).mean()))

        # YoY price change
        sub['price_yoy'] = (sub.groupby('market')['log_price']
                               .shift(52))

        # Sinusoidal seasonality
        sub['week_num'] = sub['week_start'].dt.isocalendar().week.astype(int)
        sub['sin_week'] = np.sin(2 * np.pi * sub['week_num'] / 52)
        sub['cos_week'] = np.cos(2 * np.pi * sub['week_num'] / 52)

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

        # Time features
        sub['year']       = sub['week_start'].dt.year
        sub['month']      = sub['week_start'].dt.month
        sub['year_trend'] = sub['year'] - 2017

        feat[crop] = sub
    return feat

feat = build_features(df)

# Identify feature columns (everything except targets and identifiers)
_EXCLUDE = {'crop','market','state','district','state_code','market_id',
            'arrivals_tonnes_week','modal_price_weighted','log_price','log_arr',
            'week_start','year_month','year','month','date',
            'next_price','target','spike'}

def get_feature_cols(df_crop, macro_cols, include_macro=True):
    candidates = [c for c in df_crop.columns if c not in _EXCLUDE
                  and not c.startswith('Unnamed')]
    if not include_macro:
        candidates = [c for c in candidates if c not in macro_cols]
    return candidates

for crop in CROPS:
    fcols = get_feature_cols(feat[crop], MACRO_COLS)
    print(f'   {crop:8s}: {feat[crop].shape[0]:>8,} rows | {len(fcols)} features')


# ══════════════════════════════════════════════════════════════════════════════
# 3. METRICS HELPER
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true_log, y_pred_log):
    """Back-transform from log space; compute metrics on original price scale."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask   = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, R2=np.nan, N=0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(RMSE=round(rmse,1), MAE=round(mae,1),
                MAPE=round(mape,2), R2=round(r2,4), N=len(y_true))


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROLLING-ORIGIN TRAINING — MULTI-HORIZON
# ══════════════════════════════════════════════════════════════════════════════
print('\n[3] Rolling-origin CV — multi-horizon training ...')
print(f'    Folds: {len(FOLDS)} | Horizons: {HORIZONS} | Crops: {len(CROPS)}')
print(f'    Total model fits: {len(FOLDS) * len(HORIZONS) * len(CROPS)}\n')

all_metrics = []
t0_total = time.time()

for crop in CROPS:
    df_crop = feat[crop].copy()
    fcols   = get_feature_cols(df_crop, MACRO_COLS, include_macro=True)

    for fold_info in FOLDS:
        fold = fold_info['fold']
        t_end   = pd.Timestamp(fold_info['train_end'])
        v_start = pd.Timestamp(fold_info['val_start'])
        v_end   = pd.Timestamp(fold_info['val_end'])
        te_start= pd.Timestamp(fold_info['test_start'])
        te_end  = pd.Timestamp(fold_info['test_end'])

        for h in HORIZONS:
            t0 = time.time()

            # Create h-step-ahead log price target (within each market)
            df_h = df_crop.copy()
            df_h['target'] = (df_h.groupby('market')['log_price']
                                  .shift(-h))

            # Drop rows where target is NaN (last h rows per market)
            df_h = df_h.dropna(subset=['target'] + fcols[:5])  # quick NaN filter

            # Split
            train = df_h[df_h['week_start'] <= t_end]
            val   = df_h[(df_h['week_start'] > v_start) & (df_h['week_start'] <= v_end)]
            test  = df_h[(df_h['week_start'] >= te_start) & (df_h['week_start'] <= te_end)]

            if len(train) < 100 or len(test) < 10:
                continue

            X_tr, y_tr = train[fcols].fillna(0), train['target']
            X_va, y_va = val[fcols].fillna(0),   val['target']
            X_te, y_te = test[fcols].fillna(0),  test['target']

            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])

            y_pred = model.predict(X_te)
            m = compute_metrics(y_te.values, y_pred)
            elapsed = round(time.time() - t0, 1)

            row = {'crop': crop, 'fold': fold, 'horizon_weeks': h,
                   'test_year': te_end.year, **m, 'fit_sec': elapsed}
            all_metrics.append(row)

            print(f'   {crop:7s} fold{fold} h={h:>2}w | '
                  f'RMSE={m["RMSE"]:>7.1f}  MAPE={m["MAPE"]:>5.1f}%  '
                  f'R²={m["R2"]:>6.3f}  N={m["N"]:>6,}  [{elapsed}s]')

print(f'\n   Total training time: {(time.time()-t0_total)/60:.1f} min')

metrics_df = pd.DataFrame(all_metrics)
metrics_path = os.path.join(OUT_DIR, 'table_rolling_origin_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f'\n   Saved: {metrics_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. PRICE-SPIKE PROBABILITY MODEL
# ══════════════════════════════════════════════════════════════════════════════
print('\n[4] Price-spike probability model (h=1, >90th pct) ...')

spike_metrics = []
spike_roc     = {}   # for plot

for crop in CROPS:
    df_crop = feat[crop].copy()
    fcols   = get_feature_cols(df_crop, MACRO_COLS, include_macro=True)

    for fold_info in FOLDS:
        fold    = fold_info['fold']
        t_end   = pd.Timestamp(fold_info['train_end'])
        v_start = pd.Timestamp(fold_info['val_start'])
        v_end   = pd.Timestamp(fold_info['val_end'])
        te_start= pd.Timestamp(fold_info['test_start'])
        te_end  = pd.Timestamp(fold_info['test_end'])

        # h=1 spike target
        df_h = df_crop.copy()
        df_h['next_price'] = df_h.groupby('market')['modal_price_weighted'].shift(-1)
        df_h = df_h.dropna(subset=['next_price'])

        train = df_h[df_h['week_start'] <= t_end]
        val   = df_h[(df_h['week_start'] > v_start) & (df_h['week_start'] <= v_end)]
        test  = df_h[(df_h['week_start'] >= te_start) & (df_h['week_start'] <= te_end)]

        # Threshold on TRAIN prices
        threshold = np.percentile(train['modal_price_weighted'].dropna(), SPIKE_PCT)

        train = train.copy(); train['spike'] = (train['next_price'] > threshold).astype(int)
        val   = val.copy();   val['spike']   = (val['next_price']   > threshold).astype(int)
        test  = test.copy();  test['spike']  = (test['next_price']  > threshold).astype(int)

        if train['spike'].sum() < 10:
            continue

        X_tr, y_tr = train[fcols].fillna(0), train['spike']
        X_va, y_va = val[fcols].fillna(0),   val['spike']
        X_te, y_te = test[fcols].fillna(0),  test['spike']

        clf = lgb.LGBMClassifier(**LGBM_CLF_PARAMS)
        clf.fit(X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(40, verbose=False),
                           lgb.log_evaluation(-1)])

        proba = clf.predict_proba(X_te)[:, 1]
        auc   = round(roc_auc_score(y_te, proba), 4) if y_te.sum() > 0 else np.nan
        fpr, tpr, _ = roc_curve(y_te, proba) if y_te.sum() > 0 else ([0,1],[0,1],[])

        spike_metrics.append({'crop': crop, 'fold': fold,
                              'test_year': te_end.year,
                              'threshold_Rs_q': round(threshold, 1),
                              'spike_rate_pct': round(y_te.mean()*100, 1),
                              'AUC': auc})
        spike_roc.setdefault(crop, []).append((fpr, tpr, auc, fold))
        print(f'   {crop:7s} fold{fold} | threshold={threshold:.0f}  '
              f'spike_rate={y_te.mean()*100:.1f}%  AUC={auc:.4f}')

spike_df = pd.DataFrame(spike_metrics)
spike_path = os.path.join(OUT_DIR, 'table_spike_auc.csv')
spike_df.to_csv(spike_path, index=False)
print(f'\n   Saved: {spike_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY TABLE — averaged across folds
# ══════════════════════════════════════════════════════════════════════════════
print('\n[5] Summary (mean across folds) ...')
print()

summary = (metrics_df
           .groupby(['crop','horizon_weeks'])
           [['RMSE','MAE','MAPE','R2']]
           .mean().round({'RMSE':1,'MAE':1,'MAPE':2,'R2':4})
           .reset_index())

for crop in CROPS:
    sub = summary[summary['crop']==crop]
    print(f'  {crop.upper()}')
    print(f"  {'h':>4}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>7}  {'R²':>7}")
    print(f"  {'-'*40}")
    for _, r in sub.iterrows():
        print(f"  {int(r.horizon_weeks):>4}w  {r.RMSE:>8.1f}  {r.MAE:>8.1f}  "
              f"{r.MAPE:>6.1f}%  {r.R2:>7.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print('[6] Generating figures ...')

# ── Fig A: R² by horizon ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, crop in zip(axes, CROPS):
    sub = summary[summary['crop']==crop]
    ax.plot(sub['horizon_weeks'], sub['R2'], 'o-',
            color=CROP_COLORS[crop], linewidth=2, markersize=8, label='Mean R²')

    # Per-fold dots
    for fold_info in FOLDS:
        fold = fold_info['fold']
        fsub = metrics_df[(metrics_df['crop']==crop) & (metrics_df['fold']==fold)]
        ax.scatter(fsub['horizon_weeks'], fsub['R2'],
                   color=CROP_COLORS[crop], alpha=0.35, s=30, zorder=3)

    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([f'h={h}w' for h in HORIZONS])
    ax.set_xlabel('Forecast horizon')
    ax.set_ylabel('R² (back-transformed)')
    ax.set_title(crop.capitalize(), fontsize=11, fontweight='bold',
                 color=CROP_COLORS[crop])
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=8)

fig.suptitle('LightGBM Forecast Skill (R²) vs Horizon — Rolling-Origin CV (3 folds)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_horizon_r2.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'   Saved: {p}')

# ── Fig B: RMSE per fold × horizon (heatmap) ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, crop in zip(axes, CROPS):
    pivot = (metrics_df[metrics_df['crop']==crop]
             .pivot_table(index='fold', columns='horizon_weeks', values='RMSE'))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'h={h}w' for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'Fold {f}' for f in pivot.index])
    ax.set_title(crop.capitalize(), fontsize=10, fontweight='bold',
                 color=CROP_COLORS[crop])
    plt.colorbar(im, ax=ax, label='RMSE (Rs/q)')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=8, color='black')

fig.suptitle('RMSE Heatmap — Fold × Horizon (Rs/quintal)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_rolling_origin_rmse.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'   Saved: {p}')

# ── Fig C: ROC curves for spike model ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, crop in zip(axes, CROPS):
    roc_list = spike_roc.get(crop, [])
    for fpr, tpr, auc, fold in roc_list:
        ax.plot(fpr, tpr, linewidth=1.5, alpha=0.8,
                color=CROP_COLORS[crop],
                label=f'Fold {fold} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'--',color='grey',linewidth=0.8)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{crop.capitalize()}\nSpike >P{SPIKE_PCT}',
                 fontsize=10, fontweight='bold', color=CROP_COLORS[crop])
    ax.legend(fontsize=8)

fig.suptitle('Price-Spike Probability Model — ROC Curves (Rolling-Origin CV)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_spike_roc.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'   Saved: {p}')

# ── Fig D: MAPE by horizon bar chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x  = np.arange(len(HORIZONS))
w  = 0.25
for i, crop in enumerate(CROPS):
    sub  = summary[summary['crop']==crop].set_index('horizon_weeks')
    mape = [sub.loc[h, 'MAPE'] if h in sub.index else 0 for h in HORIZONS]
    bars = ax.bar(x + i*w, mape, w, label=crop.capitalize(),
                  color=CROP_COLORS[crop], alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, mape):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x + w); ax.set_xticklabels([f'h={h}w' for h in HORIZONS])
ax.set_ylabel('MAPE (%) — mean across folds')
ax.set_xlabel('Forecast horizon')
ax.set_title('Mean Absolute Percentage Error by Horizon — Rolling-Origin CV',
             fontsize=11, fontweight='bold')
ax.legend()
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_mape_by_horizon.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'   Saved: {p}')

# ══════════════════════════════════════════════════════════════════════════════
# 8. DONE
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('Script 12 complete.')
print(f'Total elapsed: {(time.time()-t0_total)/60:.1f} min')
print()
print('Key outputs:')
for f in ['table_rolling_origin_metrics.csv','table_spike_auc.csv',
          'fig_horizon_r2.png','fig_rolling_origin_rmse.png',
          'fig_spike_roc.png','fig_mape_by_horizon.png']:
    p = os.path.join(OUT_DIR, f)
    if os.path.exists(p):
        print(f'  {f:<42} {os.path.getsize(p)/1024:>7.1f} KB')
print()
print('Next: run Script 13 (benchmark models) or Script 14 (ERA5 processing)')
