# -*- coding: utf-8 -*-
"""
Script 25 — Horizon-Stratified SHAP Analysis
=================================================
Explains WHY the M0-M6 ablation study and market-level DM tests show
crop/horizon-dependent value from the richer data layers (Script 15/18b
findings): computes SHAP feature importance for each of the 12 saved
production models (Script 23), grouped by data layer (Price, Arrivals,
Macro, Climate, Satellite, Infrastructure, Policy), and shows how the
composition shifts across horizons (h=1, 4, 13, 26 weeks).

Expected pattern, if the ablation study's story is real and not noise:
price/arrival features should dominate at h=1w (short-horizon momentum),
with macro/climate/policy features gaining share at longer horizons
(h=13w, h=26w) where price persistence has decayed — most visible for
onion (where M6 won robustly) and long-horizon tomato.

Uses TreeExplainer (shap library) on the Script 23 production models,
on a sample of real historical feature vectors per crop (not just the
single latest-week reference row) for a representative distribution.

Outputs (Model_Output/):
  table_shap_by_layer.csv         crop x horizon x layer: mean|SHAP|, % of total
  table_shap_top_features.csv     crop x horizon: top 15 individual features
  fig_shap_layer_composition.png  headline figure: layer share by horizon, per crop
  fig_shap_top_features.png       top-10 features at h=1w vs h=26w, per crop
  fig_shap_beeswarm_onion_4w.png  illustrative detail plot for the headline result

Run: python scripts/25_Horizon_SHAP_Analysis.py
"""

import io, os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
MODEL_DIR  = os.path.join(BASE, 'Model_Output', 'production_models')
OUT_DIR    = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED     = 42
SAMPLE_SIZE = 3000   # rows per crop for SHAP computation (from real history, not just latest week)

LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS = [4, 8, 13]

CROP_COLORS = {'tomato': '#E63946', 'onion': '#F4A261', 'potato': '#457B9D'}
LAYER_COLORS = {
    'Price': '#adb5bd', 'Arrivals': '#74c0fc', 'Macro': '#51cf66',
    'Climate': '#ff922b', 'Satellite': '#cc5de8', 'Infrastructure': '#20c997',
    'Policy': '#e64980',
}
LAYER_ORDER = ['Price', 'Arrivals', 'Macro', 'Climate', 'Satellite', 'Infrastructure', 'Policy']

print('=' * 65)
print('SCRIPT 25: HORIZON-STRATIFIED SHAP ANALYSIS')
print('=' * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD + JOIN ALL LAYERS (identical to Scripts 15/23)
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
# 2. FEATURE ENGINEERING (identical to Scripts 15/23)
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

# Feature -> layer lookup, for aggregating SHAP importance
LAYER_MAP = {}
for c in PRICE_FEATS:  LAYER_MAP[c] = 'Price'
for c in ARR_FEATS:    LAYER_MAP[c] = 'Arrivals'
for c in MACRO_COLS:   LAYER_MAP[c] = 'Macro'
for c in CLIMATE_FEATS: LAYER_MAP[c] = 'Climate'
for c in SAT_FEATS:    LAYER_MAP[c] = 'Satellite'
for c in INFRA_FEATS:  LAYER_MAP[c] = 'Infrastructure'
for c in POLICY_FEATS: LAYER_MAP[c] = 'Policy'

print(f'  Layer sizes: ' + ', '.join(
    f'{layer}={sum(1 for v in LAYER_MAP.values() if v == layer)}' for layer in LAYER_ORDER))


# ─────────────────────────────────────────────────────────────────────────────
# 3. SHAP COMPUTATION PER (crop, horizon)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Computing SHAP values per (crop, horizon) ...\n')

with open(os.path.join(MODEL_DIR, 'feature_columns.json'), encoding='utf-8') as f:
    feature_columns = json.load(f)

rng = np.random.RandomState(SEED)
layer_rows = []
top_feat_rows = []
shap_cache = {}  # (crop, h) -> (X_sample, shap_values) for the beeswarm plot

for crop in CROPS:
    df_crop = feat[crop]
    fcols = feature_columns.get(f'{crop}_1w')  # column order is the same across horizons
    if fcols is None:
        print(f'  WARNING: no saved model found for {crop}, skipping')
        continue

    # Require valid price_lag_1 (matches the training universe filter)
    valid = df_crop.dropna(subset=['price_lag_1'])
    n = min(SAMPLE_SIZE, len(valid))
    sample_idx = rng.choice(valid.index, size=n, replace=False)
    X_sample = valid.loc[sample_idx, fcols].fillna(0)

    for h in HORIZONS:
        model_path = os.path.join(MODEL_DIR, f'{crop}_{h}w.joblib')
        if not os.path.exists(model_path):
            print(f'  WARNING: missing model {model_path}, skipping')
            continue
        model = joblib.load(model_path)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        mean_abs = np.abs(shap_values).mean(axis=0)
        total = mean_abs.sum()

        feat_imp = pd.DataFrame({'feature': fcols, 'mean_abs_shap': mean_abs})
        feat_imp['layer'] = feat_imp['feature'].map(LAYER_MAP).fillna('Other')
        feat_imp['pct_of_total'] = 100 * feat_imp['mean_abs_shap'] / total if total > 0 else 0

        by_layer = feat_imp.groupby('layer')['mean_abs_shap'].sum().reset_index()
        by_layer['pct_of_total'] = 100 * by_layer['mean_abs_shap'] / total if total > 0 else 0
        by_layer['crop'] = crop
        by_layer['horizon_weeks'] = h
        layer_rows.append(by_layer)

        top15 = feat_imp.sort_values('mean_abs_shap', ascending=False).head(15).copy()
        top15['crop'] = crop
        top15['horizon_weeks'] = h
        top15['rank'] = range(1, len(top15) + 1)
        top_feat_rows.append(top15)

        if (crop, h) in [('onion', 4), ('tomato', 1), ('tomato', 26)]:
            shap_cache[(crop, h)] = (X_sample, shap_values)

        top1 = feat_imp.sort_values('mean_abs_shap', ascending=False).iloc[0]
        print(f'  {crop:7s} h={h:>2}w | top feature: {top1["feature"]:<22s} '
              f'({top1["pct_of_total"]:.1f}% of total importance)')

layer_df = pd.concat(layer_rows, ignore_index=True)
top_feat_df = pd.concat(top_feat_rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE TABLES
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Saving tables ...')
layer_path = os.path.join(OUT_DIR, 'table_shap_by_layer.csv')
layer_df.to_csv(layer_path, index=False, encoding='utf-8')
print(f'  Saved: {layer_path}')

top_feat_path = os.path.join(OUT_DIR, 'table_shap_top_features.csv')
top_feat_df.to_csv(top_feat_path, index=False, encoding='utf-8')
print(f'  Saved: {top_feat_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE A — layer composition by horizon (headline figure)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Generating figures ...')

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, crop in zip(axes, CROPS):
    sub = layer_df[layer_df['crop'] == crop]
    pivot = sub.pivot_table(index='horizon_weeks', columns='layer', values='pct_of_total', fill_value=0)
    pivot = pivot.reindex(columns=[l for l in LAYER_ORDER if l in pivot.columns])
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot.index))
    for layer in pivot.columns:
        ax.bar(x, pivot[layer].values, bottom=bottom, label=layer,
               color=LAYER_COLORS.get(layer, '#888'), width=0.6)
        bottom += pivot[layer].values
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}w' for h in pivot.index])
    ax.set_title(crop.capitalize(), fontweight='bold', color=CROP_COLORS[crop])
    ax.set_xlabel('Forecast horizon')
    if crop == CROPS[0]:
        ax.set_ylabel('Share of total SHAP importance (%)')
    ax.grid(axis='y', alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(LAYER_ORDER), bbox_to_anchor=(0.5, -0.05))
plt.suptitle('Data-Layer Importance by Forecast Horizon (SHAP, M6 production models)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_shap_layer_composition.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE B — top features, h=1w vs h=26w, per crop
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col_i, crop in enumerate(CROPS):
    for row_i, h in enumerate([1, 26]):
        ax = axes[row_i][col_i]
        sub = (top_feat_df[(top_feat_df['crop'] == crop) & (top_feat_df['horizon_weeks'] == h)]
               .sort_values('mean_abs_shap').tail(10))
        colors = [LAYER_COLORS.get(l, '#888') for l in sub['layer']]
        ax.barh(sub['feature'], sub['mean_abs_shap'], color=colors)
        ax.set_title(f'{crop.capitalize()}  h={h}w', fontsize=10, fontweight='bold')
        ax.set_xlabel('mean |SHAP value|')
        ax.tick_params(axis='y', labelsize=8)

from matplotlib.patches import Patch
legend_els = [Patch(color=c, label=l) for l, c in LAYER_COLORS.items()]
fig.legend(handles=legend_els, loc='lower center', ncol=len(LAYER_ORDER), bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Top 10 Features: Short Horizon (h=1w) vs Long Horizon (h=26w)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_shap_top_features.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE C — beeswarm detail for the headline case (onion h=4w)
# ─────────────────────────────────────────────────────────────────────────────
if ('onion', 4) in shap_cache:
    X_sample, shap_values = shap_cache[('onion', 4)]
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
    plt.title('Onion, h=4w — SHAP Summary (the study\'s strongest M6 result)', fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig_shap_beeswarm_onion_4w.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {p}')

print('\n' + '=' * 65)
print('Script 25 complete.')
print('\nKey outputs:')
for fname in ['table_shap_by_layer.csv', 'table_shap_top_features.csv',
              'fig_shap_layer_composition.png', 'fig_shap_top_features.png',
              'fig_shap_beeswarm_onion_4w.png']:
    fp = os.path.join(OUT_DIR, fname)
    if os.path.exists(fp):
        print(f'  {fname:<38} {os.path.getsize(fp)/1024:>7.1f} KB')
