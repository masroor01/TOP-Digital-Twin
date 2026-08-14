# -*- coding: utf-8 -*-
"""
Script 29 — Granger Causality Analysis
=========================================
The ablation study (Script 15) and SHAP analysis (Script 25) show which
data layers improve LightGBM's forecast accuracy, and by how much. Neither
answers a related but distinct econometric question: does a given driver
series (arrivals, climate, satellite, macro, policy) have statistically
significant predictive content for future price movements beyond what
price's own history already provides -- i.e., does it Granger-cause price
(Granger, 1969)? This script tests that directly, layer by layer, plus a
light market-network extension testing lead-lag relationships among each
crop's five highest-volume markets.

Method: for each crop, national weekly aggregate series (arrivals-weighted-
median real price, total real arrivals, climate/satellite/macro/policy
covariates) are tested for a unit root (Augmented Dickey-Fuller); any
series with a unit root is first-differenced (log-differenced for price
and arrivals) before testing, since Granger causality on non-stationary
series is spurious. Bidirectional pairwise Granger tests (driver->price
and price->driver, via statsmodels' F-test on a VAR system) are run at
three lags matching this study's forecast horizons in weeks that are
tractable for a weekly-frequency VAR (1, 4, 13 -- 26 weeks of lag would
leave too few effective degrees of freedom given ~500 weeks of data and
is not tested). P-values are Benjamini-Hochberg FDR-corrected within each
crop, across all driver x direction x lag tests, to control the false
discovery rate given the large number of comparisons.

Part B repeats a much narrower version of the same test among each crop's
five highest-volume markets' own weekly real prices, to check for
"bellwether" lead-lag relationships -- a market whose price move
systematically precedes another's.

Inputs:
  data/agmarknet_weekly/top_weekly_panel.csv
  data/satellite_climate/crop_weekly_features.csv
  data/rbi_dbie/rbi_dbie_macro_2017_2025.csv
  data/ppac_macro/ppac_diesel_lpg_2017_2025.csv
  data/policy_trade/policy_weekly_features.csv

Outputs (Model_Output/):
  table_granger_layers.csv          driver x direction x lag Granger test results (Part A)
  fig_granger_layers.png            -log10(p_fdr) heatmap, driver x crop, at lag=13
  table_granger_market_network.csv  top-5-market pairwise lead-lag results (Part B)
  fig_granger_market_network.png    adjacency heatmap of -log10(p_fdr) per crop, lag=1

Run: python scripts/29_Granger_Causality_Analysis.py
Estimated runtime: <2 minutes (statistical tests only, no model fitting)
"""

import io, os, sys, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE  = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
SAT_FILE  = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
RBI_FILE  = os.path.join(BASE, 'data', 'rbi_dbie', 'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE = os.path.join(BASE, 'data', 'ppac_macro', 'ppac_diesel_lpg_2017_2025.csv')
POL_FILE  = os.path.join(BASE, 'data', 'policy_trade', 'policy_weekly_features.csv')
OUT_DIR   = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS = ['tomato', 'onion', 'potato']
LAGS_TO_TEST = [1, 4, 13]
MAX_LAG = 13
FDR_ALPHA = 0.05
CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}
WPI_COL = {'tomato': 'wpi_tomato', 'onion': 'wpi_onion', 'potato': 'wpi_potato'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('=' * 65)
print('SCRIPT 29: GRANGER CAUSALITY ANALYSIS')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD NATIONAL WEEKLY AGGREGATE SERIES PER CROP
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Building national weekly aggregate series per crop ...')

panel = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
panel_real = panel[panel['imputed'] == 0]
agg = (panel_real.groupby(['crop', 'week_start'])
       .agg(price=('modal_price_weighted', 'median'),
            arrivals=('arrivals_tonnes_week', 'sum'))
       .reset_index())

sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'])
pol = pd.read_csv(POL_FILE, parse_dates=['week_start'])

rbi = pd.read_csv(RBI_FILE)
ppac = pd.read_csv(PPAC_FILE, parse_dates=['date'])
ppac['year'], ppac['month'] = ppac['date'].dt.year, ppac['date'].dt.month
macro = rbi.merge(ppac[['year', 'month', 'diesel_delhi_per_L']], on=['year', 'month'], how='outer')

series_by_crop = {}
for crop in CROPS:
    df = agg[agg['crop'] == crop].sort_values('week_start').set_index('week_start')
    df = df.join(sat[sat['crop'] == crop].set_index('week_start').drop(columns='crop'), how='left')
    df = df.join(pol[pol['crop'] == crop].set_index('week_start').drop(columns='crop'), how='left')
    df['year'], df['month'] = df.index.year, df.index.month
    df = df.reset_index().merge(macro, on=['year', 'month'], how='left').set_index('week_start')
    df = df.rename(columns={WPI_COL[crop]: 'wpi_own_crop'})
    df['log_price'] = np.log(df['price'])
    df['log_arrivals'] = np.log(df['arrivals'].clip(lower=1))
    series_by_crop[crop] = df.sort_index()
    print(f'  {crop:7s}: {len(df)} weeks, {df.index.min().date()} to {df.index.max().date()}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATIONARITY HANDLING
# ─────────────────────────────────────────────────────────────────────────────
def make_stationary(s, name):
    """Returns a stationary version of series s (first-differenced if ADF
    fails to reject a unit root at 5%), plus a flag recording what was done."""
    s = s.dropna()
    if len(s) < 30:
        return None, 'insufficient_data'
    try:
        p = adfuller(s, autolag='AIC')[1]
    except Exception:
        return None, 'adf_failed'
    if p < 0.05:
        return s, 'level'
    d = s.diff().dropna()
    if d.std() < 1e-8:
        return None, 'degenerate_after_diff'
    return d, 'diff'


# ─────────────────────────────────────────────────────────────────────────────
# 4. PART A — LAYER-LEVEL GRANGER CAUSALITY
# ─────────────────────────────────────────────────────────────────────────────
DRIVERS = {
    'arrivals':           'log_arrivals',
    'climate_tmax':       'era5_tmax',
    'climate_rain':       'chirps_rain_mm',
    'satellite_ndvi_anom':'s2_ndvi_anom',
    'macro_usdinr':       'usdinr_monthly_avg',
    'macro_repo_rate':    'repo_rate_pct',
    'macro_diesel':       'diesel_delhi_per_L',
    'macro_wpi_own_crop': 'wpi_own_crop',
    'policy_export_duty': 'export_duty_pct',
    'policy_mep':         'mep_usd_per_tonne',
    'policy_export_ban':  'export_banned',
    'policy_intervention':'market_intervention_flag',
}

print('\n[2] Part A: layer-level Granger causality (bidirectional, per driver) ...\n')

def run_granger_pair(y_stat, x_stat, lags):
    """y_stat Granger-caused BY x_stat. Returns {lag: (F, p)}."""
    merged = pd.concat([y_stat, x_stat], axis=1).dropna()
    merged.columns = ['y', 'x']
    if len(merged) < max(lags) + 15:
        return None
    try:
        res = grangercausalitytests(merged[['y', 'x']], maxlag=max(lags), verbose=False)
    except Exception:
        return None
    out = {}
    for lag in lags:
        if lag in res:
            f, p = res[lag][0]['ssr_ftest'][0], res[lag][0]['ssr_ftest'][1]
            out[lag] = (float(f), float(p))
    return out


rows_a = []
for crop in CROPS:
    df = series_by_crop[crop]
    price_stat, price_note = make_stationary(df['log_price'], 'log_price')
    if price_stat is None:
        print(f'  {crop}: SKIPPED -- price series not usable ({price_note})')
        continue
    print(f'  {crop}: price series -> {price_note} (ADF-stationary after this transform)')

    for driver_key, col in DRIVERS.items():
        if col not in df.columns or df[col].notna().sum() < 30:
            continue
        raw = df[col]
        if raw.nunique() <= 1:
            continue  # degenerate (e.g., no policy variation for this crop)
        x_stat, x_note = make_stationary(raw, driver_key)
        if x_stat is None:
            continue

        fwd = run_granger_pair(price_stat, x_stat, LAGS_TO_TEST)   # driver -> price
        rev = run_granger_pair(x_stat, price_stat, LAGS_TO_TEST)   # price -> driver
        for direction, result in [('driver_causes_price', fwd), ('price_causes_driver', rev)]:
            if result is None:
                continue
            for lag, (f, p) in result.items():
                rows_a.append({
                    'crop': crop, 'driver': driver_key, 'direction': direction,
                    'lag_weeks': lag, 'F_stat': round(f, 3), 'p_value': round(p, 4),
                    'transform': x_note,
                })

table_a = pd.DataFrame(rows_a)
# FDR correction within each crop, across all driver x direction x lag tests
table_a['p_value_fdr'] = np.nan
table_a['significant_fdr05'] = False
for crop in CROPS:
    mask = table_a['crop'] == crop
    if mask.sum() == 0:
        continue
    rej, p_adj, _, _ = multipletests(table_a.loc[mask, 'p_value'], alpha=FDR_ALPHA, method='fdr_bh')
    table_a.loc[mask, 'p_value_fdr'] = p_adj
    table_a.loc[mask, 'significant_fdr05'] = rej

table_a_path = os.path.join(OUT_DIR, 'table_granger_layers.csv')
table_a.to_csv(table_a_path, index=False)
print(f'\n  Saved: {table_a_path}  ({len(table_a)} tests)')

sig = table_a[table_a['significant_fdr05']]
print(f'\n  {len(sig)}/{len(table_a)} tests significant after FDR correction (alpha=0.05):')
for crop in CROPS:
    csig = sig[sig['crop'] == crop].sort_values(['driver', 'direction', 'lag_weeks'])
    print(f'\n  {crop.upper()}:')
    if csig.empty:
        print('    none')
    for _, r in csig.iterrows():
        arrow = f"{r['driver']} -> price" if r['direction'] == 'driver_causes_price' else f"price -> {r['driver']}"
        print(f"    {arrow:<32s} lag={r['lag_weeks']:>2}w  F={r['F_stat']:>7.2f}  "
              f"p_fdr={r['p_value_fdr']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE A — heatmap of -log10(p_fdr), driver x crop, at lag=13
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Generating Part A figure ...')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, direction, title in zip(
        axes, ['driver_causes_price', 'price_causes_driver'],
        ['Driver -> Price', 'Price -> Driver (feedback)']):
    sub = table_a[(table_a['direction'] == direction) & (table_a['lag_weeks'] == 13)]
    pivot = sub.pivot_table(index='driver', columns='crop', values='p_value_fdr')
    pivot = pivot.reindex(index=list(DRIVERS.keys()), columns=CROPS)
    neglog = -np.log10(pivot.clip(lower=1e-10))
    im = ax.imshow(neglog.values, cmap='RdYlGn', vmin=0, vmax=4, aspect='auto')
    ax.set_xticks(range(len(CROPS)))
    ax.set_xticklabels([c.capitalize() for c in CROPS])
    ax.set_yticks(range(len(DRIVERS)))
    ax.set_yticklabels(list(DRIVERS.keys()), fontsize=8)
    ax.set_title(f'{title}\n(lag=13w, FDR-corrected -log10 p; blank = untestable)', fontsize=10, fontweight='bold')
    for i in range(len(DRIVERS)):
        for j in range(len(CROPS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, label='-log10(p_fdr)', fraction=0.046, pad=0.04)

plt.tight_layout()
fig_a_path = os.path.join(OUT_DIR, 'fig_granger_layers.png')
plt.savefig(fig_a_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_a_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. PART B — TOP-5-MARKET LEAD-LAG NETWORK
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Part B: top-5-market lead-lag network per crop ...\n')

NETWORK_LAGS = [1, 4]
rows_b = []

for crop in CROPS:
    sub = panel_real[panel_real['crop'] == crop]
    # FIXED 2026-08-14 (full-layer audit): selection/keying used to be by
    # 'market' NAME -- a few names repeat across states (e.g. "Fatehabad
    # APMC" in both Haryana and UP), so grouping/filtering by name risks
    # silently blending two different markets' arrivals/price series if a
    # colliding name happened to rank in the top 5. Selecting and keying by
    # market_id instead; labels below include state for readability and to
    # disambiguate any name that does repeat.
    id_to_label = sub.drop_duplicates('market_id').set_index('market_id').apply(
        lambda r: f"{r['market']} ({r['state']})" if 'state' in r else r['market'], axis=1)
    top5_ids = sub.groupby('market_id')['arrivals_tonnes_week'].mean().nlargest(5).index.tolist()
    print(f'  {crop:7s} top-5 markets by mean arrivals: {[id_to_label.get(i, i) for i in top5_ids]}')

    market_series = {}
    for mid in top5_ids:
        m_label = id_to_label.get(mid, str(mid))
        s = sub[sub['market_id'] == mid].sort_values('week_start').set_index('week_start')['modal_price_weighted']
        s = np.log(s)
        s_stat, note = make_stationary(s, m_label)
        if s_stat is not None:
            market_series[m_label] = s_stat

    pairs_tested = 0
    for m_from in market_series:
        for m_to in market_series:
            if m_from == m_to:
                continue
            result = run_granger_pair(market_series[m_to], market_series[m_from], NETWORK_LAGS)
            if result is None:
                continue
            pairs_tested += 1
            for lag, (f, p) in result.items():
                rows_b.append({
                    'crop': crop, 'market_from': m_from, 'market_to': m_to,
                    'lag_weeks': lag, 'F_stat': round(f, 3), 'p_value': round(p, 4),
                })
    print(f'    {pairs_tested} directed pairs tested')

table_b = pd.DataFrame(rows_b)
table_b['p_value_fdr'] = np.nan
table_b['significant_fdr05'] = False
for crop in CROPS:
    mask = table_b['crop'] == crop
    if mask.sum() == 0:
        continue
    rej, p_adj, _, _ = multipletests(table_b.loc[mask, 'p_value'], alpha=FDR_ALPHA, method='fdr_bh')
    table_b.loc[mask, 'p_value_fdr'] = p_adj
    table_b.loc[mask, 'significant_fdr05'] = rej

table_b_path = os.path.join(OUT_DIR, 'table_granger_market_network.csv')
table_b.to_csv(table_b_path, index=False)
print(f'\n  Saved: {table_b_path}  ({len(table_b)} tests)')

sig_b = table_b[table_b['significant_fdr05']]
print(f'  {len(sig_b)}/{len(table_b)} market-pair tests significant after FDR correction')
for _, r in sig_b.sort_values(['crop', 'lag_weeks']).iterrows():
    print(f"    {r['crop']:7s} {r['market_from']} -> {r['market_to']:<20s} lag={r['lag_weeks']}w  "
          f"F={r['F_stat']:.2f}  p_fdr={r['p_value_fdr']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE B — adjacency heatmap, lag=1, per crop
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Generating Part B figure ...')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, crop in zip(axes, CROPS):
    sub = table_b[(table_b['crop'] == crop) & (table_b['lag_weeks'] == 1)]
    if sub.empty:
        ax.set_title(f'{crop.capitalize()} (no data)')
        ax.axis('off')
        continue
    markets = sorted(set(sub['market_from']) | set(sub['market_to']))
    mat = np.full((len(markets), len(markets)), np.nan)
    for _, r in sub.iterrows():
        i, j = markets.index(r['market_from']), markets.index(r['market_to'])
        mat[i, j] = -np.log10(max(r['p_value_fdr'], 1e-10))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=4, aspect='auto')
    short_labels = [m[:14] for m in markets]
    ax.set_xticks(range(len(markets))); ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(markets))); ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_title(f'{crop.capitalize()} (row causes column, lag=1w)', fontsize=9,
                 fontweight='bold', color=CROP_COLORS[crop])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
fig_b_path = os.path.join(OUT_DIR, 'fig_granger_market_network.png')
plt.savefig(fig_b_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_b_path}')

print('\n' + '=' * 65)
print('Script 29 complete.')
print('\nKey outputs:')
for fname in ['table_granger_layers.csv', 'fig_granger_layers.png',
              'table_granger_market_network.csv', 'fig_granger_market_network.png']:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        print(f'  {fname:<38} {os.path.getsize(fpath)/1024:>7.1f} KB')
