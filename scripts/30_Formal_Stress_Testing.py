# -*- coding: utf-8 -*-
"""
Script 30 — Formal Stress-Testing Module
===========================================
The dashboard (Script 24) already lets a user manually drag sliders to
build one what-if scenario at a time. This script systematises that into
a fixed, reproducible battery of named stress scenarios, run in batch
across every market in the production models' reference set (not one
hand-picked market), using the same saved M6 production models -- no
retraining. Two scenario classes are tested:

  CALIBRATED REPLAYS -- scenario feature values set to the exact,
    verified policy-event values from Script 19's event log (export
    duty/MEP/ban levels actually notified during the 2023-24 onion
    crisis; Operation Greens/MIS activation as it happened for tomato)
    or the largest REAL historical move in a continuous driver (diesel
    price: +29.7%, the largest 12-month increase in the 2017-2026 PPAC
    series; climate: each crop's own 95th-percentile historical extreme
    from its satellite/climate zone). Because Script 28 already
    documents what actually happened to price during the onion and
    tomato policy episodes these replay, this script's model-implied
    response can be checked directly against the real outcome -- a
    genuine calibration check on whether the stress-testing module's
    output is realistic, not merely internally consistent.

  EXPLORATORY (potato only) -- a hypothetical 20% cold-storage-capacity
    reduction, included because potato has no verified policy-event
    regime to replay (Script 19's log has no potato-specific entries in
    the study period) and its Layer 5 infrastructure features are its
    most distinctive lever; explicitly labelled as uncalibrated.

For each scenario, every market's own reference feature row (Script 23's
`reference_rows.csv`) is used as the baseline; only the scenario's
specified columns are overridden, everything else (market/state
encoding, lags, unrelated covariates) is left at that market's real
latest value. The model-implied price response is the % change between
the shocked and baseline prediction, summarised as the median across
all of a crop's markets (with the 10th-90th percentile spread reported
alongside, since cross-market heterogeneity is itself informative).

Inputs:
  Model_Output/production_models/  (Script 23: models, feature_columns.json,
                                     reference_rows.csv)
  data/policy_trade/export_policy_events.csv  (Script 19, calibration source)
  data/ppac_macro/ppac_diesel_lpg_2017_2025.csv, data/satellite_climate/
    crop_weekly_features.csv  (calibration sources)

Outputs (Model_Output/):
  table_stress_test_results.csv   scenario x crop x horizon median/p10/p90 % response
  fig_stress_test_tomato.png
  fig_stress_test_onion.png
  fig_stress_test_potato.png

Run: python scripts/30_Formal_Stress_Testing.py
Estimated runtime: <1 minute (inference only, no model fitting)
"""

import io, os, sys, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, 'Model_Output', 'production_models')
OUT_DIR   = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('=' * 65)
print('SCRIPT 30: FORMAL STRESS-TESTING MODULE')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD PRODUCTION MODELS & REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_DIR):
    print(f'ERROR: {MODEL_DIR} not found. Run scripts/23_Train_Production_Models.py first.')
    sys.exit(1)

with open(os.path.join(MODEL_DIR, 'feature_columns.json'), encoding='utf-8') as f:
    feature_columns = json.load(f)
reference = pd.read_csv(os.path.join(MODEL_DIR, 'reference_rows.csv'))

models = {}
for crop in CROPS:
    for h in HORIZONS:
        path = os.path.join(MODEL_DIR, f'{crop}_{h}w.joblib')
        if os.path.exists(path):
            models[(crop, h)] = joblib.load(path)

print(f'\n  Loaded {len(models)} production models, {len(reference)} reference market rows')


def predict_batch(crop, h, overrides):
    """Predicts price for every market of `crop` under `overrides` (dict of
    {column: value} applied on top of that market's own reference row).
    Returns array of predicted prices, one per market."""
    cols = feature_columns[f'{crop}_{h}w']
    sub = reference[reference['crop'] == crop].copy()
    for col, val in overrides.items():
        if col in sub.columns:
            sub[col] = val
    X = sub.reindex(columns=cols, fill_value=0)
    log_pred = models[(crop, h)].predict(X)
    return np.expm1(log_pred)


def baseline_batch(crop, h):
    cols = feature_columns[f'{crop}_{h}w']
    sub = reference[reference['crop'] == crop]
    X = sub.reindex(columns=cols, fill_value=0)
    log_pred = models[(crop, h)].predict(X)
    return np.expm1(log_pred)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CALIBRATION VALUES (computed from real data, not invented)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Computing calibration values from real historical data ...')

ppac = pd.read_csv(os.path.join(BASE, 'data', 'ppac_macro', 'ppac_diesel_lpg_2017_2025.csv'), parse_dates=['date'])
ppac = ppac.sort_values('date')
diesel_shock_pct = float(ppac['diesel_delhi_per_L'].pct_change(12).max() * 100)  # largest real 12-month rise
print(f'  Diesel shock magnitude: +{diesel_shock_pct:.1f}% '
      f'(largest real 12-month increase in the 2017-2026 PPAC series)')

sat = pd.read_csv(os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv'))
climate_p95 = {}
for crop in CROPS:
    s = sat[sat['crop'] == crop]
    climate_p95[crop] = {
        'era5_tmax': float(s['era5_tmax'].quantile(0.95)),
        'era5_heat_35': float(s['era5_heat_35'].quantile(0.95)),
        'chirps_rain_mm': float(s['chirps_rain_mm'].quantile(0.95)),
        'chirps_excess': float(s['chirps_excess'].quantile(0.95)),
    }
    print(f"  {crop:7s} climate p95: tmax={climate_p95[crop]['era5_tmax']:.1f}C  "
          f"rain={climate_p95[crop]['chirps_rain_mm']:.0f}mm")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def climate_overrides(crop):
    c = climate_p95[crop]
    def fn(df):
        return {'era5_tmax': c['era5_tmax'], 'era5_heat_35': c['era5_heat_35'],
                'chirps_rain_mm': c['chirps_rain_mm'], 'chirps_excess': c['chirps_excess']}
    return fn


def diesel_shock_overrides(df):
    factor = 1 + diesel_shock_pct / 100
    return {'diesel_delhi_per_L': df['diesel_delhi_per_L'] * factor,
            'diesel_4city_rs_litre': df['diesel_4city_rs_litre'] * factor}


SCENARIOS = {
    'tomato': [
        {'name': 'diesel_shock', 'kind': 'calibrated',
         'desc': f'Diesel +{diesel_shock_pct:.0f}% (largest real 12-month rise, PPAC 2017-2026)',
         'overrides': diesel_shock_overrides, 'real_comparison': None},
        {'name': 'climate_extreme', 'kind': 'calibrated',
         'desc': "Heat + rainfall at tomato's own 95th-percentile zone-week extreme",
         'overrides': climate_overrides('tomato'), 'real_comparison': None},
        {'name': 'op_greens_active', 'kind': 'calibrated',
         'desc': 'Operation Greens + Market Intervention Scheme active (replays Jul 2023/2024 state)',
         'overrides': lambda df: {'operation_greens_active': 1, 'market_intervention_flag': 1},
         'real_comparison': 'Actual: Jun-Sep 2023 spike/crash episode saw tomato price move '
                             '+454%/-70% over 4 weeks around this intervention window (Script 28) -- '
                             'note the intervention was a REACTION to the spike, not its cause '
                             '(Script 29 confirms price Granger-causes this flag, not the reverse), '
                             'so this scenario should not be read as "if we flip this switch, price falls."'},
    ],
    'onion': [
        {'name': 'diesel_shock', 'kind': 'calibrated',
         'desc': f'Diesel +{diesel_shock_pct:.0f}% (largest real 12-month rise, PPAC 2017-2026)',
         'overrides': diesel_shock_overrides, 'real_comparison': None},
        {'name': 'climate_extreme', 'kind': 'calibrated',
         'desc': "Heat + rainfall at onion's own 95th-percentile zone-week extreme",
         'overrides': climate_overrides('onion'), 'real_comparison': None},
        {'name': 'export_duty_40pct', 'kind': 'calibrated',
         'desc': '40% export duty (replays the verified 19-Aug-2023 notification)',
         'overrides': lambda df: {'export_duty_pct': 40, 'export_banned': 0}, 'real_comparison': None},
        {'name': 'mep_800usd', 'kind': 'calibrated',
         'desc': 'MEP USD 800/MT (replays the verified 28-Oct-2023 notification)',
         'overrides': lambda df: {'mep_usd_per_tonne': 800, 'export_banned': 0}, 'real_comparison': None},
        {'name': 'export_ban', 'kind': 'calibrated',
         'desc': 'Export prohibited (replays the verified 8-Dec-2023 ban)',
         'overrides': lambda df: {'export_banned': 1}, 'real_comparison': None},
        {'name': 'full_crisis_replay', 'kind': 'calibrated',
         'desc': 'Duty 40% + MEP $800/MT + export ban, combined (the full Aug-Dec 2023 escalation state)',
         'overrides': lambda df: {'export_duty_pct': 40, 'mep_usd_per_tonne': 800, 'export_banned': 1},
         'real_comparison': 'Actual: the real Aug 2023-Jan 2024 escalation saw a +89% spike (late Oct '
                             '2023, before the ban) followed by a -46% crash within 3 weeks of the '
                             '8-Dec-2023 ban (Script 28) -- the ban itself is associated with the price '
                             'FALLING (its stated intent: "support domestic availability"), so a '
                             'well-calibrated model should predict a price decrease under this scenario, '
                             'not an increase.'},
    ],
    'potato': [
        {'name': 'diesel_shock', 'kind': 'calibrated',
         'desc': f'Diesel +{diesel_shock_pct:.0f}% (largest real 12-month rise, PPAC 2017-2026)',
         'overrides': diesel_shock_overrides, 'real_comparison': None},
        {'name': 'climate_extreme', 'kind': 'calibrated',
         'desc': "Heat + rainfall at potato's own 95th-percentile zone-week extreme",
         'overrides': climate_overrides('potato'), 'real_comparison': None},
        {'name': 'cold_storage_capacity_down20pct', 'kind': 'exploratory',
         'desc': '20% cold-storage-capacity reduction (HYPOTHETICAL -- no verified real event to '
                 'calibrate against; potato has no export/MEP policy regime in the study period, '
                 'Script 19)',
         'overrides': lambda df: {'cold_storage_capacity_mt': df['cold_storage_capacity_mt'] * 0.8},
         'real_comparison': None},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. RUN SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Running scenario battery ...\n')

rows = []
for crop in CROPS:
    ref_crop = reference[reference['crop'] == crop]
    if ref_crop.empty:
        continue
    base_medians = {}
    for scen in SCENARIOS[crop]:
        print(f"  [{crop}] {scen['name']:<32s} ({scen['kind']})  {scen['desc']}")
        for h in HORIZONS:
            if (crop, h) not in models:
                continue
            baseline_prices = baseline_batch(crop, h)
            # overrides may depend on each market's own baseline row (e.g. % shock on diesel);
            # applied vectorized across the whole crop's reference DataFrame
            sub = ref_crop.copy()
            cols = feature_columns[f'{crop}_{h}w']
            overrides = scen['overrides'](sub)
            for col, val in overrides.items():
                if col in sub.columns:
                    sub[col] = val
            X = sub.reindex(columns=cols, fill_value=0).apply(pd.to_numeric, errors='coerce').fillna(0)
            shocked_prices = np.expm1(models[(crop, h)].predict(X))

            pct_change = (shocked_prices - baseline_prices) / baseline_prices * 100
            rows.append({
                'crop': crop, 'scenario': scen['name'], 'kind': scen['kind'],
                'horizon_weeks': h, 'n_markets': len(sub),
                'median_pct_change': round(float(np.median(pct_change)), 1),
                'p10_pct_change': round(float(np.percentile(pct_change, 10)), 1),
                'p90_pct_change': round(float(np.percentile(pct_change, 90)), 1),
            })
        h1_row = [r for r in rows if r['crop'] == crop and r['scenario'] == scen['name'] and r['horizon_weeks'] == 1]
        if h1_row:
            print(f"      h=1w median response: {h1_row[0]['median_pct_change']:+.1f}% "
                  f"[{h1_row[0]['p10_pct_change']:+.1f}%, {h1_row[0]['p90_pct_change']:+.1f}%]")
        if scen['real_comparison']:
            print(f"      {scen['real_comparison']}")
        print()

table = pd.DataFrame(rows)
table_path = os.path.join(OUT_DIR, 'table_stress_test_results.csv')
table.to_csv(table_path, index=False)
print(f'Saved: {table_path}  ({len(table)} rows)')


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURES — one heatmap per crop, scenario x horizon
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Generating per-crop figures ...')

for crop in CROPS:
    sub = table[table['crop'] == crop]
    if sub.empty:
        continue
    scen_names = [s['name'] for s in SCENARIOS[crop]]
    pivot = sub.pivot_table(index='scenario', columns='horizon_weeks', values='median_pct_change')
    pivot = pivot.reindex(index=scen_names, columns=HORIZONS)

    fig, ax = plt.subplots(figsize=(7, 1.1 + 0.6 * len(scen_names)))
    vmax = max(1, np.nanmax(np.abs(pivot.values)))
    im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels([f'{h}w' for h in HORIZONS])
    ax.set_yticks(range(len(scen_names))); ax.set_yticklabels(scen_names, fontsize=8)
    for i in range(len(scen_names)):
        for j in range(len(HORIZONS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', fontsize=8)
    ax.set_title(f'{crop.capitalize()} -- model-implied median price response by scenario',
                 fontsize=10, fontweight='bold', color=CROP_COLORS[crop])
    plt.colorbar(im, ax=ax, label='% change vs baseline', fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f'fig_stress_test_{crop}.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fig_path}')

print('\n' + '=' * 65)
print('Script 30 complete.')
print('\nKey outputs:')
for fname in ['table_stress_test_results.csv', 'fig_stress_test_tomato.png',
              'fig_stress_test_onion.png', 'fig_stress_test_potato.png']:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        print(f'  {fname:<38} {os.path.getsize(fpath)/1024:>7.1f} KB')
