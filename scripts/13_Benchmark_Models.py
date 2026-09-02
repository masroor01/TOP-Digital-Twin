# -*- coding: utf-8 -*-
"""
Script 13 — Benchmark Models: Naive Persistence, Seasonal Naive, ARIMA
=======================================================================
Computes three classical baselines against the same rolling-origin folds
and horizons as Script 12, so results are directly comparable.

Benchmarks
----------
B1  Naive persistence    : P̂_{t+h} = P_t  (current price, all horizons)
B2  Seasonal naive       : P̂_{t+h} = P_{t-52}  (same week last year)
B3  4-week moving avg    : P̂_{t+h} = mean(P_{t-4}...P_{t-1})
B4  ARIMA(1,1,1)         : fitted per market, h=1 only (rolling-origin)
                           h=4,13,26 via recursive multi-step forecast

Scope
-----
B1–B3 : per-market evaluation, all 4 horizons, all 3 folds
B4    : per-market ARIMA(1,1,1), h=1 all folds; h=4,13,26 national-level
        (per-market ARIMA at long horizons is computationally prohibitive
        for 2,495 markets — noted as limitation in Methods)
        STATUS: the h=4,13,26 national-level variants described above are
        NOT implemented in this script (still only h=1 runs below) — this
        remains an open gap versus the docstring, not just a Methods
        footnote. The h=1 forecast itself was date-misaligned (compared
        against the wrong test weeks by position) until the fix noted at
        the ARIMA forecast call below.

Outputs (Model_Output/)
-----------------------
  table_benchmarks.csv          — all benchmark metrics (crop×model×horizon×fold)
  table_comparison.csv          — LightGBM vs best benchmark, mean across folds
  fig_benchmark_comparison.png  — MAPE bar chart: LightGBM vs B1–B4
  fig_skill_score.png           — Skill score vs naive: (MAPE_naive - MAPE_lgbm)/MAPE_naive

Run: python scripts/13_Benchmark_Models.py
"""

import io, os, sys, time, warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
LGB_FILE = os.path.join(BASE, 'Model_Output', 'table_rolling_origin_metrics.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED     = 42

FOLDS = [
    {'fold': 1, 'train_end': '2021-06-30',
     'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'fold': 2, 'train_end': '2022-06-30',
     'test_start': '2023-01-01', 'test_end': '2023-12-31'},
    {'fold': 3, 'train_end': '2023-06-30',
     'test_start': '2024-01-01', 'test_end': '2024-12-31'},
]

CROP_COLORS = {'tomato': '#E63946', 'onion': '#F4A261', 'potato': '#457B9D'}

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE
# ══════════════════════════════════════════════════════════════════════════════
print('='*65)
print('SCRIPT 13 — BENCHMARK MODELS')
print('='*65)

print('\n[1] Loading weekly panel ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2024-12-31')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)
print(f'    {len(df):,} rows | {df["market"].nunique()} markets')


# ══════════════════════════════════════════════════════════════════════════════
# 2. METRICS HELPER
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, label):
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(model=label, RMSE=np.nan, MAE=np.nan, MAPE=np.nan, R2=np.nan, N=0)
    rmse   = np.sqrt(mean_squared_error(yt, yp))
    mae    = mean_absolute_error(yt, yp)
    mape   = np.mean(np.abs((yt - yp) / yt)) * 100
    ss_res = np.sum((yt - yp)**2)
    ss_tot = np.sum((yt - yt.mean())**2)
    r2     = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return dict(model=label, RMSE=round(rmse,1), MAE=round(mae,1),
                MAPE=round(mape,2), R2=round(r2,4), N=int(len(yt)))


# ══════════════════════════════════════════════════════════════════════════════
# 3. B1–B3: NAIVE BENCHMARKS (per-market, all horizons, all folds)
# ══════════════════════════════════════════════════════════════════════════════
print('\n[2] Computing naive benchmarks (B1–B3) ...')

all_bench = []
t0 = time.time()

for crop in CROPS:
    sub = df[df['crop'] == crop].copy()

    # Build per-market lookup: (market, market_id, week) → price. Grouped/
    # sorted on market_id (not the 'market' NAME) below -- a few market
    # names repeat across different states (e.g. "Fatehabad APMC" in both
    # Haryana and Uttar Pradesh), so grouping by name would interleave two
    # physically different markets' price series into one shift/rolling
    # computation. Matches the fix in Script 15 / Script 12.
    sub = sub.set_index(['market', 'market_id', 'week_start'])['modal_price_weighted'].to_frame()
    sub = sub.reset_index().sort_values(['market_id', 'week_start'])

    # Create shifted columns within each market
    sub['p_lag52'] = sub.groupby('market_id')['modal_price_weighted'].shift(52)
    sub['p_ma4']   = (sub.groupby('market_id')['modal_price_weighted']
                         .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean()))

    for fold_info in FOLDS:
        fold     = fold_info['fold']
        te_start = pd.Timestamp(fold_info['test_start'])
        te_end   = pd.Timestamp(fold_info['test_end'])

        for h in HORIZONS:
            # True future price (h weeks ahead)
            sub_h = sub.copy()
            sub_h['y_true'] = sub_h.groupby('market_id')['modal_price_weighted'].shift(-h)
            test = sub_h[(sub_h['week_start'] >= te_start) &
                         (sub_h['week_start'] <= te_end)].dropna(subset=['y_true'])

            yt = test['y_true'].values

            # B1: Naive persistence — use the origin-week price P_t itself
            # (test row's own 'modal_price_weighted' IS the price as of the
            # forecast's as-of week_start). FIXED: this previously used
            # p_lag1 (P_{t-1}, shift(1)), which staled the naive baseline
            # by one week and inflated every other model's reported
            # "skill vs naive".
            yp_naive   = test['modal_price_weighted'].values   # P_t, current/origin price
            # B2: Seasonal naive — same week last year
            yp_seasonal= test['p_lag52'].values
            # B3: 4-week moving average
            yp_ma4     = test['p_ma4'].values

            for label, yp in [('B1_Naive',    yp_naive),
                               ('B2_Seasonal', yp_seasonal),
                               ('B3_MA4',      yp_ma4)]:
                m = compute_metrics(yt, yp, label)
                m.update({'crop': crop, 'fold': fold,
                          'horizon_weeks': h, 'test_year': te_end.year})
                all_bench.append(m)

        print(f'    {crop:8s} fold {fold} — naive benchmarks done')

print(f'    Elapsed: {time.time()-t0:.1f}s')


# ══════════════════════════════════════════════════════════════════════════════
# 4. B4: ARIMA(1,1,1) — per market, batch forecast (fit once per fold)
#    Uses integer index (no date index) to avoid statsmodels frequency warnings.
#    Forecast: fit on train → forecast len(test) steps → compare to test.
# ══════════════════════════════════════════════════════════════════════════════
print('\n[3] Fitting ARIMA(1,1,1) per market (h=1 batch forecast, all folds) ...')

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    import warnings as _w
    HAS_ARIMA = True
except ImportError:
    print('    statsmodels not installed — skipping.')
    HAS_ARIMA = False

if HAS_ARIMA:
    arima_results = []
    t0 = time.time()
    MIN_OBS = 52

    for crop in CROPS:
        sub = (df[df['crop'] == crop]
               .sort_values(['market_id', 'week_start'])
               [['market', 'market_id', 'week_start', 'modal_price_weighted']])
        # Grouped/iterated by market_id, not the 'market' NAME -- see fix
        # note in section 3 above (same name-collision issue applies here).
        markets = sub['market_id'].unique()
        print(f'\n    {crop.upper()} — {len(markets)} markets')

        for fold_info in FOLDS:
            fold     = fold_info['fold']
            t_end    = pd.Timestamp(fold_info['train_end'])
            te_start = pd.Timestamp(fold_info['test_start'])
            te_end   = pd.Timestamp(fold_info['test_end'])

            y_true_all, y_pred_all = [], []
            skipped = 0

            for mkt in markets:
                mkt_df   = sub[sub['market_id'] == mkt].sort_values('week_start')
                train_df = mkt_df[mkt_df['week_start'] <= t_end].dropna(subset=['modal_price_weighted'])
                test_df  = mkt_df[(mkt_df['week_start'] >= te_start) &
                                   (mkt_df['week_start'] <= te_end)].dropna(subset=['modal_price_weighted'])
                train_vals = train_df['modal_price_weighted'].values

                if len(train_vals) < MIN_OBS or len(test_df) == 0:
                    skipped += 1
                    continue

                try:
                    log_train = np.log1p(np.clip(train_vals, 1, None))
                    with _w.catch_warnings():
                        _w.simplefilter('ignore')
                        mdl = _ARIMA(log_train, order=(1, 1, 1)).fit()
                        # FIXED: forecast now spans the FULL gap from the
                        # last training observation through the end of the
                        # test window (there's a ~26-week gap between
                        # train_end and te_start per the fold definition),
                        # then is aligned to the actual test dates below --
                        # previously `steps=len(test_vals)` forecast the
                        # weeks immediately after train_end (not the real
                        # test-year window) and was compared to test-year
                        # actuals purely by POSITION, pairing forecast-for-
                        # week-X against actual-price-for-a-different-week.
                        last_train_date = train_df['week_start'].max()
                        n_steps = int(round(
                            (test_df['week_start'].max() - last_train_date).days / 7))
                        fc_log = mdl.forecast(steps=n_steps)
                    fc_dates  = pd.date_range(last_train_date + pd.Timedelta(weeks=1),
                                               periods=n_steps, freq='7D')
                    fc_series = pd.Series(np.expm1(np.array(fc_log)), index=fc_dates)
                    # Slice by actual DATE, not position
                    aligned = fc_series.reindex(test_df['week_start'].values)
                    valid   = aligned.notna().values
                    if valid.sum() == 0:
                        skipped += 1
                        continue
                    y_true_all.extend(test_df['modal_price_weighted'].values[valid].tolist())
                    y_pred_all.extend(aligned.values[valid].tolist())
                except Exception as e:
                    skipped += 1
                    print(f'    ARIMA skip for {crop} fold{fold} market_id={mkt}: {e}')

            if y_true_all:
                m = compute_metrics(np.array(y_true_all), np.array(y_pred_all), 'B4_ARIMA')
                m.update({'crop': crop, 'fold': fold,
                          'horizon_weeks': 1, 'test_year': te_end.year})
                arima_results.append(m)
                print(f'    {crop:8s} fold {fold} | RMSE={m["RMSE"]:>7.1f}  '
                      f'MAPE={m["MAPE"]:>5.1f}%  R²={m["R2"]:>6.3f}  '
                      f'N={m["N"]:>7,}  skipped={skipped}')

    all_bench.extend(arima_results)
    print(f'\n    ARIMA total time: {(time.time()-t0)/60:.1f} min')


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE BENCHMARK TABLE
# ══════════════════════════════════════════════════════════════════════════════
bench_df = pd.DataFrame(all_bench)
bench_path = os.path.join(OUT_DIR, 'table_benchmarks.csv')
bench_df.to_csv(bench_path, index=False)
print(f'\n[4] Saved: {bench_path}  ({len(bench_df)} rows)')


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPARISON TABLE: LightGBM vs best benchmark
# ══════════════════════════════════════════════════════════════════════════════
print('\n[5] Building comparison table ...')

lgbm_df = pd.read_csv(LGB_FILE)

# Mean across folds for each crop × horizon
lgbm_mean = (lgbm_df
             .groupby(['crop','horizon_weeks'])[['RMSE','MAE','MAPE','R2']]
             .mean().round({'RMSE':1,'MAE':1,'MAPE':2,'R2':4})
             .reset_index()
             .assign(model='LightGBM'))

bench_mean = (bench_df
              .groupby(['crop','model','horizon_weeks'])[['RMSE','MAE','MAPE','R2']]
              .mean().round({'RMSE':1,'MAE':1,'MAPE':2,'R2':4})
              .reset_index())

comparison = pd.concat([lgbm_mean, bench_mean], ignore_index=True)
comp_path  = os.path.join(OUT_DIR, 'table_comparison.csv')
comparison.to_csv(comp_path, index=False)

# Print side-by-side for h=1 (primary horizon)
print(f'\n  h=1 week comparison (mean across folds):')
print(f'  {"Crop":8s} {"Model":14s} {"RMSE":>8s} {"MAPE":>8s} {"R²":>8s}')
print(f'  {"-"*52}')
h1 = comparison[comparison['horizon_weeks']==1].sort_values(['crop','model'])
for _, r in h1.iterrows():
    print(f'  {r.crop:8s} {r.model:14s} {r.RMSE:>8.1f} {r.MAPE:>7.1f}% {r.R2:>8.4f}')

# Skill score: how much better is LightGBM vs naive?
print(f'\n  Skill score vs B1_Naive (h=1): (MAPE_naive - MAPE_lgbm) / MAPE_naive × 100')
for crop in CROPS:
    lgbm_mape  = lgbm_mean[(lgbm_mean['crop']==crop) &
                            (lgbm_mean['horizon_weeks']==1)]['MAPE'].values
    naive_mape = bench_mean[(bench_mean['crop']==crop) &
                             (bench_mean['model']=='B1_Naive') &
                             (bench_mean['horizon_weeks']==1)]['MAPE'].values
    if len(lgbm_mape) and len(naive_mape):
        skill = (naive_mape[0] - lgbm_mape[0]) / naive_mape[0] * 100
        print(f'  {crop:8s}: naive MAPE={naive_mape[0]:.1f}%  '
              f'LightGBM MAPE={lgbm_mape[0]:.1f}%  '
              f'skill={skill:+.1f}%')


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print('\n[6] Generating figures ...')

MODELS_ORDERED = ['B1_Naive','B2_Seasonal','B3_MA4','B4_ARIMA','LightGBM']
MODEL_COLORS   = {
    'B1_Naive':   '#BDBDBD',
    'B2_Seasonal':'#90A4AE',
    'B3_MA4':     '#78909C',
    'B4_ARIMA':   '#546E7A',
    'LightGBM':   '#E63946',
}
MODEL_LABELS = {
    'B1_Naive':   'B1: Naive',
    'B2_Seasonal':'B2: Seasonal naive',
    'B3_MA4':     'B3: 4-wk MA',
    'B4_ARIMA':   'B4: ARIMA(1,1,1)',
    'LightGBM':   'LightGBM (ours)',
}

# ── Fig A: MAPE bar chart — all models × horizons, one panel per crop ────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

for ax, crop in zip(axes, CROPS):
    models_present = [m for m in MODELS_ORDERED if m in comparison['model'].unique()]
    x  = np.arange(len(HORIZONS))
    w  = 0.8 / len(models_present)

    for i, model in enumerate(models_present):
        sub = comparison[(comparison['crop']==crop) &
                         (comparison['model']==model)].set_index('horizon_weeks')
        mape = [sub.loc[h,'MAPE'] if h in sub.index else np.nan for h in HORIZONS]
        offset = (i - len(models_present)/2 + 0.5) * w
        bars = ax.bar(x + offset, mape, w,
                      label=MODEL_LABELS.get(model, model),
                      color=MODEL_COLORS.get(model, '#888'),
                      alpha=0.9, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, mape):
            if not np.isnan(v) and v < 120:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.5,
                        f'{v:.0f}%', ha='center', va='bottom',
                        fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f'h={h}w' for h in HORIZONS])
    ax.set_ylabel('MAPE (%) — mean across folds')
    ax.set_title(crop.capitalize(), fontsize=11, fontweight='bold',
                 color=CROP_COLORS[crop], loc='left')
    ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Benchmark Comparison — MAPE by Model and Horizon (Rolling-Origin CV)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_benchmark_comparison.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'    Saved: {p}')

# ── Fig B: Skill score vs naive (h=1) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

models_to_show = [m for m in MODELS_ORDERED if m != 'B1_Naive'
                  and m in comparison['model'].unique()]
x = np.arange(len(CROPS))
w = 0.7 / len(models_to_show)

for i, model in enumerate(models_to_show):
    skills = []
    for crop in CROPS:
        naive_mape = bench_mean[(bench_mean['crop']==crop) &
                                (bench_mean['model']=='B1_Naive') &
                                (bench_mean['horizon_weeks']==1)]['MAPE'].values
        mdl_mape   = comparison[(comparison['crop']==crop) &
                                 (comparison['model']==model) &
                                 (comparison['horizon_weeks']==1)]['MAPE'].values
        if len(naive_mape) and len(mdl_mape):
            skills.append((naive_mape[0] - mdl_mape[0]) / naive_mape[0] * 100)
        else:
            skills.append(np.nan)

    offset = (i - len(models_to_show)/2 + 0.5) * w
    bars = ax.bar(x + offset, skills, w,
                  label=MODEL_LABELS.get(model, model),
                  color=MODEL_COLORS.get(model, '#888'),
                  alpha=0.9, edgecolor='white')
    for bar, v in zip(bars, skills):
        if not np.isnan(v):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + (0.5 if v >= 0 else -2),
                    f'{v:+.1f}%', ha='center', va='bottom', fontsize=8.5)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([c.capitalize() for c in CROPS], fontsize=11)
ax.set_ylabel('Skill score vs naive (h=1 week)\n(positive = better than naive)')
ax.set_title('Forecast Skill Improvement over Naive Persistence — h=1 Week',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_skill_score.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'    Saved: {p}')

# ── Fig C: R² comparison table heatmap ───────────────────────────────────────
h1_pivot = (comparison[comparison['horizon_weeks']==1]
            .pivot_table(index='model', columns='crop', values='R2'))
h1_pivot = h1_pivot.reindex([m for m in MODELS_ORDERED if m in h1_pivot.index])

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(h1_pivot.values.astype(float), cmap='RdYlGn',
               vmin=-0.2, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(h1_pivot.columns)))
ax.set_xticklabels([c.capitalize() for c in h1_pivot.columns], fontsize=10)
ax.set_yticks(range(len(h1_pivot.index)))
ax.set_yticklabels([MODEL_LABELS.get(m,m) for m in h1_pivot.index], fontsize=9)
for i in range(len(h1_pivot.index)):
    for j in range(len(h1_pivot.columns)):
        v = h1_pivot.values[i,j]
        if not np.isnan(float(v)):
            ax.text(j, i, f'{float(v):.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if float(v) < 0.3 else 'black')
plt.colorbar(im, ax=ax, label='R²', shrink=0.8)
ax.set_title('R² Comparison — All Models, h=1 Week (mean across folds)',
             fontsize=10, fontweight='bold')
plt.tight_layout()
p = os.path.join(OUT_DIR, 'fig_r2_comparison_heatmap.png')
plt.savefig(p, dpi=200, bbox_inches='tight')
plt.close()
print(f'    Saved: {p}')


# ══════════════════════════════════════════════════════════════════════════════
# 8. DONE
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('Script 13 complete.')
total_time = time.time() - t0
print(f'Total time: {(time.time()-t0)/60:.1f} min')
print()
for f in ['table_benchmarks.csv','table_comparison.csv',
          'fig_benchmark_comparison.png','fig_skill_score.png',
          'fig_r2_comparison_heatmap.png']:
    p = os.path.join(OUT_DIR, f)
    if os.path.exists(p):
        print(f'  {f:<44} {os.path.getsize(p)/1024:>7.1f} KB')
print()
print('Next: Script 14 — ERA5 climate feature processing (Layer 3)')
