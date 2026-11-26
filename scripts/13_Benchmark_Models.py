# -*- coding: utf-8 -*-
"""
Script 13 — Benchmark models on the same folds and rows as the production model
=============================================================================
Corrected 2026-11-26. The earlier version (a) scored the classical methods on three test years (2022-2024) while the
model it was compared with (Script 15, M6) averages five (2022-2026); (b) fitted ARIMA once per fold and read the h=1
forecast off a path that ran up to about 18 months from the last training week; (c) scored the h=4/13/26 ARIMA against a
national median series, not market prices; and (d) used P_{t-52} as the "seasonal naive" forecast of t+h.
Earlier outputs are kept in Model_Output/_benchmarks_superseded_2026-11-26/.

RULES, fixed before the corrected run
-------------------------------------
Rows     : the rows Script 15 scores for M6: per market, target (price h rows ahead) and price_lag_1 present,
           origin week in the test year; test years 2022-2026; MAPE per fold, then the mean of the five folds.
B1       : no change, P_t.
B2       : seasonal naive, the price of week t+h-52 (known at the origin because h <= 26).
B3       : mean of the four latest weeks t-3..t (at least two present).
B4       : ARIMA(1,1,1) on log1p price, one per market; parameters estimated on the weeks up to the fold's train_end
           (at least 52 observed weeks); forecast h weeks ahead from EVERY origin with the state updated on the prices
           observed up to that origin (parameters not re-estimated; gaps carry the last price forward).
Fallback : where a method has no forecast for a row (new market, missing lag or origin price), it takes B1 for that
           row, so every method is scored on identical rows. The share of fallback rows is recorded.
Headline : in a crop-horizon cell M6 "beats the classical methods" if its five-fold mean MAPE is below that of the best
           of B1-B4. Folds in which M6 is below that same best method are reported beside it.

Outputs (Model_Output/): table_benchmarks.csv, table_comparison.csv, table_benchmark_headline.csv,
          fig_benchmark_comparison.png, fig_skill_score.png, fig_r2_comparison_heatmap.png
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
# 2026-11-06: the model rows now come from the re-run ablation (Script 15, variant M6 = production model, per fold); they used to come from
# table_rolling_origin_metrics.csv (Script 12), which pre-dates the validation-window, origin-price and data repairs.
LGB_FILE = os.path.join(BASE, 'Model_Output', 'ablation_raw_results.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
HORIZONS = [1, 4, 13, 26]
SEED     = 42

FOLDS = [{'fold': k + 1, 'train_end': f'{2021 + k}-06-30', 'test_start': f'{2022 + k}-01-01', 'test_end': f'{2022 + k}-12-31'}
         for k in range(5)]

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
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2026-12-31')]
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
# 3. B1-B4, per market, on the rows of the production model
# ══════════════════════════════════════════════════════════════════════════════
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
MIN_OBS = 52

rows, checked = {}, False
t0 = time.time()
for crop in CROPS:
    sub = df[df['crop'] == crop].sort_values(['market_id', 'week_start'])
    for n_mkt, (mid, g) in enumerate(sub.groupby('market_id', sort=False)):
        g = g.reset_index(drop=True)
        price = g['modal_price_weighted']
        weeks = pd.DatetimeIndex(g['week_start'])
        if price.notna().sum() == 0:
            continue
        grid = pd.date_range(weeks.min(), weeks.max(), freq='7D')
        full = pd.Series(price.values, index=weeks).reindex(grid)
        first_obs = full.first_valid_index()
        filled = full.ffill().bfill()                       # bfill only touches weeks before the first price
        y_log = np.log1p(np.clip(filled.values, 1, None))
        ma4 = full.rolling(4, min_periods=2).mean().reindex(weeks).values
        lag1 = price.shift(1)
        for fold_info in FOLDS:
            fold = fold_info['fold']
            te = (weeks >= pd.Timestamp(fold_info['test_start'])) & (weeks <= pd.Timestamp(fold_info['test_end']))
            if not te.any():
                continue
            fc = None
            train = full[full.index <= pd.Timestamp(fold_info['train_end'])].dropna()
            if len(train) >= MIN_OBS:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        mdl = _ARIMA(np.log1p(np.clip(train.values, 1, None)), order=(1, 1, 1)).fit()
                        upd = mdl.apply(y_log, refit=False)
                    phi, theta = float(mdl.params[0]), float(mdl.params[1])
                    resid = np.asarray(upd.resid, dtype=float).copy()
                    resid[0] = 0.0
                    dy = np.diff(y_log, prepend=y_log[0])
                    step = phi * dy + theta * resid
                    fc = {h: y_log + step * np.sum(phi ** np.arange(h)) for h in HORIZONS}
                    if not checked and len(y_log) > 300:        # check the closed form against statsmodels
                        for k in (len(y_log) - 40, len(y_log) - 90):
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                ref = mdl.apply(y_log[:k + 1], refit=False).forecast(13)[-1]
                            assert abs(ref - fc[13][k]) < 5e-3, (ref, fc[13][k])
                            print(f'    ARIMA closed form checked against statsmodels: {ref:.5f} vs {fc[13][k]:.5f}')
                        checked = True
                except AssertionError:
                    raise
                except Exception as e:
                    print(f'    ARIMA skip {crop} market_id={mid} fold {fold}: {e}')
            for h in HORIZONS:
                y_true = price.shift(-h)
                use = np.asarray(te) & y_true.notna().values & lag1.notna().values
                if not use.any():
                    continue
                w = weeks[use]
                naive = price.values[use].astype(float)
                no_origin = ~np.isfinite(naive)
                naive = np.where(no_origin, lag1.values[use], naive)   # origin price missing: last known (t-1)
                pred = {'B1_Naive': np.where(no_origin, np.nan, naive),
                        'B2_Seasonal': full.reindex(w + pd.Timedelta(weeks=h - 52)).values,
                        'B3_MA4': ma4[use]}
                if fc is not None:
                    a_ = np.expm1(fc[h][grid.get_indexer(w)])
                    a_[np.asarray(w < first_obs)] = np.nan
                    pred['B4_ARIMA'] = a_
                else:
                    pred['B4_ARIMA'] = np.full(int(use.sum()), np.nan)
                for label, yp in pred.items():
                    yp = np.asarray(yp, dtype=float)
                    miss = ~np.isfinite(yp)
                    yp = np.where(miss, naive, yp)
                    rows.setdefault((crop, label, h, fold), []).append((y_true.values[use], yp, miss))
        if n_mkt % 100 == 0:
            print(f'    {crop:8s} market {n_mkt:>4}  elapsed {(time.time() - t0) / 60:.1f} min', flush=True)

all_bench = []
for (crop, label, h, fold), parts in rows.items():
    yt = np.concatenate([p_[0] for p_ in parts]); yp = np.concatenate([p_[1] for p_ in parts]); miss = np.concatenate([p_[2] for p_ in parts])
    m = compute_metrics(yt, yp, label)
    m.update({'crop': crop, 'fold': fold, 'horizon_weeks': h, 'test_year': 2021 + fold, 'fallback_share': round(float(miss.mean()), 4)})
    all_bench.append(m)
print(f'    benchmarks done in {(time.time() - t0) / 60:.1f} min')


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
lgbm_df = lgbm_df[lgbm_df['variant'] == 'M6'].copy()

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

hl = []
for (crop, h), g in bench_df.groupby(['crop', 'horizon_weeks']):
    means = g.groupby('model').MAPE.mean()
    best = means.idxmin()
    m6 = lgbm_df[(lgbm_df.crop == crop) & (lgbm_df.horizon_weeks == h)].set_index('fold').MAPE
    bf = g[g.model == best].set_index('fold').MAPE
    assert len(m6) == 5 and len(bf) == 5, (crop, h, len(m6), len(bf))
    hl.append(dict(crop=crop, horizon_weeks=h, M6_MAPE=round(m6.mean(), 2), best_classical=best, best_MAPE=round(means.min(), 2),
                   M6_beats_best=bool(m6.mean() < means.min()), M6_folds_below_best=int((m6.loc[bf.index] < bf).sum())))
hl = pd.DataFrame(hl)
hl.to_csv(os.path.join(OUT_DIR, 'table_benchmark_headline.csv'), index=False)
print(hl.to_string(index=False))
print('  M6 beats the best classical method: %d of 12 cells; %d of 6 at 13 and 26 weeks' % (hl.M6_beats_best.sum(), hl[hl.horizon_weeks >= 13].M6_beats_best.sum()))

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
