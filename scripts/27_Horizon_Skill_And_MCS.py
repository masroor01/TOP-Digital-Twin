# -*- coding: utf-8 -*-
"""
Script 27 — Horizon-Conditional Skill Table & Model Confidence Set
=====================================================================
Two additions to the model-evaluation toolkit, both aimed at answering:
"naive persistence beats every model at short horizons — where, if
anywhere, do the added data layers (M0-M6) actually earn their keep,
and is that difference statistically real rather than a small-sample
artifact of a single point-estimate comparison?"

Part A — Horizon-conditional skill table
  Reshapes the per-(crop, horizon, variant) MASE already computed by
  Script 15 into a skill-score view: skill% = (1 - MASE) * 100, i.e.
  the % reduction in MAE relative to naive persistence (positive =
  model beats naive). Reports the "crossover horizon" per crop: the
  shortest horizon at which the headline model (M6) first beats naive.

Part B — Model Confidence Set (Hansen, Lunde & Nason, 2011)
  A point-estimate MASE < 1 does not by itself mean a model is
  *significantly* better than naive — MASE is silent on sampling
  uncertainty. The MCS procedure formalizes this: starting from the
  full set of models {B1_Naive, M0..M6} at each (crop, horizon), it
  repeatedly performs a bootstrap equal-predictive-ability test on
  squared-error loss and eliminates the worst performer whenever the
  test rejects, until the surviving set can no longer be distinguished
  from the best model at the chosen confidence level. Unlike Script 18's
  pairwise Diebold-Mariano tests (one comparison at a time, no multiple-
  testing control), MCS tests all 8 models jointly and controls the
  familywise error rate — the right tool for "which of these models
  are we statistically confident belong in the winners' circle."

  Loss: squared error (same convention as Script 18's DM tests, for
  direct comparability). Bootstrap: stationary bootstrap (Politis &
  Romano, 1994) with expected block length = max(h, 2), matching the
  MA(h-1) forecast-error dependence structure Script 18 already uses
  for its Newey-West-style variance correction. Statistic: range
  statistic T_R = max_i(t_i) - min_i(t_i) on excess loss relative to
  the cross-sectional mean of the surviving set, bootstrapped B=1000
  times. Two confidence levels reported, both from Hansen et al.'s
  conventional choices: the 90% MCS (alpha=0.10) and the stricter 75%
  MCS (alpha=0.25, which rejects equal-ability more readily and so
  eliminates more models, producing a smaller "best models only" set
  nested inside the 90% MCS).

Inputs:
  Model_Output/ablation_predictions.csv   (Script 15)
  Model_Output/table_mase.csv             (Script 15)

Outputs (Model_Output/):
  table_horizon_skill.csv    skill% by crop x horizon x variant, crossover horizons
  fig_horizon_skill.png      skill% vs horizon, one line per crop (M6), naive = 0 line
  table_mcs.csv              full elimination sequence + MCS_10/MCS_25 membership
  fig_mcs_membership.png     grid: which models survive in the 90%/75% MCS, by crop x horizon

Run: python scripts/27_Horizon_Skill_And_MCS.py
Estimated runtime: <1 minute (bootstrap on already-computed predictions, no model fitting)
"""

import io, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_FILE = os.path.join(BASE, 'Model_Output', 'ablation_predictions.csv')
MASE_FILE = os.path.join(BASE, 'Model_Output', 'table_mase.csv')
OUT_DIR   = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS      = ['tomato', 'onion', 'potato']
HORIZONS   = [1, 4, 13, 26]
ALL_MODELS = ['B1_Naive', 'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']

CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}

MCS_B       = 1000
MCS_ALPHAS  = [0.10, 0.25]
MCS_SEED    = 20260730

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('=' * 65)
print('SCRIPT 27: HORIZON-CONDITIONAL SKILL TABLE & MODEL CONFIDENCE SET')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# PART A — HORIZON-CONDITIONAL SKILL TABLE
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Part A] Horizon-conditional skill table ...\n')

if not os.path.exists(MASE_FILE):
    print(f'ERROR: {MASE_FILE} not found. Run scripts/15_Ablation_Study_M0_M4.py first.')
    sys.exit(1)

mase = pd.read_csv(MASE_FILE)
mase['skill_pct'] = (1 - mase['MASE']) * 100  # positive = model beats naive

skill_table = mase.pivot_table(index=['crop', 'horizon_weeks'], columns='variant',
                                values='skill_pct').reset_index()
skill_table = skill_table[['crop', 'horizon_weeks'] + [m for m in ALL_MODELS if m in skill_table.columns]]
skill_table = skill_table.sort_values(['crop', 'horizon_weeks'])

# Crossover horizon: shortest horizon where M6 skill_pct > 0 (beats naive)
crossover_rows = []
for crop in CROPS:
    sub = mase[(mase['crop'] == crop) & (mase['variant'] == 'M6')].sort_values('horizon_weeks')
    if sub.empty:
        crossover_rows.append({'crop': crop, 'M6_crossover_horizon_weeks': None})
        print(f'  {crop:7s}: M6 crossover horizon = '
              f'no data available for this crop -- crossover horizon cannot be determined')
        continue
    beats = sub[sub['skill_pct'] > 0]
    crossover_h = int(beats['horizon_weeks'].iloc[0]) if not beats.empty else None
    crossover_rows.append({'crop': crop, 'M6_crossover_horizon_weeks': crossover_h})
    print(f'  {crop:7s}: M6 crossover horizon = '
          f'{f"{crossover_h}w" if crossover_h else "never (naive wins at all horizons tested)"}')

crossover = pd.DataFrame(crossover_rows)

table_path = os.path.join(OUT_DIR, 'table_horizon_skill.csv')
skill_table.to_csv(table_path, index=False)
print(f'\n  Saved: {table_path}  ({len(skill_table)} rows)')

crossover_path = os.path.join(OUT_DIR, 'table_horizon_skill_crossover.csv')
crossover.to_csv(crossover_path, index=False)
print(f'  Saved: {crossover_path}')

print('\n  M6 skill% by crop x horizon (positive = beats naive persistence):')
m6_view = mase[mase['variant'] == 'M6'].pivot(index='crop', columns='horizon_weeks', values='skill_pct')
m6_view = m6_view.reindex(index=CROPS, columns=HORIZONS)
print(m6_view.round(1).to_string())

# Figure: skill% vs horizon, one line per crop, for M6
fig, ax = plt.subplots(figsize=(7, 5))
for crop in CROPS:
    sub = mase[(mase['crop'] == crop) & (mase['variant'] == 'M6')].sort_values('horizon_weeks')
    ax.plot(sub['horizon_weeks'], sub['skill_pct'], marker='o', linewidth=2,
            color=CROP_COLORS[crop], label=crop.capitalize())
ax.axhline(0, color='black', linewidth=1, linestyle='--', label='Naive persistence (0% skill)')
ax.set_xticks(HORIZONS)
ax.set_xlabel('Forecast horizon (weeks)')
ax.set_ylabel('Skill over naive persistence (%)\n[(1 - MASE) × 100, MAE basis]')
ax.set_title('M6 (full pipeline) skill over naive persistence, by horizon',
              fontsize=11, fontweight='bold')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_horizon_skill.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'\n  Saved: {fig_path}')


# ─────────────────────────────────────────────────────────────────────────────
# PART B — MODEL CONFIDENCE SET (Hansen, Lunde & Nason, 2011)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Part B] Model Confidence Set (range statistic, stationary bootstrap) ...\n')

if not os.path.exists(PRED_FILE):
    print(f'ERROR: {PRED_FILE} not found. Run scripts/15_Ablation_Study_M0_M4.py first.')
    sys.exit(1)

preds = pd.read_csv(PRED_FILE, parse_dates=['week_start'])
preds = preds.sort_values(['crop', 'horizon_weeks', 'variant', 'fold', 'week_start'])


def stationary_bootstrap_index(T, avg_block_len, rng):
    """One stationary-bootstrap (Politis & Romano, 1994) resampled index path
    of length T, expected block length avg_block_len, wrapping circularly."""
    p = 1.0 / max(avg_block_len, 1)
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(0, T)
    for t in range(1, T):
        if rng.random() < p:
            idx[t] = rng.integers(0, T)
        else:
            idx[t] = (idx[t - 1] + 1) % T
    return idx


def mcs_elimination_path(loss_df, avg_block_len, B=MCS_B, seed=MCS_SEED):
    """
    Full Hansen-Lunde-Nason elimination sequence down to 1 surviving model.
    loss_df: T x M dataframe of per-period losses (lower = better), one
    column per model. Returns a list of dicts, one per elimination round:
    {eliminated, p_value, remaining_before_elimination}. Models never
    "eliminated" (the last one standing) get eliminated=None with the
    round's own p-value (always 1.0, trivially -- no test possible on 1 model).
    """
    models = list(loss_df.columns)
    L = loss_df.values.astype(float)
    T = L.shape[0]
    rng = np.random.default_rng(seed)
    boot_idx = np.array([stationary_bootstrap_index(T, avg_block_len, rng) for _ in range(B)])

    current = list(range(len(models)))
    path = []
    while len(current) > 1:
        sub_L = L[:, current]
        Lbar_t = sub_L.mean(axis=1)
        d = sub_L - Lbar_t[:, None]           # excess loss vs cross-sectional mean
        d_bar = d.mean(axis=0)

        d_bar_boot = np.empty((B, len(current)))
        for b in range(B):
            d_bar_boot[b] = d[boot_idx[b], :].mean(axis=0)

        se = d_bar_boot.std(axis=0, ddof=1)
        se = np.where(se < 1e-12, 1e-12, se)
        t_stat = d_bar / se
        t_stat_boot = (d_bar_boot - d_bar[None, :]) / se[None, :]

        T_R_obs  = t_stat.max() - t_stat.min()
        T_R_boot = t_stat_boot.max(axis=1) - t_stat_boot.min(axis=1)
        p_value  = float((T_R_boot >= T_R_obs).mean())

        worst_local = int(np.argmax(t_stat))
        worst_model = models[current[worst_local]]
        path.append({'eliminated': worst_model, 'p_value': round(p_value, 4),
                      'n_remaining_before': len(current)})
        current.pop(worst_local)

    path.append({'eliminated': models[current[0]], 'p_value': 1.0,
                  'n_remaining_before': 1})
    return path


def mcs_membership(path, model_names, alpha):
    """Given the full elimination path, the MCS at level alpha is the set of
    models NOT YET eliminated at the point the test first fails to reject
    (first p_value > alpha encountered going down the path)."""
    eliminated_by_then = set()
    for step in path:
        if step['p_value'] > alpha:
            break
        eliminated_by_then.add(step['eliminated'])
    return [m for m in model_names if m not in eliminated_by_then]


mcs_rows = []
membership_rows = []
for crop in CROPS:
    for h in HORIZONS:
        sub = preds[(preds['crop'] == crop) & (preds['horizon_weeks'] == h)]
        if sub.empty:
            continue

        series = {}
        for variant in ALL_MODELS:
            vsub = (sub[sub['variant'] == variant]
                    .sort_values(['fold', 'week_start'])
                    .drop_duplicates(subset='week_start'))
            series[variant] = vsub.set_index('week_start')

        common_weeks = None
        for variant in ALL_MODELS:
            weeks = set(series[variant].index)
            common_weeks = weeks if common_weeks is None else (common_weeks & weeks)
        common_weeks = sorted(common_weeks)
        if len(common_weeks) < 20:
            print(f'  {crop:7s} h={h:>2}w  SKIPPED (only {len(common_weeks)} common weeks)')
            continue

        loss_df = pd.DataFrame(index=common_weeks)
        for variant in ALL_MODELS:
            s = series[variant].loc[common_weeks]
            loss_df[variant] = (s['y_true'] - s['y_pred']) ** 2   # squared-error loss

        avg_block_len = max(h, 2)
        path = mcs_elimination_path(loss_df, avg_block_len)

        for i, step in enumerate(path):
            mcs_rows.append({
                'crop': crop, 'horizon_weeks': h, 'elimination_round': i + 1,
                'eliminated_model': step['eliminated'], 'p_value': step['p_value'],
                'n_models_before_this_round': step['n_remaining_before'],
            })

        mcs10 = mcs_membership(path, ALL_MODELS, 0.10)
        mcs25 = mcs_membership(path, ALL_MODELS, 0.25)
        for m in ALL_MODELS:
            membership_rows.append({
                'crop': crop, 'horizon_weeks': h, 'model': m,
                'in_mcs_10pct': m in mcs10, 'in_mcs_25pct': m in mcs25,
            })

        print(f'  {crop:7s} h={h:>2}w  n={len(common_weeks):>3}  '
              f'MCS(10%)={",".join(mcs10):<40s} MCS(25%)={",".join(mcs25)}')

mcs_path_df = pd.DataFrame(mcs_rows)
mcs_membership_df = pd.DataFrame(membership_rows)

mcs_path_table = os.path.join(OUT_DIR, 'table_mcs.csv')
mcs_path_df.to_csv(mcs_path_table, index=False)
print(f'\n  Saved: {mcs_path_table}  ({len(mcs_path_df)} rows)')

mcs_membership_table = os.path.join(OUT_DIR, 'table_mcs_membership.csv')
mcs_membership_df.to_csv(mcs_membership_table, index=False)
print(f'  Saved: {mcs_membership_table}  ({len(mcs_membership_df)} rows)')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE — MCS membership grid (crop x horizon panels, models on y-axis)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Part B] Generating MCS membership figure ...')

fig, axes = plt.subplots(1, len(CROPS), figsize=(15, 5), sharey=True)
for ax, crop in zip(axes, CROPS):
    grid10 = np.zeros((len(ALL_MODELS), len(HORIZONS)))
    grid25 = np.zeros((len(ALL_MODELS), len(HORIZONS)))
    for j, h in enumerate(HORIZONS):
        sub = mcs_membership_df[(mcs_membership_df['crop'] == crop) &
                                 (mcs_membership_df['horizon_weeks'] == h)]
        for i, m in enumerate(ALL_MODELS):
            row = sub[sub['model'] == m]
            if row.empty:
                continue
            grid10[i, j] = 1 if row['in_mcs_10pct'].iloc[0] else 0
            grid25[i, j] = 1 if row['in_mcs_25pct'].iloc[0] else 0

    # 25% membership = light shade, 10% membership = full shade (10% implies 25%)
    display_grid = grid25 * 0.45 + grid10 * 0.55
    im = ax.imshow(display_grid, cmap='Greens', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f'{h}w' for h in HORIZONS])
    ax.set_yticks(range(len(ALL_MODELS)))
    ax.set_yticklabels(ALL_MODELS)
    ax.set_title(crop.capitalize(), fontsize=11, fontweight='bold', color=CROP_COLORS[crop])
    ax.set_xlabel('Horizon')
    for i in range(len(ALL_MODELS)):
        for j in range(len(HORIZONS)):
            if grid10[i, j]:
                ax.text(j, i, '●', ha='center', va='center', fontsize=9, color='white')
            elif grid25[i, j]:
                ax.text(j, i, '○', ha='center', va='center', fontsize=9, color='#1E5C37')

axes[0].set_ylabel('Model')
fig.suptitle('Model Confidence Set membership  (● = in 90% MCS,  ○ = in 75% MCS only,  '
              'blank = eliminated at both levels)', fontsize=10, y=1.02)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_mcs_membership.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

print('\n' + '=' * 65)
print('Script 27 complete.')
print('\nKey outputs:')
for fname in ['table_horizon_skill.csv', 'table_horizon_skill_crossover.csv',
              'fig_horizon_skill.png', 'table_mcs.csv', 'table_mcs_membership.csv',
              'fig_mcs_membership.png']:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        print(f'  {fname:<35} {os.path.getsize(fpath)/1024:>7.1f} KB')
