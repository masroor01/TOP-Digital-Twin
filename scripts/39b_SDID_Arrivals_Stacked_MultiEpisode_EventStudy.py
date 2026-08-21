# -*- coding: utf-8 -*-
"""
Script 39b — Stacked Multi-Episode SDID Event Study, ARRIVALS Outcome
(Onion Export Bans 2019, 2020, 2023-24)
=============================================================================
Script 39 runs the stacked multi-episode SDID design on PRICE across all
three documented onion export bans, using tomato/potato at the same three
real dates as a placebo band -- the project's multi-episode analogue of a
placebo-in-time test (distinct from Script 38/38b's in-space permutation
test across markets). Script 38b then found that the single-episode
ARRIVALS-outcome postban ATT (Script 31 Part C.4) FAILS an in-space placebo
test decisively (p=0.680) despite looking directionally clean in point-
estimate form.

This script runs the arrivals-outcome analogue of Script 39: the identical
stacked multi-episode design, applied to log1p(arrivals) instead of
log(price), across all three onion export-ban episodes, to check whether
the arrivals effect looks any more like a genuine, event-time-aligned signal
once tested against a proper multi-episode placebo band -- the second and
final robustness check flagged as open in the results audit (2026-08-21).

Per-episode market qualification mirrors Script 31 Part C.4's arrivals logic
exactly, but re-evaluated on EACH episode's own window (coverage requirements
shift over time as Agmarknet reporting grows, same rationale as Script 39's
price-based per-episode requalification): among that episode's window's
price-qualifying markets (>=95% real price coverage, zero price gaps), keep
markets with >=95% arrivals coverage, linearly interpolate arrivals gaps
<=4 weeks, then require a fully rectangular matrix.

Market-name-vs-market_id collision fix applied throughout, consistent with
Scripts 31/38/38b/39.

Inputs:
  data/agmarknet_weekly/top_weekly_panel.csv

Outputs (Model_Output/):
  table_sdid_arrivals_stacked_event_study.csv   per-episode x role x weeks-since-ban dynamic ATT
  table_sdid_arrivals_stacked_summary.csv       stacked mean/spread per role x event-time bin
  fig_sdid_arrivals_stacked_event_study.png     all 3 episodes aligned, real vs placebo (arrivals)

Run: python scripts/39b_SDID_Arrivals_Stacked_MultiEpisode_EventStudy.py
Estimated runtime: ~3-5 minutes (9 SDID fits: 3 episodes x 3 units).
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

PANEL_START = pd.Timestamp('2017-01-02')
COVERAGE_THRESHOLD = 0.95
INTERP_LIMIT = 4
POST_WEEKS_FOR_FIT = 26
EVENT_WINDOW = (-20, 30)
MIN_MARKETS_PER_CROP = 5   # same floor Script 31 Part C.4 uses before attempting the arrivals design

EPISODES = {
    '2019': dict(ban_start=pd.Timestamp('2019-09-29'), label='Sep 2019 ban (DGFT 21/2015-2020)'),
    '2020': dict(ban_start=pd.Timestamp('2020-09-14'), label='Sep 2020 ban (DGFT 31/2015-2020)'),
    '2023': dict(ban_start=pd.Timestamp('2023-12-08'), label='Dec 2023 ban'),
}
CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}
EPISODE_COLORS = {'2019': '#4C86A8', '2020': '#B8860B', '2023': '#7B2C8E'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                      'axes.spines.top': False, 'axes.spines.right': False})

print('=' * 65)
print('SCRIPT 39b: STACKED MULTI-EPISODE SDID EVENT STUDY, ARRIVALS OUTCOME')
print('  Onion export bans: 2019, 2020, 2023-24')
print('=' * 65)

df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
_collision = df.groupby(['crop', 'market'])['market_id'].transform('nunique') > 1
df.loc[_collision, 'market'] = df.loc[_collision, 'market'] + ' (' + df.loc[_collision, 'state'] + ')'
print(f'  [collision fix] relabeled {int(_collision.sum())} rows across '
      f'{df.loc[_collision, ["crop", "market"]].drop_duplicates().shape[0]} (crop, market) pairs')

placebo_crops_for = {'onion': ['tomato', 'potato'], 'tomato': ['onion', 'potato'], 'potato': ['onion', 'tomato']}


# ─────────────────────────────────────────────────────────────────────────────
# SDID ESTIMATOR (identical to Script 31/38/38b/39)
# ─────────────────────────────────────────────────────────────────────────────
def solve_simplex_regression(X, y, ridge_penalty, x0=None, maxiter=500):
    n_obs, n_vars = X.shape

    def objective(params):
        w0, w = params[0], params[1:]
        resid = y - w0 - X @ w
        return float(np.sum(resid ** 2) + ridge_penalty * np.sum(w ** 2))

    if x0 is None:
        x0 = np.concatenate([[float(np.mean(y))], np.full(n_vars, 1.0 / n_vars)])
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[1:]) - 1.0}]
    bounds = [(None, None)] + [(0.0, 1.0)] * n_vars
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                    constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-10})
    return res.x[0], res.x[1:]


def compute_zeta(donors, pre_mask, post_mask):
    pre_diffs = donors.loc[pre_mask].diff().dropna().values
    sigma_hat = float(np.std(pre_diffs, ddof=1)) if pre_diffs.size else 0.0
    n_post = int(post_mask.sum())
    zeta = (n_post ** 0.25) * sigma_hat if sigma_hat > 0 else 1e-6
    return sigma_hat, zeta


def fit_time_weights(donors, pre_mask, post_mask, zeta):
    J = donors.shape[1]
    X_time = donors.loc[pre_mask].values.T
    y_time = donors.loc[post_mask].mean(axis=0).values
    w0_time, time_weights = solve_simplex_regression(X_time, y_time, ridge_penalty=J * zeta ** 2)
    return w0_time, time_weights


def fit_unit_weights(donors, treated, pre_mask, zeta, maxiter=500):
    X_unit = donors.loc[pre_mask].values
    y_unit = treated.loc[pre_mask].values
    n_pre = X_unit.shape[0]
    w0_unit, unit_weights = solve_simplex_regression(
        X_unit, y_unit, ridge_penalty=n_pre * zeta ** 2, maxiter=maxiter)
    return w0_unit, unit_weights


def event_study_series(treated, donors, pre_mask, post_mask, w0_unit, unit_weights, time_weights):
    synthetic = w0_unit + donors.values @ np.r_[unit_weights]
    synthetic = pd.Series(synthetic, index=treated.index)
    gap = treated - synthetic
    baseline = float(np.sum(time_weights * gap.loc[pre_mask].values))
    dyn_log = gap - baseline
    return dyn_log.apply(lambda v: float(np.expm1(v) * 100))


# ─────────────────────────────────────────────────────────────────────────────
# PER-EPISODE ARRIVALS PANEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_episode_arrivals_panel(win_start, win_end):
    w = df[(df['week_start'] >= win_start) & (df['week_start'] <= win_end)].copy()
    weeks = sorted(w['week_start'].unique())
    nw = len(weeks)
    weeks_s = pd.Series(weeks)

    # Price-qualifying set for this episode's window (Script 39's own logic)
    cov = w.groupby(['crop', 'market'])['imputed'].agg(['mean', 'count'])
    cov['real_cov'] = 1 - cov['mean']
    qualifying = cov[(cov['count'] == nw) & (cov['real_cov'] >= COVERAGE_THRESHOLD)].reset_index()
    price_pivot = w.pivot_table(index='week_start', columns=['crop', 'market'], values='modal_price_weighted').reindex(weeks)
    price_complete_cols = price_pivot.columns[price_pivot.notna().all(axis=0)]
    qualifying = qualifying[qualifying.apply(lambda r: (r['crop'], r['market']) in price_complete_cols, axis=1)]

    # Arrivals qualification among that set (Script 31 Part C.4 logic, re-evaluated per episode)
    arr_pivot = w.pivot_table(index='week_start', columns=['crop', 'market'], values='arrivals_tonnes_week').reindex(weeks)
    qualifying_cols = [(r['crop'], r['market']) for _, r in qualifying.iterrows()]
    arr_pivot_q = arr_pivot.reindex(columns=[c for c in qualifying_cols if c in arr_pivot.columns])
    arr_cov = arr_pivot_q.notna().mean(axis=0)
    arr_candidate_cols = arr_cov[arr_cov >= COVERAGE_THRESHOLD].index
    arr_interp = arr_pivot_q[arr_candidate_cols].interpolate(method='linear', limit=INTERP_LIMIT, limit_area='inside')
    arr_complete_cols = arr_interp.columns[arr_interp.notna().all(axis=0)]
    qualifying_arr = qualifying[qualifying.apply(lambda r: (r['crop'], r['market']) in arr_complete_cols, axis=1)].copy()
    log_arr_pivot = np.log1p(arr_interp[arr_complete_cols])

    return qualifying_arr, log_arr_pivot, weeks_s


def series_for(crop, markets_df, log_pivot):
    markets = markets_df[markets_df['crop'] == crop]['market'].tolist()
    return log_pivot[[(crop, m) for m in markets]].mean(axis=1)


def donors_for(crops, markets_df, log_pivot):
    rows = markets_df[markets_df['crop'].isin(crops)]
    cols = [(r['crop'], r['market']) for _, r in rows.iterrows()]
    d = log_pivot[cols]
    d.columns = [f'{c}__{m}' for c, m in cols]
    return d


# ─────────────────────────────────────────────────────────────────────────────
# RUN EACH EPISODE
# ─────────────────────────────────────────────────────────────────────────────
event_rows = []
skipped_episodes = []
for ep_name, ep in EPISODES.items():
    ban_start = ep['ban_start']
    win_end = ban_start + pd.Timedelta(weeks=EVENT_WINDOW[1] + 4)
    print(f'\n[{ep_name}] {ep["label"]}: window {PANEL_START.date()} to {win_end.date()} ...')

    qualifying_arr, log_arr_pivot, weeks_s = build_episode_arrivals_panel(PANEL_START, win_end)
    pre_mask = weeks_s.between(PANEL_START, ban_start, inclusive='left').values
    post_mask_fit = weeks_s.between(ban_start, ban_start + pd.Timedelta(weeks=POST_WEEKS_FOR_FIT),
                                     inclusive='both').values
    counts = qualifying_arr.groupby('crop').size().to_dict()
    n_onion = counts.get('onion', 0)
    n_tomato = counts.get('tomato', 0)
    n_potato = counts.get('potato', 0)
    print(f'  Arrivals-qualifying markets (>= {COVERAGE_THRESHOLD:.0%} coverage, gaps <={INTERP_LIMIT}wk interpolated): '
          f'onion={n_onion}, tomato={n_tomato}, potato={n_potato}  |  pre={pre_mask.sum()}wk  post(fit)={post_mask_fit.sum()}wk')

    if n_onion < MIN_MARKETS_PER_CROP or (n_tomato + n_potato) < MIN_MARKETS_PER_CROP:
        print(f'  SKIPPED -- too few arrivals-qualifying markets for a meaningful fit '
              f'(need >= {MIN_MARKETS_PER_CROP} onion and >= {MIN_MARKETS_PER_CROP} combined donors).')
        skipped_episodes.append(ep_name)
        continue

    for unit_crop in ['onion', 'tomato', 'potato']:
        if counts.get(unit_crop, 0) < MIN_MARKETS_PER_CROP:
            print(f'    [{unit_crop:6s}] SKIPPED -- only {counts.get(unit_crop, 0)} arrivals-qualifying markets')
            continue
        role = 'treated' if unit_crop == 'onion' else 'placebo'
        treated_u = series_for(unit_crop, qualifying_arr, log_arr_pivot)
        donors_u = donors_for(placebo_crops_for[unit_crop], qualifying_arr, log_arr_pivot)
        if donors_u.shape[1] < 2:
            print(f'    [{unit_crop:6s}] SKIPPED -- fewer than 2 donor markets available')
            continue

        sigma_hat, zeta = compute_zeta(donors_u, pre_mask, post_mask_fit)
        w0_time, time_weights = fit_time_weights(donors_u, pre_mask, post_mask_fit, zeta)
        w0_unit, unit_weights = fit_unit_weights(donors_u, treated_u, pre_mask, zeta)
        dyn_pct = event_study_series(treated_u, donors_u, pre_mask, post_mask_fit,
                                      w0_unit, unit_weights, time_weights)

        for w, v in dyn_pct.items():
            weeks_since = (w - ban_start).days / 7
            if EVENT_WINDOW[0] <= weeks_since <= EVENT_WINDOW[1]:
                event_rows.append({'episode': ep_name, 'unit': unit_crop, 'role': role,
                                    'week_start': str(w.date()), 'weeks_since_ban': round(weeks_since, 1),
                                    'dynamic_ATT_pct': round(float(v), 2)})
        print(f'    [{unit_crop:6s} ({role})] fit complete, sigma_hat={sigma_hat:.4f}, '
              f'n_donors={donors_u.shape[1]}')

if skipped_episodes:
    print(f'\n  NOTE: episode(s) {skipped_episodes} skipped entirely for lack of arrivals-qualifying markets.')

event_df = pd.DataFrame(event_rows)
if len(event_df) == 0:
    print('\nNo episodes produced usable arrivals fits -- exiting without writing output.')
    sys.exit(0)

event_path = os.path.join(OUT_DIR, 'table_sdid_arrivals_stacked_event_study.csv')
event_df.to_csv(event_path, index=False)
print(f'\n[Saved] {event_path}  ({len(event_df)} rows)')


# ─────────────────────────────────────────────────────────────────────────────
# STACK
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Stacking across episodes] ...')
event_df['week_bin'] = event_df['weeks_since_ban'].round().astype(int)

summary_rows = []
for role, grp in event_df.groupby('role'):
    for wk, g in grp.groupby('week_bin'):
        per_episode = g.groupby('episode')['dynamic_ATT_pct'].mean()
        summary_rows.append({
            'role': role, 'week_bin': wk, 'n_episodes': len(per_episode),
            'mean_pct': round(float(per_episode.mean()), 2),
            'std_pct': round(float(per_episode.std(ddof=1)), 2) if len(per_episode) > 1 else None,
            'min_pct': round(float(per_episode.min()), 2), 'max_pct': round(float(per_episode.max()), 2),
        })
summary_df = pd.DataFrame(summary_rows).sort_values(['role', 'week_bin'])
summary_path = os.path.join(OUT_DIR, 'table_sdid_arrivals_stacked_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f'[Saved] {summary_path}  ({len(summary_df)} rows)')

post_window = summary_df[(summary_df['week_bin'] >= 0) & (summary_df['week_bin'] <= 12)]
print('\n  Post-ban (weeks 0-12) stacked mean dynamic ATT, by role (ARRIVALS outcome):')
for role, g in post_window.groupby('role'):
    print(f'    {role:8s}: mean={g["mean_pct"].mean():+.1f}%  (n_episodes per week ~{g["n_episodes"].mean():.1f})')

# Explicit onion-vs-placebo-band separation check, mirroring how Script 39's
# price result is read: does onion's stacked mean sit outside the placebo band
# (min-max across tomato+potato) at each event-time bin in the post-ban window?
onion_post = summary_df[(summary_df['role'] == 'treated') & (summary_df['week_bin'] >= 0) & (summary_df['week_bin'] <= 12)]
placebo_post = summary_df[(summary_df['role'] == 'placebo') & (summary_df['week_bin'] >= 0) & (summary_df['week_bin'] <= 12)]
merged = onion_post.merge(placebo_post, on='week_bin', suffixes=('_onion', '_placebo'))
merged['onion_outside_band'] = (merged['mean_pct_onion'] < merged['min_pct_placebo']) | (merged['mean_pct_onion'] > merged['max_pct_placebo'])
n_outside = int(merged['onion_outside_band'].sum())
n_total = len(merged)
print(f'\n  Onion stacked mean OUTSIDE the tomato/potato placebo range in {n_outside}/{n_total} '
      f'post-ban week-bins (weeks 0-12).')
print(merged[['week_bin', 'mean_pct_onion', 'min_pct_placebo', 'max_pct_placebo', 'onion_outside_band']].to_string(index=False))

print('\n  NOTE: with only 3 episodes (fewer if any were skipped above for lack of arrivals')
print('  coverage), a formal cross-episode standard error is weak -- read this alongside the')
print('  figure and the outside-band count above, not as a p-value from n=3.')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Figure] ...')
fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

ax = axes[0]
episodes_present = event_df['episode'].unique().tolist()
for ep_name in episodes_present:
    g = event_df[(event_df['unit'] == 'onion') & (event_df['episode'] == ep_name)].sort_values('weeks_since_ban')
    ax.plot(g['weeks_since_ban'], g['dynamic_ATT_pct'], color=EPISODE_COLORS.get(ep_name, '#333333'),
            linewidth=1.6, alpha=0.85, label=f'{ep_name} ({EPISODES[ep_name]["label"]})')
onion_summary = summary_df[summary_df['role'] == 'treated']
ax.plot(onion_summary['week_bin'], onion_summary['mean_pct'], color='black', linewidth=2.6,
        label=f'Stacked mean (all {len(episodes_present)} episodes)')
ax.axhline(0, color='#888888', linewidth=0.8)
ax.axvline(0, color='#333333', linewidth=1.1, linestyle='--')
ax.set_title('Onion ARRIVALS (treated): dynamic ATT vs. synthetic control, aligned by weeks since ban',
              fontsize=10.5, fontweight='bold')
ax.set_ylabel('Dynamic ATT on arrivals (%)')
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.grid(axis='y', alpha=0.25)

ax = axes[1]
placebo_summary = summary_df[summary_df['role'] == 'placebo']
ax.fill_between(placebo_summary['week_bin'], placebo_summary['min_pct'], placebo_summary['max_pct'],
                 color='#888888', alpha=0.25, label='Placebo range (tomato+potato arrivals, all episodes)')
ax.plot(placebo_summary['week_bin'], placebo_summary['mean_pct'], color='#666666', linewidth=1.8,
        linestyle='--', label='Placebo mean')
ax.plot(onion_summary['week_bin'], onion_summary['mean_pct'], color=CROP_COLORS['onion'], linewidth=2.6,
        label='Onion (treated) stacked mean')
ax.axhline(0, color='#888888', linewidth=0.8)
ax.axvline(0, color='#333333', linewidth=1.1, linestyle='--')
ax.set_title('Stacked mean: onion (treated) vs. tomato+potato (placebo) range, ARRIVALS outcome',
              fontsize=10.5, fontweight='bold')
ax.set_xlabel('Weeks since ban')
ax.set_ylabel('Dynamic ATT on arrivals (%)')
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_sdid_arrivals_stacked_event_study.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[Saved] {fig_path}')

print('\n' + '=' * 65)
print('Script 39b complete.')
