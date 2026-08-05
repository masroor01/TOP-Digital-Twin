# -*- coding: utf-8 -*-
"""
Script 39 — Stacked Multi-Episode SDID Event Study: Onion Export Bans
(2019, 2020, 2023-24)
=============================================================================
Script 31's SDID design has a single treated episode (the 2023-24 ban),
and both its cross-crop and within-onion placebo checks fail for the
postban window -- the estimated effect is statistically indistinguishable
from noise (Script 38's in-space placebo test: p=0.131). The root problem
is n=1: a single confounded natural experiment can't separate the ban's
effect from the coincident weather shock that prompted it.

Verification (session discussion, prior turn) found two MORE primary-
source-documented onion export bans inside the existing panel window:
  - 29 Sep 2019 (DGFT Notification 21/2015-2020), lifted 15 Mar 2020
  - 14 Sep 2020 (DGFT Notification 31/2015-2020), lifted 28 Dec 2020
Checking CHIRPS rainfall anomalies confirmed BOTH also coincide with real
weather shocks -- but shocks of the OPPOSITE character to 2023's (2019/
2020: wetter than normal by +46/+55 mm/week on average; 2023: drier than
normal by -24 mm/week). That's the useful part: three episodes with
DIFFERENT confound signs give a real test that one episode cannot. A price
pattern that consistently kinks at the same event-time offset across all
three, despite their weather anomalies pulling in different directions,
cannot be explained by weather alone -- that consistency is the
identification strategy here, not a bigger single-episode estimate.

Design: for each of the 3 episodes, refit the IDENTICAL SDID procedure as
Script 31 Part A (onion = treated crop average, tomato+potato = donor
pool), each on its own qualifying-market set (coverage requirements are
evaluated over that episode's own window, since which markets have
>=95% real coverage shifts over time as Agmarknet reporting grows).
Extract the dynamic (event-study) ATT trajectory per episode via the same
algebraic decomposition as Script 31 Part C.1, but align by WEEKS SINCE
BAN rather than calendar date, then stack across episodes. Tomato and
potato are run as placebo-treated units at the same 3 real dates, exactly
as Script 31 Part A does for the single 2023-24 episode, giving a stacked
null-reference band.

With only 3 episodes, a formal cross-episode standard error is weak (this
is stated explicitly in the output, not glossed over) -- the visual
alignment (or lack of it) across differently-shocked episodes is the
primary evidence here, not a p-value from n=3.

Inputs:
  data/agmarknet_weekly/top_weekly_panel.csv
  data/satellite_climate/crop_weekly_features.csv   (rainfall anomaly recap only)

Outputs (Model_Output/):
  table_sdid_stacked_event_study.csv   per-episode x role x weeks-since-ban dynamic ATT
  table_sdid_stacked_summary.csv       stacked mean/spread per role x event-time bin
  fig_sdid_stacked_event_study.png     all 3 episodes aligned, real vs placebo

Run: python scripts/39_SDID_Stacked_MultiEpisode_EventStudy.py
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
POST_WEEKS_FOR_FIT = 26     # post-ban weeks used to fit time weights (matches Script 31 scale)
EVENT_WINDOW = (-20, 30)    # weeks-since-ban range shown in the stacked event study

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
print('SCRIPT 39: STACKED MULTI-EPISODE SDID EVENT STUDY')
print('  Onion export bans: 2019, 2020, 2023-24')
print('=' * 65)

df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
placebo_crops_for = {'onion': ['tomato', 'potato'], 'tomato': ['onion', 'potato'], 'potato': ['onion', 'tomato']}


# ─────────────────────────────────────────────────────────────────────────────
# SDID ESTIMATOR (duplicated from Script 31/38 -- identical implementation)
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
# PER-EPISODE PANEL BUILDER -- coverage re-evaluated on that episode's own window
# ─────────────────────────────────────────────────────────────────────────────
def build_episode_panel(win_start, win_end):
    w = df[(df['week_start'] >= win_start) & (df['week_start'] <= win_end)].copy()
    weeks = sorted(w['week_start'].unique())
    nw = len(weeks)
    cov = w.groupby(['crop', 'market'])['imputed'].agg(['mean', 'count'])
    cov['real_cov'] = 1 - cov['mean']
    qualifying = cov[(cov['count'] == nw) & (cov['real_cov'] >= COVERAGE_THRESHOLD)].reset_index()
    pivot = w.pivot_table(index='week_start', columns=['crop', 'market'], values='modal_price_weighted').reindex(weeks)
    complete_cols = pivot.columns[pivot.notna().all(axis=0)]
    qualifying = qualifying[qualifying.apply(lambda r: (r['crop'], r['market']) in complete_cols, axis=1)]
    log_pivot = np.log(pivot)
    weeks_s = pd.Series(weeks)
    return qualifying, log_pivot, weeks_s


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
for ep_name, ep in EPISODES.items():
    ban_start = ep['ban_start']
    win_end = ban_start + pd.Timedelta(weeks=EVENT_WINDOW[1] + 4)   # a little buffer past the display window
    print(f'\n[{ep_name}] {ep["label"]}: window {PANEL_START.date()} to {win_end.date()} ...')

    qualifying, log_pivot, weeks_s = build_episode_panel(PANEL_START, win_end)
    pre_mask = weeks_s.between(PANEL_START, ban_start, inclusive='left').values
    post_mask_fit = weeks_s.between(ban_start, ban_start + pd.Timedelta(weeks=POST_WEEKS_FOR_FIT),
                                     inclusive='both').values
    n_onion = len(qualifying[qualifying['crop'] == 'onion'])
    n_donor = len(qualifying[qualifying['crop'].isin(['tomato', 'potato'])])
    print(f'  Qualifying markets (>= {COVERAGE_THRESHOLD:.0%} coverage, no gaps): '
          f'{n_onion} onion, {n_donor} tomato+potato  |  pre={pre_mask.sum()}wk  post(fit)={post_mask_fit.sum()}wk')

    for unit_crop in ['onion', 'tomato', 'potato']:
        role = 'treated' if unit_crop == 'onion' else 'placebo'
        treated_u = series_for(unit_crop, qualifying, log_pivot)
        donors_u = donors_for(placebo_crops_for[unit_crop], qualifying, log_pivot)

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
        print(f'    [{unit_crop:6s} ({role})] fit complete, sigma_hat={sigma_hat:.4f}')

event_df = pd.DataFrame(event_rows)
event_path = os.path.join(OUT_DIR, 'table_sdid_stacked_event_study.csv')
event_df.to_csv(event_path, index=False)
print(f'\n[Saved] {event_path}  ({len(event_df)} rows)')


# ─────────────────────────────────────────────────────────────────────────────
# STACK: round weeks-since-ban to the nearest integer week for cross-episode
# alignment, then take mean/std across episodes at each integer offset.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Stacking across episodes] ...')
event_df['week_bin'] = event_df['weeks_since_ban'].round().astype(int)

summary_rows = []
for role, grp in event_df.groupby('role'):
    for wk, g in grp.groupby('week_bin'):
        per_episode = g.groupby('episode')['dynamic_ATT_pct'].mean()  # collapse any same-week dupes within an episode
        summary_rows.append({
            'role': role, 'week_bin': wk, 'n_episodes': len(per_episode),
            'mean_pct': round(float(per_episode.mean()), 2),
            'std_pct': round(float(per_episode.std(ddof=1)), 2) if len(per_episode) > 1 else None,
            'min_pct': round(float(per_episode.min()), 2), 'max_pct': round(float(per_episode.max()), 2),
        })
summary_df = pd.DataFrame(summary_rows).sort_values(['role', 'week_bin'])
summary_path = os.path.join(OUT_DIR, 'table_sdid_stacked_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f'[Saved] {summary_path}  ({len(summary_df)} rows)')

# Post-ban headline: average dynamic ATT over weeks 0-12 post-ban, per role
post_window = summary_df[(summary_df['week_bin'] >= 0) & (summary_df['week_bin'] <= 12)]
print('\n  Post-ban (weeks 0-12) stacked mean dynamic ATT, by role:')
for role, g in post_window.groupby('role'):
    print(f'    {role:8s}: mean={g["mean_pct"].mean():+.1f}%  (n_episodes per week ~{g["n_episodes"].mean():.1f})')
print('\n  NOTE: with only 3 episodes, the cross-episode std above is a weak signal on its own --')
print('  read this alongside the figure for whether onion consistently separates from the tomato/potato')
print('  placebo band across all three episodes, not as a formal significance test.')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
print('\n[Figure] ...')
fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

ax = axes[0]
for ep_name in EPISODES:
    g = event_df[(event_df['unit'] == 'onion') & (event_df['episode'] == ep_name)].sort_values('weeks_since_ban')
    ax.plot(g['weeks_since_ban'], g['dynamic_ATT_pct'], color=EPISODE_COLORS[ep_name],
            linewidth=1.6, alpha=0.85, label=f'{ep_name} ({EPISODES[ep_name]["label"]})')
onion_summary = summary_df[summary_df['role'] == 'treated']
ax.plot(onion_summary['week_bin'], onion_summary['mean_pct'], color='black', linewidth=2.6,
        label='Stacked mean (all 3 episodes)')
ax.axhline(0, color='#888888', linewidth=0.8)
ax.axvline(0, color='#333333', linewidth=1.1, linestyle='--')
ax.set_title('Onion (treated): dynamic ATT vs. synthetic control, aligned by weeks since ban',
              fontsize=10.5, fontweight='bold')
ax.set_ylabel('Dynamic ATT (%)')
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.grid(axis='y', alpha=0.25)

ax = axes[1]
placebo_summary = summary_df[summary_df['role'] == 'placebo']
ax.fill_between(placebo_summary['week_bin'], placebo_summary['min_pct'], placebo_summary['max_pct'],
                 color='#888888', alpha=0.25, label='Placebo range (tomato+potato, all 3 episodes)')
ax.plot(placebo_summary['week_bin'], placebo_summary['mean_pct'], color='#666666', linewidth=1.8,
        linestyle='--', label='Placebo mean')
ax.plot(onion_summary['week_bin'], onion_summary['mean_pct'], color=CROP_COLORS['onion'], linewidth=2.6,
        label='Onion (treated) stacked mean')
ax.axhline(0, color='#888888', linewidth=0.8)
ax.axvline(0, color='#333333', linewidth=1.1, linestyle='--')
ax.set_title('Stacked mean: onion (treated) vs. tomato+potato (placebo) range, all 3 episodes',
              fontsize=10.5, fontweight='bold')
ax.set_xlabel('Weeks since ban')
ax.set_ylabel('Dynamic ATT (%)')
ax.legend(frameon=False, fontsize=8.5, loc='upper left')
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_sdid_stacked_event_study.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'[Saved] {fig_path}')

print('\n' + '=' * 65)
print('Script 39 complete.')
