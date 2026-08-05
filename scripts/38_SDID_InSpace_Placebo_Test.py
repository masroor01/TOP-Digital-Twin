# -*- coding: utf-8 -*-
"""
Script 38 — SDID In-Space Placebo Permutation Test (postban window)
=============================================================================
Follow-up to Script 31's cross-crop placebo check, which found the postban
onion ATT (+14.5%) statistically indistinguishable from CROP-AVERAGED
tomato/potato placebo ATTs (+14.5%, +15.2%) via a jackknife 90% CI overlap
test. That comparison has only 2 placebo units. This script runs the
standard synthetic-control literature falsification test instead (Abadie,
Diamond & Hainmueller 2010, "Synthetic Control Methods for Comparative
Case Studies"): treat EVERY individual tomato/potato market as if it were
the "treated" unit, one at a time, re-fit the identical SDID procedure
against the remaining markets as its donor pool, and build a null
distribution of placebo ATTs. The real onion ATT is judged against that
distribution's own spread (rank-based p-value), not against 2 aggregated
placebo point estimates.

This reuses the same panel construction, coverage filter, pre-period, and
postban window as Script 31 Part A (self-contained duplicate of the small
set of estimator functions -- Script 31 executes top-level on import, so
it can't be imported directly without re-running its full 20-30 minute
Part A-C pipeline).

Inputs:
  data/agmarknet_weekly/top_weekly_panel.csv

Outputs (Model_Output/):
  table_sdid_inspace_placebo.csv   per-market placebo ATT (postban window)
  fig_sdid_inspace_placebo.png     histogram of placebo ATTs vs. real onion ATT

Run: python scripts/38_SDID_InSpace_Placebo_Test.py
Estimated runtime: ~5-10 minutes (one SLSQP refit per donor market, warm-
started from the full-sample solution).
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

WINDOW_START = pd.Timestamp('2017-01-02')
WINDOW_END   = pd.Timestamp('2024-05-04')
TREAT_START  = pd.Timestamp('2023-08-19')
BAN_START    = pd.Timestamp('2023-12-08')
COVERAGE_THRESHOLD = 0.95
CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                      'axes.spines.top': False, 'axes.spines.right': False})

print('=' * 65)
print('SCRIPT 38: SDID IN-SPACE PLACEBO PERMUTATION TEST (postban)')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1. REBUILD THE SAME PANEL AS SCRIPT 31 PART A
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Rebuilding the market-level panel (identical to Script 31 Part A) ...')

df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
window = df[(df['week_start'] >= WINDOW_START) & (df['week_start'] <= WINDOW_END)].copy()
all_weeks = sorted(window['week_start'].unique())
n_weeks = len(all_weeks)

cov = window.groupby(['crop', 'market'])['imputed'].agg(['mean', 'count'])
cov['real_cov'] = 1 - cov['mean']
qualifying = cov[(cov['count'] == n_weeks) & (cov['real_cov'] >= COVERAGE_THRESHOLD)].reset_index()

pivot = window.pivot_table(index='week_start', columns=['crop', 'market'], values='modal_price_weighted')
pivot = pivot.reindex(all_weeks)
complete_cols = pivot.columns[pivot.notna().all(axis=0)]
qualifying = qualifying[qualifying.apply(lambda r: (r['crop'], r['market']) in complete_cols, axis=1)]

treated_markets = qualifying[qualifying['crop'] == 'onion']['market'].tolist()
donor_markets = qualifying[qualifying['crop'].isin(['tomato', 'potato'])][['crop', 'market']]
print(f'  Treated (onion) markets: {len(treated_markets)}')
print(f'  Donor (tomato+potato) markets: {len(donor_markets)}')

log_pivot = np.log(pivot)
treated_series = log_pivot[[('onion', m) for m in treated_markets]].mean(axis=1)
donor_cols = [(r['crop'], r['market']) for _, r in donor_markets.iterrows()]
donor_matrix_full = log_pivot[donor_cols]
donor_matrix_full.columns = [f'{c}__{m}' for c, m in donor_cols]

weeks_s = pd.Series(all_weeks)
pre_mask = weeks_s.between(WINDOW_START, TREAT_START, inclusive='left').values
post_mask_postban = weeks_s.between(BAN_START, WINDOW_END, inclusive='both').values
print(f'  Pre-period: {pre_mask.sum()} weeks  |  Postban window: {post_mask_postban.sum()} weeks')


# ─────────────────────────────────────────────────────────────────────────────
# 2. SDID ESTIMATOR (duplicated from Script 31 -- identical implementation)
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


def fit_time_weights(donors, pre_mask, post_mask, zeta, x0=None):
    J = donors.shape[1]
    X_time = donors.loc[pre_mask].values.T
    y_time = donors.loc[post_mask].mean(axis=0).values
    w0_time, time_weights = solve_simplex_regression(X_time, y_time, ridge_penalty=J * zeta ** 2, x0=x0)
    return w0_time, time_weights


def fit_unit_weights(donors, treated, pre_mask, zeta, x0=None, maxiter=500):
    X_unit = donors.loc[pre_mask].values
    y_unit = treated.loc[pre_mask].values
    n_pre = X_unit.shape[0]
    w0_unit, unit_weights = solve_simplex_regression(
        X_unit, y_unit, ridge_penalty=n_pre * zeta ** 2, x0=x0, maxiter=maxiter)
    return w0_unit, unit_weights


def sdid_att(treated, donors, pre_mask, post_mask, w0_unit, unit_weights, time_weights):
    treated_post_mean = treated.loc[post_mask].mean()
    treated_pre_weighted = np.sum(time_weights * treated.loc[pre_mask].values)
    donor_post_weighted = np.sum(unit_weights * donors.loc[post_mask].mean(axis=0).values)
    donor_pre_weighted = np.sum(unit_weights * np.sum(
        time_weights[:, None] * donors.loc[pre_mask].values, axis=0))
    att = (treated_post_mean - treated_pre_weighted) - (donor_post_weighted - donor_pre_weighted)
    att_pct = float(np.expm1(att) * 100)
    return att, att_pct


# ─────────────────────────────────────────────────────────────────────────────
# 3. REAL ONION ATT (postban) -- same fit as Script 31
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Fitting the real onion ATT (postban) ...')
sigma_hat, zeta = compute_zeta(donor_matrix_full, pre_mask, post_mask_postban)
w0_time, time_weights = fit_time_weights(donor_matrix_full, pre_mask, post_mask_postban, zeta)
w0_unit_real, unit_weights_real = fit_unit_weights(donor_matrix_full, treated_series, pre_mask, zeta)
att_real, att_real_pct = sdid_att(treated_series, donor_matrix_full, pre_mask, post_mask_postban,
                                   w0_unit_real, unit_weights_real, time_weights)
print(f'  Real onion ATT (postban): {att_real:+.4f} log-pts ({att_real_pct:+.1f}%)')
x0_warm_real = np.concatenate([[w0_unit_real], unit_weights_real])


# ─────────────────────────────────────────────────────────────────────────────
# 4. IN-SPACE PLACEBO LOOP -- each donor market as a placebo-treated unit,
# against the REMAINING donor markets as its own donor pool. Time weights are
# refit too (donor pool changed by one column), unit weights warm-started
# from the real fit for speed.
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[3] In-space placebo loop: {len(donor_cols)} donor markets, each as a placebo-treated unit ...')

placebo_rows = []
donor_col_names = donor_matrix_full.columns.tolist()
for i, (crop_p, mkt_p) in enumerate(donor_cols):
    col_p = f'{crop_p}__{mkt_p}'
    placebo_treated = donor_matrix_full[col_p]
    placebo_donors = donor_matrix_full.drop(columns=[col_p])

    sigma_p, zeta_p = compute_zeta(placebo_donors, pre_mask, post_mask_postban)
    w0_time_p, time_weights_p = fit_time_weights(placebo_donors, pre_mask, post_mask_postban, zeta_p)
    # Warm-start from a uniform-ish guess over the reduced donor set (dropping
    # one column from the real warm-start vector and renormalizing) -- cheap
    # and close enough to speed up SLSQP without needing a fresh cold solve.
    keep_idx = [j for j, c in enumerate(donor_col_names) if c != col_p]
    w_warm = x0_warm_real[1:][keep_idx]
    w_warm = w_warm / w_warm.sum() if w_warm.sum() > 0 else np.full(len(keep_idx), 1.0 / len(keep_idx))
    x0_p = np.concatenate([[x0_warm_real[0]], w_warm])

    w0_unit_p, unit_weights_p = fit_unit_weights(placebo_donors, placebo_treated, pre_mask, zeta_p,
                                                  x0=x0_p, maxiter=200)
    att_p, att_p_pct = sdid_att(placebo_treated, placebo_donors, pre_mask, post_mask_postban,
                                 w0_unit_p, unit_weights_p, time_weights_p)
    placebo_rows.append({'crop': crop_p, 'market': mkt_p, 'ATT_log_points': round(att_p, 4),
                          'ATT_pct': round(att_p_pct, 1)})
    if (i + 1) % 40 == 0:
        print(f'    ... {i + 1}/{len(donor_cols)} placebo markets done')

placebo_df = pd.DataFrame(placebo_rows)
placebo_path = os.path.join(OUT_DIR, 'table_sdid_inspace_placebo.csv')
placebo_df.to_csv(placebo_path, index=False)
print(f'\n  Saved: {placebo_path}  ({len(placebo_df)} placebo markets)')


# ─────────────────────────────────────────────────────────────────────────────
# 5. RANK-BASED P-VALUE + SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
placebo_atts_log = placebo_df['ATT_log_points'].values
n_placebo = len(placebo_atts_log)
n_as_extreme = int(np.sum(np.abs(placebo_atts_log) >= np.abs(att_real)))
p_value = n_as_extreme / n_placebo

print('\n[4] In-space placebo distribution (postban window):')
print(f'  n placebo markets: {n_placebo}')
print(f'  Placebo ATT (log-pts): mean={placebo_atts_log.mean():+.4f}  std={placebo_atts_log.std(ddof=1):.4f}  '
      f'min={placebo_atts_log.min():+.4f}  max={placebo_atts_log.max():+.4f}')
print(f'  Real onion ATT: {att_real:+.4f} log-pts ({att_real_pct:+.1f}%)')
print(f'  Placebo markets with |ATT| >= |real onion ATT|: {n_as_extreme}/{n_placebo}')
print(f'  Rank-based p-value (two-sided, Abadie-style): {p_value:.3f}')
verdict = 'FAILS' if p_value > 0.10 else 'PASSES'
print(f'  Verdict at 10% level: {verdict} the placebo falsification test '
      f'({"cannot rule out the real ATT is within normal placebo noise" if verdict == "FAILS" else "real ATT is unusually extreme relative to the null distribution"})')

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(placebo_df['ATT_pct'], bins=30, color='#888888', alpha=0.7, edgecolor='white',
        label=f'Placebo ATTs (n={n_placebo} donor markets)')
ax.axvline(att_real_pct, color=CROP_COLORS['onion'], linewidth=2.2,
           label=f'Real onion ATT ({att_real_pct:+.1f}%)')
ax.axvline(-att_real_pct if att_real_pct != 0 else 0, color=CROP_COLORS['onion'],
           linewidth=1.2, linestyle=':', alpha=0.6, label='Mirror (for two-sided reference)')
ax.set_title(f'In-space placebo test, postban window -- rank-based p={p_value:.3f} '
             f'({n_as_extreme}/{n_placebo} placebos at least as extreme)',
             fontsize=10.5, fontweight='bold')
ax.set_xlabel('SDID ATT (%)')
ax.set_ylabel('Count of placebo markets')
ax.legend(frameon=False, fontsize=9)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_sdid_inspace_placebo.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'\n  Saved: {fig_path}')

print('\n' + '=' * 65)
print('Script 38 complete.')
