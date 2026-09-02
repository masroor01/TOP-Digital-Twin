# -*- coding: utf-8 -*-
"""
Script 38b — SDID Arrivals (Quantity) In-Space Placebo Permutation Test (postban)
=============================================================================
Script 31 Part C.4 fits the cross-crop SDID design to onion's ARRIVALS
(quantity) rather than price, testing whether the 2023-24 export ban's stated
mechanism -- retain more supply domestically, raising arrivals -- actually
shows up in the data. It reports a real, directionally consistent ATT (onion
arrivals DOWN across all three sub-windows, tomato/potato placebo arrivals UP),
but unlike the price-based postban ATT (Script 38), this arrivals ATT has never
been tested against a proper null distribution -- only eyeballed against 2
placebo point estimates.

This script runs the identical in-space placebo permutation test Script 38
already ran for price (Abadie, Diamond & Hainmueller 2010 design: treat every
individual tomato/potato donor market as if it were "treated", one at a time,
refit against the remaining markets, build a null distribution, rank the real
onion ATT against it), but on log1p(arrivals) instead of log(price), using the
EXACT same arrivals-qualification logic as Script 31 Part C.4 (95% coverage of
the price-qualifying market set, linear interpolation of gaps <=4 weeks, then
requiring a fully rectangular matrix).

Market-name-vs-market_id collision fix (2026-08-14 pipeline-wide bug) is
applied here from the start, consistent with Script 31 and Script 38.

Inputs:
  data/agmarknet_weekly/top_weekly_panel.csv

Outputs (Model_Output/):
  table_sdid_arrivals_inspace_placebo.csv   per-market placebo ATT (postban window, arrivals outcome)
  fig_sdid_arrivals_inspace_placebo.png     histogram of placebo ATTs vs. real onion arrivals ATT

Run: python scripts/38b_SDID_Arrivals_InSpace_Placebo_Test.py
Estimated runtime: ~2-5 minutes (fewer qualifying donor markets than the
price-based test, since arrivals coverage is patchier -- see Script 31 Part C.4).
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
INTERP_LIMIT = 4
CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                      'axes.spines.top': False, 'axes.spines.right': False})

print('=' * 65)
print('SCRIPT 38b: SDID ARRIVALS IN-SPACE PLACEBO PERMUTATION TEST (postban)')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 1. REBUILD THE SAME PANEL AS SCRIPT 31 PART A/C.4
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Rebuilding the market-level panel (identical to Script 31 Part A / C.4) ...')

df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
_collision = df.groupby(['crop', 'market'])['market_id'].transform('nunique') > 1
df.loc[_collision, 'market'] = df.loc[_collision, 'market'] + ' (' + df.loc[_collision, 'state'] + ')'
print(f'  [collision fix] relabeled {int(_collision.sum())} rows across '
      f'{df.loc[_collision, ["crop", "market"]].drop_duplicates().shape[0]} (crop, market) pairs')

window = df[(df['week_start'] >= WINDOW_START) & (df['week_start'] <= WINDOW_END)].copy()
all_weeks = sorted(window['week_start'].unique())
n_weeks = len(all_weeks)
weeks_s = pd.Series(all_weeks)

# Price-qualifying set (Script 31 Part A): 100%-complete price columns, >=95% real coverage
cov = window.groupby(['crop', 'market'])['imputed'].agg(['mean', 'count'])
cov['real_cov'] = 1 - cov['mean']
qualifying = cov[(cov['count'] == n_weeks) & (cov['real_cov'] >= COVERAGE_THRESHOLD)].reset_index()
price_pivot = window.pivot_table(index='week_start', columns=['crop', 'market'], values='modal_price_weighted')
price_pivot = price_pivot.reindex(all_weeks)
price_complete_cols = price_pivot.columns[price_pivot.notna().all(axis=0)]
qualifying = qualifying[qualifying.apply(lambda r: (r['crop'], r['market']) in price_complete_cols, axis=1)]
print(f'  Price-qualifying markets: {qualifying.groupby("crop").size().to_dict()}')

# Arrivals qualification (Script 31 Part C.4, exact logic): among the price-qualifying
# set, keep markets with >=95% arrivals coverage, linearly interpolate internal gaps
# up to 4 weeks, then require a fully rectangular (complete) matrix.
arr_pivot = window.pivot_table(index='week_start', columns=['crop', 'market'], values='arrivals_tonnes_week')
arr_pivot = arr_pivot.reindex(all_weeks)
qualifying_cols = [(r['crop'], r['market']) for _, r in qualifying.iterrows()]
arr_pivot_q = arr_pivot.reindex(columns=[c for c in qualifying_cols if c in arr_pivot.columns])
arr_cov = arr_pivot_q.notna().mean(axis=0)
arr_candidate_cols = arr_cov[arr_cov >= COVERAGE_THRESHOLD].index
arr_interp = arr_pivot_q[arr_candidate_cols].interpolate(method='linear', limit=INTERP_LIMIT, limit_area='inside')
arr_complete_cols = arr_interp.columns[arr_interp.notna().all(axis=0)]
qualifying_arr = qualifying[qualifying.apply(lambda r: (r['crop'], r['market']) in arr_complete_cols, axis=1)].copy()
print(f'  Arrivals-qualifying markets (>= {COVERAGE_THRESHOLD:.0%} coverage, gaps <={INTERP_LIMIT}wk interpolated): '
      f'{qualifying_arr.groupby("crop").size().to_dict()}')

log_arr_pivot = np.log1p(arr_interp[arr_complete_cols])

treated_markets = qualifying_arr[qualifying_arr['crop'] == 'onion']['market'].tolist()
donor_markets = qualifying_arr[qualifying_arr['crop'].isin(['tomato', 'potato'])][['crop', 'market']]
print(f'  Treated (onion) markets: {len(treated_markets)}')
print(f'  Donor (tomato+potato) markets: {len(donor_markets)}')

treated_series = log_arr_pivot[[('onion', m) for m in treated_markets]].mean(axis=1)
donor_cols = [(r['crop'], r['market']) for _, r in donor_markets.iterrows()]
donor_matrix_full = log_arr_pivot[donor_cols]
donor_matrix_full.columns = [f'{c}__{m}' for c, m in donor_cols]

pre_mask = weeks_s.between(WINDOW_START, TREAT_START, inclusive='left').values
post_mask_postban = weeks_s.between(BAN_START, WINDOW_END, inclusive='both').values
print(f'  Pre-period: {pre_mask.sum()} weeks  |  Postban window: {post_mask_postban.sum()} weeks')


# ─────────────────────────────────────────────────────────────────────────────
# 2. SDID ESTIMATOR (identical to Script 31/38)
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
    # NOTE: outcome is log1p(arrivals), not log(price) -- use expm1 consistently,
    # same as the price script, since log1p/expm1 are the correct inverse pair and
    # the ATT is a difference of already-logged quantities either way.
    att_pct = float(np.expm1(att) * 100)
    return att, att_pct


# ─────────────────────────────────────────────────────────────────────────────
# 3. REAL ONION ARRIVALS ATT (postban) -- should match Script 31 Part C.4's
# postban row for onion (ATT_log_points=-0.1548, ATT_pct=-14.3) as a sanity check.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Fitting the real onion arrivals ATT (postban) ...')
sigma_hat, zeta = compute_zeta(donor_matrix_full, pre_mask, post_mask_postban)
w0_time, time_weights = fit_time_weights(donor_matrix_full, pre_mask, post_mask_postban, zeta)
w0_unit_real, unit_weights_real = fit_unit_weights(donor_matrix_full, treated_series, pre_mask, zeta)
att_real, att_real_pct = sdid_att(treated_series, donor_matrix_full, pre_mask, post_mask_postban,
                                   w0_unit_real, unit_weights_real, time_weights)
print(f'  Real onion arrivals ATT (postban): {att_real:+.4f} log-pts ({att_real_pct:+.1f}%)')
print(f'  [Sanity check vs table_sdid_arrivals_effect.csv postban row: ATT_log_points=-0.1548, ATT_pct=-14.3]')
x0_warm_real = np.concatenate([[w0_unit_real], unit_weights_real])


# ─────────────────────────────────────────────────────────────────────────────
# 4. IN-SPACE PLACEBO LOOP
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
    if (i + 1) % 10 == 0:
        print(f'    ... {i + 1}/{len(donor_cols)} placebo markets done')

placebo_df = pd.DataFrame(placebo_rows)
placebo_path = os.path.join(OUT_DIR, 'table_sdid_arrivals_inspace_placebo.csv')
placebo_df.to_csv(placebo_path, index=False)
print(f'\n  Saved: {placebo_path}  ({len(placebo_df)} placebo markets)')


# ─────────────────────────────────────────────────────────────────────────────
# 5. RANK-BASED P-VALUE + SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
placebo_atts_log = placebo_df['ATT_log_points'].values
n_placebo = len(placebo_atts_log)
n_as_extreme = int(np.sum(np.abs(placebo_atts_log) >= np.abs(att_real)))
p_value = (n_as_extreme + 1) / (n_placebo + 1) if n_placebo > 0 else float('nan')

print('\n[4] In-space placebo distribution (postban window, ARRIVALS outcome):')
print(f'  n placebo markets: {n_placebo}')
if n_placebo > 0:
    print(f'  Placebo ATT (log-pts): mean={placebo_atts_log.mean():+.4f}  std={placebo_atts_log.std(ddof=1):.4f}  '
          f'min={placebo_atts_log.min():+.4f}  max={placebo_atts_log.max():+.4f}')
print(f'  Real onion arrivals ATT: {att_real:+.4f} log-pts ({att_real_pct:+.1f}%)')
print(f'  Placebo markets with |ATT| >= |real onion ATT|: {n_as_extreme}/{n_placebo}')
print(f'  Rank-based p-value (two-sided, Abadie-style): {p_value:.3f}')
verdict = 'FAILS' if p_value > 0.10 else 'PASSES'
print(f'  Verdict at 10% level: {verdict} the placebo falsification test')

if n_placebo > 0:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(placebo_df['ATT_pct'], bins=min(30, max(5, n_placebo // 2)), color='#888888', alpha=0.7,
            edgecolor='white', label=f'Placebo ATTs (n={n_placebo} donor markets, arrivals outcome)')
    ax.axvline(att_real_pct, color=CROP_COLORS['onion'], linewidth=2.2,
               label=f'Real onion arrivals ATT ({att_real_pct:+.1f}%)')
    ax.axvline(-att_real_pct if att_real_pct != 0 else 0, color=CROP_COLORS['onion'],
               linewidth=1.2, linestyle=':', alpha=0.6, label='Mirror (for two-sided reference)')
    ax.set_title(f'In-space placebo test, postban window, ARRIVALS outcome -- rank-based p={p_value:.3f} '
                 f'({n_as_extreme}/{n_placebo} placebos at least as extreme)',
                 fontsize=10.5, fontweight='bold')
    ax.set_xlabel('SDID ATT on log1p(arrivals) (%)')
    ax.set_ylabel('Count of placebo markets')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_sdid_arrivals_inspace_placebo.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {fig_path}')
else:
    print('\n  SKIPPED figure -- zero placebo markets available.')

print('\n' + '=' * 65)
print('Script 38b complete.')
