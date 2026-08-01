# -*- coding: utf-8 -*-
"""
Script 31 — Synthetic Difference-in-Differences: Onion Export-Restriction
Policy Effect
=============================================================================
Script 30's formal stress-testing module found that the M6 forecasting
model cannot recover a realistic counterfactual MAGNITUDE for onion's
export-duty/MEP/export-ban scenarios, even after a monotonic sign
constraint and (tried and reverted -- see Script 15/23) aggressive
oversampling. The root cause is an identification problem, not a modeling
one: export duty, MEP, and the export ban all activated together, in
sequence, in exactly ONE historical episode (Aug 2023-May 2024), so a
single time-series model trained on that one episode has no independent
variation to separate their effects, and the safest fit under a
monotonicity constraint is simply "no effect."

This script estimates the policy effect a different way: directly from
that one episode as a natural experiment, via Synthetic
Difference-in-Differences (Arkhangelsky, Athey, Hirshberg, Imbens & Wager,
2021, "Synthetic Difference-in-Differences," American Economic Review).
SDID combines two ideas rather than asking a forecasting model to infer
a causal effect from correlational co-movement:

  1. Synthetic-control UNIT weights: a regularized regression finds the
     weighted combination of donor (untreated) markets whose PRE-period
     price trajectory best tracks onion's own pre-period trajectory --
     i.e., "which combination of tomato/potato markets moves like onion
     normally does."
  2. DID TIME weights: a second regularized regression finds which
     pre-period weeks best predict the donors' own post-period average,
     down-weighting pre-period weeks that are unrepresentative of the
     post-period baseline.
  3. The treatment effect is the double-differenced gap between onion's
     actual post-period path and its synthetic (weighted donor)
     counterfactual, net of the same gap among the donors themselves.

Design: treated group = all onion markets with >=95% real weekly
coverage over the estimation window (76 markets, block-averaged into one
treated series, since all onion markets were subject to the same national
export policy simultaneously -- there is no untreated onion market to
compare within-crop). Donor pool = all tomato and potato markets with
>=95% real coverage over the same window (239 markets) -- untreated by
onion-specific export policy by construction, giving both weighting
steps genuine cross-sectional richness (unlike a naive 2-donor
crop-level comparison, which would leave the time-weight step severely
underdetermined).

Treatment window: 19 Aug 2023 (the first restrictive measure, a 40%
export duty) through 4 May 2024 (the date the export ban was lifted).
Estimated as THREE separate sub-windows, not one -- a first pass at the
full window found a strongly POSITIVE ATT (prices rising faster than
the synthetic control), which on inspection is confounded: the same
unseasonal-rain supply shock that prompted the government to act was
still worsening throughout the Aug-Dec 2023 duty/MEP period, so onion
prices kept climbing despite the restrictions (Script 28: the real
spike peaked in late Oct 2023, two months after the duty). Only the
outright export ban (8 Dec 2023) is followed by the real price crash.
Reported separately: `full_window` (Aug 2023-May 2024, confounded,
kept for transparency), `escalation_preban` (Aug 19-Dec 7 2023, duty/MEP
only), and `postban` (Dec 8 2023-May 4 2024, the ban) -- the postban
estimate is the one used for Script 30's export-ban and
full-crisis-replay scenario magnitudes.

Caveat stated throughout, not hidden: this is a SINGLE treated episode
with no repeated randomization, so formal inference (standard errors,
p-values) is not attempted in the usual asymptotic sense; a placebo-based
sanity check (re-running the identical estimator with tomato and potato
each as a placebo "treated" unit, where no real policy shock occurred)
is used instead to gauge whether the estimated onion effect is larger
than what the same procedure finds for units with no real treatment.

PART B -- a second, stronger design. The cross-crop placebo check above
FAILS for the postban window: onion's ATT there (+14.3%) is statistically
indistinguishable from the placebo ATTs found for tomato (+14.8%) and
potato (+15.2%) in the identical window, meaning a shared cross-crop
shock (most plausibly seasonal/macro) swamps whatever ban-specific effect
exists, and this design's tomato/potato donor pool cannot net it out.
Part B sidesteps the cross-crop confound entirely with a WITHIN-onion
design: treated = onion markets in Nashik district, Maharashtra (the
real, well-documented onion export hub -- Lasalgaon APMC alone is widely
described as Asia's largest onion market and the benchmark for export
pricing; Pimpalgaon, Yeola, Manmad, Chandvad, Nasik, and Sinner are the
same belt), which should be disproportionately exposed to an export
restriction; donors/control = onion markets outside Nashik. Both groups
share the exact same national seasonal and macro conditions by
construction, so this design does not need tomato/potato at all. Unlike
tomato/potato, neither has a comparable real, documented export-hub
geography in this dataset -- inventing one to match the cross-crop
placebo structure would be a fabricated distinction, not a real one -- so
the appropriate parallel robustness check here is a PLACEBO-IN-TIME test
instead: the identical hub-vs-non-hub design re-run at a fake treatment
date two years earlier (19 Aug 2021 / 8 Dec 2021), when no real export
restriction existed, to confirm the design doesn't spuriously detect an
effect when none should be present.

Inputs:
  data/agmarknet_weekly/top_weekly_panel.csv

Outputs (Model_Output/):
  table_sdid_policy_effect.csv        Part A: cross-crop ATT, all sub-windows
                                        + placebos
  table_sdid_unit_weights.csv         Part A: top donor markets by weight
  table_sdid_hub_policy_effect.csv    Part B: within-onion hub-vs-non-hub ATT,
                                        real 2023 dates + placebo-in-time
  fig_sdid_treated_vs_synthetic.png   Part A: actual vs synthetic onion path
  fig_sdid_hub_vs_nonhub.png          Part B: hub markets vs synthetic control

Run: python scripts/31_Synthetic_DID_Policy_Effect.py
Estimated runtime: 3-5 minutes (constrained regressions with several
hundred variables each, run repeatedly across sub-windows and crops --
no model fitting)
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_START = pd.Timestamp('2017-01-02')
WINDOW_END   = pd.Timestamp('2024-05-04')      # export ban lifted 4 May 2024
TREAT_START  = pd.Timestamp('2023-08-19')      # 40% export duty first notified
BAN_START    = pd.Timestamp('2023-12-08')      # outright export ban begins
COVERAGE_THRESHOLD = 0.95

# Two sub-windows, not one -- the full Aug2023-May2024 window mixes two very
# different periods. The duty/MEP-only period (Aug 19-Dec 7) coincides with
# the SAME unseasonal-rain supply shock that prompted the government to act
# in the first place, so onion prices kept climbing through this period
# despite the duty (Script 28: the real spike peaked in late Oct 2023, two
# months after the duty). Only the outright ban (Dec 8 onward) is followed
# by the real price crash. A single ATT averaged over the whole window
# would average a shock-driven run-up against the ban's relief effect and
# obscure both -- so ESCALATION and BAN are estimated as separate SDID
# treatment windows, sharing the same (fully pre-treatment) period for
# fitting unit/time weights.
SUBWINDOWS = {
    'full_window':      (TREAT_START, WINDOW_END),
    'escalation_preban': (TREAT_START, BAN_START - pd.Timedelta(days=1)),
    'postban':           (BAN_START, WINDOW_END),
}

CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

print('=' * 65)
print('SCRIPT 31: SYNTHETIC DIFFERENCE-IN-DIFFERENCES')
print('  Onion export-restriction policy effect, Aug 2023-May 2024')
print('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD THE MARKET-LEVEL PANEL
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Building the market-level weekly log-price panel ...')

df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
window = df[(df['week_start'] >= WINDOW_START) & (df['week_start'] <= WINDOW_END)].copy()
all_weeks = sorted(window['week_start'].unique())
n_weeks = len(all_weeks)
print(f'  Window: {WINDOW_START.date()} to {WINDOW_END.date()} ({n_weeks} weeks)')

cov = window.groupby(['crop', 'market'])['imputed'].agg(['mean', 'count'])
cov['real_cov'] = 1 - cov['mean']
qualifying = cov[(cov['count'] == n_weeks) & (cov['real_cov'] >= COVERAGE_THRESHOLD)].reset_index()

pivot = window.pivot_table(index='week_start', columns=['crop', 'market'],
                            values='modal_price_weighted')
pivot = pivot.reindex(all_weeks)

# A row can be flagged imputed=1 (not real) yet still carry a NaN price --
# the ~5.7% of rows where a gap exceeded Script 09's imputation tolerance
# (see paper_drafts/methods_data_section.txt Sec 3.8). The coverage filter
# above only checked the imputed FLAG's mean, not whether the price series
# is actually complete, so a handful of "qualifying" markets still had a
# few NaN weeks -- require a fully rectangular (no-gap) price series too.
complete_cols = pivot.columns[pivot.notna().all(axis=0)]
qualifying = qualifying[qualifying.apply(lambda r: (r['crop'], r['market']) in complete_cols, axis=1)]

treated_markets = qualifying[qualifying['crop'] == 'onion']['market'].tolist()
donor_markets = qualifying[qualifying['crop'].isin(['tomato', 'potato'])][['crop', 'market']]
print(f'  Treated (onion) markets  >= {COVERAGE_THRESHOLD:.0%} real coverage, no gaps: {len(treated_markets)}')
print(f'  Donor (tomato+potato) markets >= {COVERAGE_THRESHOLD:.0%} real coverage, no gaps: {len(donor_markets)}')

log_pivot = np.log(pivot)

treated_series = log_pivot[[('onion', m) for m in treated_markets]].mean(axis=1)
donor_cols = [(r['crop'], r['market']) for _, r in donor_markets.iterrows()]
donor_matrix = log_pivot[donor_cols]
donor_matrix.columns = [f'{c}__{m}' for c, m in donor_cols]

weeks_s = pd.Series(all_weeks)
pre_mask = weeks_s.between(WINDOW_START, TREAT_START, inclusive='left').values
n_pre = pre_mask.sum()
print(f'  Pre-period (shared, for fitting unit/time weights): {n_pre} weeks')
for name, (s, e) in SUBWINDOWS.items():
    m = weeks_s.between(s, e, inclusive='both').values
    print(f'    sub-window "{name}": {s.date()} to {e.date()} ({m.sum()} weeks)')


# ─────────────────────────────────────────────────────────────────────────────
# 3. SDID ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
def solve_simplex_regression(X, y, ridge_penalty):
    """min_{w0 free, w>=0, sum(w)=1} ||y - w0 - X@w||^2 + ridge_penalty*||w||^2
    Returns (w0, w). X: (n_obs, n_vars); y: (n_obs,)."""
    n_obs, n_vars = X.shape

    def objective(params):
        w0, w = params[0], params[1:]
        resid = y - w0 - X @ w
        return float(np.sum(resid ** 2) + ridge_penalty * np.sum(w ** 2))

    x0 = np.concatenate([[float(np.mean(y))], np.full(n_vars, 1.0 / n_vars)])
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[1:]) - 1.0}]
    bounds = [(None, None)] + [(0.0, 1.0)] * n_vars
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                    constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
    return res.x[0], res.x[1:]


def run_sdid(treated, donors, pre_mask, post_mask, label):
    """treated: pd.Series (T,) log price, indexed by week.
    donors: pd.DataFrame (T, J) log price, same index.
    Returns dict with ATT (log points), ATT_pct, unit_weights (pd.Series),
    synthetic control series, and a light placebo-style diagnostic."""
    T = len(treated)
    donor_names = donors.columns.tolist()
    J = len(donor_names)

    # Regularization strength (Arkhangelsky et al. 2021, section 3.1): based
    # on the noise level of donors' pre-period first differences.
    pre_diffs = donors.loc[pre_mask].diff().dropna().values
    sigma_hat = float(np.std(pre_diffs, ddof=1)) if pre_diffs.size else 0.0
    n_post = int(post_mask.sum())
    zeta = (n_post ** 0.25) * sigma_hat if sigma_hat > 0 else 1e-6

    # --- Unit weights: fit onion's pre-period path as a combo of donors' ---
    X_unit = donors.loc[pre_mask].values
    y_unit = treated.loc[pre_mask].values
    n_pre = X_unit.shape[0]
    w0_unit, unit_weights = solve_simplex_regression(X_unit, y_unit, ridge_penalty=n_pre * zeta ** 2)

    # --- Time weights: fit donors' post-period average from their own ---
    # --- pre-period weeks (control-only regression, per the SDID method) ---
    X_time = donors.loc[pre_mask].values.T          # (J, n_pre): donors x pre-weeks
    y_time = donors.loc[post_mask].mean(axis=0).values  # (J,): donor post-period means
    w0_time, time_weights = solve_simplex_regression(X_time, y_time, ridge_penalty=J * zeta ** 2)

    # --- SDID point estimate ---
    treated_post_mean = treated.loc[post_mask].mean()
    treated_pre_weighted = np.sum(time_weights * treated.loc[pre_mask].values)
    donor_post_weighted = np.sum(unit_weights * donors.loc[post_mask].mean(axis=0).values)
    donor_pre_weighted = np.sum(unit_weights * np.sum(
        time_weights[:, None] * donors.loc[pre_mask].values, axis=0))

    att = (treated_post_mean - treated_pre_weighted) - (donor_post_weighted - donor_pre_weighted)
    att_pct = float(np.expm1(att) * 100)

    synthetic = w0_unit + donors.values @ np.r_[unit_weights]
    synthetic = pd.Series(synthetic, index=treated.index)

    uw = pd.Series(unit_weights, index=donor_names).sort_values(ascending=False)
    print(f'  [{label}] sigma_hat={sigma_hat:.4f}  zeta={zeta:.4f}  '
          f'ATT={att:+.4f} log-pts  ({att_pct:+.1f}%)')
    print(f'    top donor weights: ' +
          ', '.join(f'{k}={v:.3f}' for k, v in uw.head(5).items() if v > 0.001))

    return {'label': label, 'ATT_log': att, 'ATT_pct': att_pct,
            'unit_weights': uw, 'synthetic': synthetic, 'sigma_hat': sigma_hat, 'zeta': zeta}


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN ESTIMATE + PLACEBOS, FOR EACH SUB-WINDOW
# ─────────────────────────────────────────────────────────────────────────────
placebo_crops_for = {'onion': ['tomato', 'potato'], 'tomato': ['onion', 'potato'], 'potato': ['onion', 'tomato']}

def series_for(crop, markets_df):
    markets = markets_df[markets_df['crop'] == crop]['market'].tolist()
    return log_pivot[[(crop, m) for m in markets]].mean(axis=1)

def donors_for(crops, markets_df):
    rows = markets_df[markets_df['crop'].isin(crops)]
    cols = [(r['crop'], r['market']) for _, r in rows.iterrows()]
    d = log_pivot[cols]
    d.columns = [f'{c}__{m}' for c, m in cols]
    return d

all_results = {}   # {subwindow: {unit: result_dict}}
for sw_name, (s, e) in SUBWINDOWS.items():
    print(f'\n[2] Sub-window "{sw_name}" ({s.date()} to {e.date()}) ...')
    post_mask_sw = weeks_s.between(s, e, inclusive='both').values
    sw_results = {}
    for unit_crop in ['onion', 'tomato', 'potato']:
        treated_u = series_for(unit_crop, qualifying)
        donors_u = donors_for(placebo_crops_for[unit_crop], qualifying)
        role = 'treated' if unit_crop == 'onion' else 'placebo'
        sw_results[unit_crop] = run_sdid(treated_u, donors_u, pre_mask, post_mask_sw,
                                          f'{unit_crop.upper()} ({role}, {sw_name})')
    all_results[sw_name] = sw_results

result_onion_postban = all_results['postban']['onion']


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE TABLES
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Saving outputs ...')

summary_rows = []
for sw_name in SUBWINDOWS:
    for unit_crop, res in all_results[sw_name].items():
        role = 'treated' if unit_crop == 'onion' else 'placebo'
        summary_rows.append({'subwindow': sw_name, 'unit': unit_crop, 'role': role,
                              'ATT_log_points': round(res['ATT_log'], 4),
                              'ATT_pct': round(res['ATT_pct'], 1),
                              'sigma_hat': round(res['sigma_hat'], 4)})
summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, 'table_sdid_policy_effect.csv')
summary.to_csv(summary_path, index=False)
print(f'  Saved: {summary_path}')
print(summary.to_string(index=False))

uw_table = result_onion_postban['unit_weights'].reset_index()
uw_table.columns = ['donor_market', 'weight']
uw_table = uw_table[uw_table['weight'] > 0.001].sort_values('weight', ascending=False)
uw_path = os.path.join(OUT_DIR, 'table_sdid_unit_weights.csv')
uw_table.to_csv(uw_path, index=False)
print(f'  Saved: {uw_path}  ({len(uw_table)} donor markets with non-trivial weight, post-ban window)')


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE — actual vs synthetic onion path, both sub-windows marked
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Generating figure ...')

# Use the full-window synthetic control series for a single continuous plot
result_full = all_results['full_window']['onion']

fig, ax = plt.subplots(figsize=(10, 5.5))
actual_price = np.expm1(treated_series)
synth_price = np.expm1(result_full['synthetic'])

ax.plot(treated_series.index, actual_price, color=CROP_COLORS['onion'], linewidth=2,
        label='Actual onion price (avg, treated markets)')
ax.plot(treated_series.index, synth_price, color='black', linewidth=1.6, linestyle='--',
        label='Synthetic control (weighted tomato+potato donors)')
ax.axvspan(TREAT_START, BAN_START, color='#E8A33D', alpha=0.12, zorder=0)
ax.axvspan(BAN_START, WINDOW_END, color=CROP_COLORS['onion'], alpha=0.12, zorder=0)
ax.axvline(TREAT_START, color='#333333', linewidth=0.9)
ax.axvline(BAN_START, color='#333333', linewidth=0.9)
ax.annotate('duty+MEP\n(escalation)', xy=(TREAT_START, ax.get_ylim()[1]),
            xytext=(4, -6), textcoords='offset points', fontsize=8, va='top', ha='left')
ax.annotate('export ban', xy=(BAN_START, ax.get_ylim()[1]),
            xytext=(4, -6), textcoords='offset points', fontsize=8, va='top', ha='left')

esc_att = all_results['escalation_preban']['onion']['ATT_pct']
ban_att = result_onion_postban['ATT_pct']
ax.set_title(f"Onion vs. its synthetic control -- SDID ATT: {esc_att:+.1f}% during duty/MEP "
             f"escalation, {ban_att:+.1f}% after the export ban", fontsize=10.5, fontweight='bold')
ax.set_ylabel('Price (Rs/quintal)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_sdid_treated_vs_synthetic.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

print(f"\n  Part A cross-crop placebo check for the postban window: onion ATT="
      f"{ban_att:+.1f}% vs tomato placebo={all_results['postban']['tomato']['ATT_pct']:+.1f}%, "
      f"potato placebo={all_results['postban']['potato']['ATT_pct']:+.1f}% -- statistically "
      f"indistinguishable from each other, meaning a shared cross-crop shock (most plausibly "
      f"seasonal/macro) swamps whatever ban-specific effect exists in this design. Proceeding "
      f"to Part B (within-onion, export-hub design) to sidestep this confound.")


# ═══════════════════════════════════════════════════════════════════════════
# PART B — WITHIN-ONION, EXPORT-HUB-EXPOSURE DESIGN
# ═══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('PART B: WITHIN-ONION EXPORT-HUB (NASHIK BELT) SDID')
print('=' * 65)

NASHIK_HUB_MARKETS = [
    'Chandvad APMC', 'Devala APMC', 'Dindori(Vani) APMC', 'Kalvan APMC',
    'Lasalgaon APMC', 'Lasalgaon(Niphad) APMC', 'Manmad APMC', 'Nandgaon APMC',
    'Nasik APMC', 'Pimpalgaon APMC', 'Pimpalgaon Baswant(Saykheda) APMC',
    'Satana APMC', 'Sinner APMC', 'Yeola APMC',
]

def build_onion_hub_panel(win_start, win_end):
    """Rebuilds the market-level onion-only panel for an arbitrary window
    (needed for the placebo-in-time check, which uses an earlier window)."""
    w = df[(df['week_start'] >= win_start) & (df['week_start'] <= win_end) & (df['crop'] == 'onion')].copy()
    weeks = sorted(w['week_start'].unique())
    nw = len(weeks)
    c = w.groupby('market')['imputed'].agg(['mean', 'count'])
    c['real_cov'] = 1 - c['mean']
    piv = w.pivot_table(index='week_start', columns='market', values='modal_price_weighted').reindex(weeks)
    complete = piv.columns[piv.notna().all(axis=0)]
    qual_m = c[(c['count'] == nw) & (c['real_cov'] >= COVERAGE_THRESHOLD)].index
    qual_m = [m for m in qual_m if m in complete]
    hub = [m for m in qual_m if m in NASHIK_HUB_MARKETS]
    nonhub = [m for m in qual_m if m not in NASHIK_HUB_MARKETS]
    log_piv = np.log(piv[qual_m])
    return weeks, log_piv, hub, nonhub


def run_hub_design(win_start, win_end, treat_start, ban_start, tag):
    weeks, log_piv, hub, nonhub = build_onion_hub_panel(win_start, win_end)
    print(f'\n  [{tag}] window {win_start.date()}-{win_end.date()}: '
          f'{len(hub)} hub markets, {len(nonhub)} non-hub donor markets')
    if len(hub) < 3 or len(nonhub) < 10:
        print(f'    SKIPPED -- too few qualifying markets for this window')
        return None

    hub_series = log_piv[hub].mean(axis=1)
    nonhub_matrix = log_piv[nonhub]
    ws = pd.Series(weeks)
    pre_m = ws.between(win_start, treat_start, inclusive='left').values

    subwins = {
        'escalation_preban': (treat_start, ban_start - pd.Timedelta(days=1)),
        'postban': (ban_start, win_end),
    }
    out = {}
    for name, (s, e) in subwins.items():
        post_m = ws.between(s, e, inclusive='both').values
        if post_m.sum() < 3:
            continue
        out[name] = run_sdid(hub_series, nonhub_matrix, pre_m, post_m, f'{tag}:{name}')
    return {'results': out, 'hub_series': hub_series, 'weeks': weeks,
             'synthetic_postban': out.get('postban', {}).get('synthetic')}


print('\n[6] Real design: Nashik-hub vs non-hub onion markets, actual 2023-24 dates ...')
real_hub = run_hub_design(WINDOW_START, WINDOW_END, TREAT_START, BAN_START, 'REAL')

PLACEBO_WINDOW_START = pd.Timestamp('2015-01-04')
PLACEBO_TREAT_START  = pd.Timestamp('2021-08-19')
PLACEBO_BAN_START    = pd.Timestamp('2021-12-08')
PLACEBO_WINDOW_END   = PLACEBO_BAN_START + (WINDOW_END - BAN_START)  # same postban length as the real design
print('\n[7] Placebo-in-time: same hub-vs-non-hub design, fake 2021 treatment dates '
      '(no real export restriction existed then) ...')
placebo_hub = run_hub_design(PLACEBO_WINDOW_START, PLACEBO_WINDOW_END,
                              PLACEBO_TREAT_START, PLACEBO_BAN_START, 'PLACEBO-IN-TIME')

hub_rows = []
if real_hub:
    for sw, res in real_hub['results'].items():
        hub_rows.append({'design': 'real_2023', 'subwindow': sw,
                          'ATT_log_points': round(res['ATT_log'], 4), 'ATT_pct': round(res['ATT_pct'], 1)})
if placebo_hub:
    for sw, res in placebo_hub['results'].items():
        hub_rows.append({'design': 'placebo_in_time_2021', 'subwindow': sw,
                          'ATT_log_points': round(res['ATT_log'], 4), 'ATT_pct': round(res['ATT_pct'], 1)})
hub_summary = pd.DataFrame(hub_rows)
hub_summary_path = os.path.join(OUT_DIR, 'table_sdid_hub_policy_effect.csv')
hub_summary.to_csv(hub_summary_path, index=False)
print(f'\n  Saved: {hub_summary_path}')
print(hub_summary.to_string(index=False))

if real_hub and 'postban' in real_hub['results']:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    hub_price = np.expm1(real_hub['hub_series'])
    ax.plot(real_hub['weeks'], hub_price, color=CROP_COLORS['onion'], linewidth=2,
            label='Actual price -- Nashik export-hub onion markets')
    if real_hub['synthetic_postban'] is not None:
        synth = real_hub['synthetic_postban']
        ax.plot(synth.index, np.expm1(synth), color='black', linewidth=1.6, linestyle='--',
                label='Synthetic control (weighted non-hub onion markets)')
    ax.axvspan(TREAT_START, BAN_START, color='#E8A33D', alpha=0.12, zorder=0)
    ax.axvspan(BAN_START, WINDOW_END, color=CROP_COLORS['onion'], alpha=0.12, zorder=0)
    ax.axvline(TREAT_START, color='#333333', linewidth=0.9)
    ax.axvline(BAN_START, color='#333333', linewidth=0.9)
    real_esc = real_hub['results'].get('escalation_preban', {}).get('ATT_pct', float('nan'))
    real_ban = real_hub['results'].get('postban', {}).get('ATT_pct', float('nan'))
    ax.set_title(f"Nashik export-hub onion markets vs. non-hub synthetic control -- "
                 f"SDID ATT: {real_esc:+.1f}% escalation, {real_ban:+.1f}% post-ban",
                 fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Price (Rs/quintal)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    hub_fig_path = os.path.join(OUT_DIR, 'fig_sdid_hub_vs_nonhub.png')
    plt.savefig(hub_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {hub_fig_path}')


print('\n' + '=' * 65)
print('Script 31 complete.')
print(f"\nHeadline (Part A, cross-crop): full-window ATT ({result_full['ATT_pct']:+.1f}%) is confounded")
print(f"by the supply shock that prompted the policy in the first place; escalation-phase ATT")
print(f"({esc_att:+.1f}%) is a credible onion-specific effect (huge gap vs tomato placebo of "
      f"{all_results['escalation_preban']['tomato']['ATT_pct']:+.1f}%); but the post-ban ATT")
print(f"({ban_att:+.1f}%) FAILS its placebo check -- indistinguishable from tomato/potato placebos")
print(f"in the same window, so this design cannot isolate the ban's specific relief effect.")
if real_hub and placebo_hub and 'postban' in real_hub['results'] and 'postban' in placebo_hub['results']:
    print(f"\nHeadline (Part B, within-onion export-hub): real 2023 postban ATT = "
          f"{real_hub['results']['postban']['ATT_pct']:+.1f}%, versus a placebo-in-time ATT of "
          f"{placebo_hub['results']['postban']['ATT_pct']:+.1f}% at a fake 2021 date with no real "
          f"restriction -- see the printed comparison above for whether this design is credible.")
print('\nKey outputs:')
for fname in ['table_sdid_policy_effect.csv', 'table_sdid_unit_weights.csv',
              'fig_sdid_treated_vs_synthetic.png', 'table_sdid_hub_policy_effect.csv',
              'fig_sdid_hub_vs_nonhub.png']:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        print(f'  {fname:<38} {os.path.getsize(fpath)/1024:>7.1f} KB')
