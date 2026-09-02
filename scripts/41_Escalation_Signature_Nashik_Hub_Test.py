# -*- coding: utf-8 -*-
"""
Script 41 — Escalation-Signature Head: Nashik-Hub Localized-Signal Test
=============================================================================
Direct test of a hypothesis raised while digging into why onion's Sep-2020
ban episode failed Script 40's placebo-in-time test even after switching to
a data-driven detection window: the 2020 ban's own documented trigger was
"heavy rainfall and floods in the Nashik region" specifically -- a LOCALIZED
shock. Script 40's escalation-signature model is built entirely on crop-
level (nationally-averaged) price, which would dilute a geographically
concentrated shock even if it was severe enough locally to trigger a
national policy response. This mirrors the exact reasoning behind Script
31 Part B's within-onion Nashik-hub-vs-non-hub SDID design.

Test: rerun the IDENTICAL escalation-detection pipeline from Script 40 v4
(same data-driven labelling rule, same per-episode leave-one-out structure,
same placebo-in-time significance test), but with the crop-level national
price series replaced by a NASHIK-HUB-ONLY price series (12 of the 14
markets from Script 31's NASHIK_HUB_MARKETS list are present in the
longhistory panel with usable coverage; Kalvan and Satana APMC are not in
that panel). Arrivals and climate features stay at the national/crop level
(unchanged from Script 40) -- the hypothesis is specifically about price
being diluted by national averaging, not about arrivals or climate.

If the hypothesis is right: onion 2020's detected escalation should look
stronger and more sustained at the Nashik-hub level than it did nationally,
and should pass (or come closer to passing) the placebo-in-time test.
If not: 2020 will look just as weak locally as it did nationally, which
would argue AGAINST the "localized shock, diluted nationally" explanation
and toward "this episode's price response was genuinely more muted, full
stop" instead. Both outcomes are reported as-is.

Inputs:
  data/agmarknet_weekly/longhistory/top_weekly_panel_longhistory.csv
  data/satellite_climate/crop_weekly_features.csv

Outputs (Model_Output/):
  table_escalation_signature_nashik_loeo.csv     within-crop LOEO metrics, Nashik-hub price
  table_escalation_signature_nashik_placebo.csv  placebo-in-time test, Nashik-hub price
  fig_escalation_signature_nashik_vs_national.png  side-by-side comparison, all 3 episodes

Run: python scripts/41_Escalation_Signature_Nashik_Hub_Test.py
"""

import io, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LH_FILE  = os.path.join(BASE, 'data', 'agmarknet_weekly', 'longhistory', 'top_weekly_panel_longhistory.csv')
SAT_FILE = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LOOKBACK_WEEKS = 20
DEV_THRESHOLD = 0.15
N_OOF_FOLDS = 8   # time-block CV folds for the placebo test's full-history OOF scoring (matches Script 40)

# Same list as scripts/31_Synthetic_DID_Policy_Effect.py's NASHIK_HUB_MARKETS.
# Kalvan APMC and Satana APMC are not present in the longhistory panel
# (insufficient coverage to have qualified for Script 32's own filter);
# the remaining 12 give a solid hub-level aggregate.
NASHIK_HUB_MARKETS = [
    'Chandvad APMC', 'Devala APMC', 'Dindori(Vani) APMC', 'Kalvan APMC',
    'Lasalgaon APMC', 'Lasalgaon(Niphad) APMC', 'Manmad APMC', 'Nandgaon APMC',
    'Nasik APMC', 'Pimpalgaon APMC', 'Pimpalgaon Baswant(Saykheda) APMC',
    'Satana APMC', 'Sinner APMC', 'Yeola APMC',
]

EPISODES = [
    dict(name='onion_2019', first_action=pd.Timestamp('2019-09-29'), label='Onion Sep-2019 ban'),
    dict(name='onion_2020', first_action=pd.Timestamp('2020-09-14'), label='Onion Sep-2020 ban'),
    dict(name='onion_2023', first_action=pd.Timestamp('2023-08-19'),
         label='Onion Aug-2023 duty (first action of 2023-24 sequence)'),
]
COLOR_NASHIK = '#7B2C8E'
COLOR_NATIONAL = '#A9A9A9'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                      'axes.spines.top': False, 'axes.spines.right': False})

print('=' * 65)
print('SCRIPT 41: ESCALATION-SIGNATURE HEAD -- NASHIK-HUB LOCALIZED TEST')
print('=' * 65)
print('\nTesting: does onion 2020\'s escalation signal, weak at the national-average')
print('level (Script 40), look stronger using Nashik-hub-only price -- consistent')
print('with the 2020 ban\'s documented trigger being a localized flood shock?\n')


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD NASHIK-HUB-ONLY PRICE SERIES + national arrivals/climate (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
print('[1] Building Nashik-hub-only onion price series ...')
lh = pd.read_csv(LH_FILE, parse_dates=['week_start'])
onion_all = lh[lh['crop'] == 'onion']
hub_present = [m for m in NASHIK_HUB_MARKETS if m in onion_all['market'].unique()]
print(f'  {len(hub_present)}/{len(NASHIK_HUB_MARKETS)} hub markets present: {hub_present}')

hub_price = (onion_all[onion_all['market'].isin(hub_present)]
             .groupby('week_start')['modal_price_weighted'].mean()
             .rename('price').reset_index())
national_arrivals = (onion_all.groupby('week_start')['arrivals_tonnes_week'].mean()
                      .rename('arrivals').reset_index())
data = hub_price.merge(national_arrivals, on='week_start', how='left').sort_values('week_start')

sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'],
                   usecols=['crop', 'week_start', 'era5_heat_35', 'chirps_rain_mm'])
sat_onion = sat[sat['crop'] == 'onion'].drop(columns=['crop'])
data = data.merge(sat_onion, on='week_start', how='left')
print(f'  {len(data):,} weeks, {data["week_start"].min().date()} to {data["week_start"].max().date()}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES — identical construction to Script 40 v4
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Engineering features (Nashik-hub price, national arrivals/climate) ...')
data['price_roll4_pct'] = data['price'].pct_change(4)
data['price_roll8_pct'] = data['price'].pct_change(8)
data['arrivals_roll4_pct'] = data['arrivals'].pct_change(4)
data['era5_heat_35_roll4'] = data['era5_heat_35'].rolling(4, min_periods=2).mean()
data['chirps_rain_mm_roll4'] = data['chirps_rain_mm'].rolling(4, min_periods=2).mean()

data['iso_wk'] = data['week_start'].dt.isocalendar().week.astype(int)
data['iso_yr'] = data['week_start'].dt.isocalendar().year.astype(int)
yearly = (data.groupby(['iso_wk', 'iso_yr'])['price'].mean().rename('price_yr').reset_index()
              .sort_values(['iso_wk', 'iso_yr']))
yearly['norm'] = yearly.groupby('iso_wk')['price_yr'].transform(lambda x: x.shift(1).expanding().mean())
data = data.merge(yearly[['iso_wk', 'iso_yr', 'norm']], on=['iso_wk', 'iso_yr'], how='left')
data['price_vs_seasonal_norm'] = (data['price'] - data['norm']) / data['norm']
data = data.drop(columns=['iso_wk', 'iso_yr', 'norm'])

# FIXED 2026-09-02 (audit finding, confirmed, same fix as Script 40):
# price_vs_seasonal_norm is excluded from FEATURES -- it's the exact
# quantity thresholded against DEV_THRESHOLD to define the label itself
# (Section 3 below), so feeding it to the classifier as a raw input gives
# it a near-deterministic shortcut within any episode's labelled window.
# It stays in `data` for label construction (and the placebo scoring
# below); only excluded from FEATURES.
FEATURES = ['price_roll4_pct', 'price_roll8_pct',
            'arrivals_roll4_pct', 'era5_heat_35_roll4', 'chirps_rain_mm_roll4']
before = len(data)
# Drop on FEATURES plus price_vs_seasonal_norm so removing it from FEATURES
# doesn't change which rows survive (see Script 40's identical fix note).
data = data.dropna(subset=FEATURES + ['price_vs_seasonal_norm']).reset_index(drop=True)
print(f'  {len(data):,} / {before:,} rows have complete features')


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABEL — same data-driven rule as Script 40 v4
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[3] Labelling (deviation-from-Nashik-norm >= {DEV_THRESHOLD:.0%}, {MAX_LOOKBACK_WEEKS}wk ceiling) ...')
data['label'] = 0
data['episode'] = ''
for ep in EPISODES:
    win_start = ep['first_action'] - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS)
    mask = ((data['week_start'] >= win_start) & (data['week_start'] < ep['first_action']) &
            (data['price_vs_seasonal_norm'] >= DEV_THRESHOLD))
    data.loc[mask, 'label'] = 1
    data.loc[mask, 'episode'] = ep['name']
    n = mask.sum()
    print(f'  [{ep["name"]:12s}] {n} weeks detected (Nashik-hub price)')


# ─────────────────────────────────────────────────────────────────────────────
# 4. WITHIN-EPISODE LEAVE-ONE-OUT + PLACEBO-IN-TIME TEST — identical logic
# to Script 40 v4, single "crop" (Nashik-hub onion) this time.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Leave-one-episode-out validation (Nashik-hub price) ...')

def fit_lgbm(X_train, y_train):
    n_pos = y_train.sum()
    spw = (len(y_train) - n_pos) / max(n_pos, 1)
    model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, num_leaves=15,
                                learning_rate=0.05, scale_pos_weight=spw,
                                min_child_samples=5, verbose=-1)
    model.fit(X_train, y_train)
    return model

loeo_rows = []
episode_models = {}
data['score'] = np.nan

for ep in EPISODES:
    test_mask = data['episode'] == ep['name']
    around = ((data['week_start'] >= ep['first_action'] - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS * 2)) &
              (data['week_start'] <= ep['first_action'] + pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS // 2)))
    # FIXED 2026-09-02 (audit finding, inherited from Script 40 -- see its
    # matching 2026-09-02 fix comment): these padding negatives satisfied
    # `episode != ep['name']`, identical to train_mask's own condition, so
    # they were being trained AND tested on. Excluded from train_mask now.
    extra_test_negatives = around & (data['episode'] != ep['name']) & (data['label'] == 0)
    test_mask_full = test_mask | extra_test_negatives
    train_mask = (data['episode'] != ep['name']) & (~extra_test_negatives)

    X_train, y_train = data.loc[train_mask, FEATURES], data.loc[train_mask, 'label']
    X_test, y_test = data.loc[test_mask_full, FEATURES], data.loc[test_mask_full, 'label']

    model = fit_lgbm(X_train, y_train)
    episode_models[ep['name']] = model
    scores = model.predict_proba(X_test)[:, 1]
    data.loc[X_test.index, 'score'] = scores

    win_start = ep['first_action'] - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS * 1.5)
    win_end = ep['first_action'] + pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS // 2)
    plot_mask = (data['week_start'] >= win_start) & (data['week_start'] <= win_end)
    data.loc[plot_mask, 'score'] = model.predict_proba(data.loc[plot_mask, FEATURES])[:, 1]

    if 0 < y_test.sum() < len(y_test):
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
    else:
        auc, ap = np.nan, np.nan
    loeo_rows.append({'episode': ep['name'], 'label': ep['label'], 'n_test_weeks': len(y_test),
                       'n_test_positive': int(y_test.sum()),
                       'auc': round(auc, 3) if pd.notna(auc) else None,
                       'avg_precision': round(ap, 3) if pd.notna(ap) else None})
    print(f'  [{ep["name"]:12s}] test n={len(y_test):3d} (pos={int(y_test.sum())})  AUC={auc:.3f}  AP={ap:.3f}')

loeo_df = pd.DataFrame(loeo_rows)
loeo_path = os.path.join(OUT_DIR, 'table_escalation_signature_nashik_loeo.csv')
loeo_df.to_csv(loeo_path, index=False)
print(f'\n  Saved: {loeo_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 4.5 FULL-HISTORY OUT-OF-FOLD SCORING for the placebo-in-time test
# ─────────────────────────────────────────────────────────────────────────────
# FIXED 2026-09-02 (audit finding, inherited from Script 40 -- see its
# matching 2026-09-02 fix comment for the full rationale). The placebo test
# below used to reuse each episode's LOEO model to score the ENTIRE Nashik-
# hub history -- nearly every placebo candidate week was in that model's
# training set (only the one held-out episode's own window was genuinely
# unseen). Same fix as Script 40: K-fold time-block CV across the full
# series, boundaries nudged so no episode's own window is split across
# folds, giving every week (episode or background) a genuinely out-of-fold
# score.
print(f'\n[4.5] Building full-history out-of-fold scores for the placebo test '
      f'(K={N_OOF_FOLDS}-fold time-block CV, episode-window-safe) ...')


def build_oof_fold_map(d, episodes, k):
    weeks = d['week_start'].sort_values().unique()
    protected = [(ep['first_action'] - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS), ep['first_action'])
                 for ep in episodes]

    def in_protected(ts):
        return any(lo <= ts < hi for lo, hi in protected)

    raw_boundaries = np.linspace(0, len(weeks), k + 1).astype(int)[1:-1]
    boundaries = set()
    for b in raw_boundaries:
        idx = b
        while idx < len(weeks) and in_protected(pd.Timestamp(weeks[idx])):
            idx += 1
        if idx >= len(weeks):
            idx = b
            while idx > 0 and in_protected(pd.Timestamp(weeks[idx - 1])):
                idx -= 1
        if 0 < idx < len(weeks):
            boundaries.add(idx)
    edges = [0] + sorted(boundaries) + [len(weeks)]
    fold_of_week = {}
    for i in range(len(edges) - 1):
        for w in weeks[edges[i]:edges[i + 1]]:
            fold_of_week[w] = i
    return d['week_start'].map(fold_of_week)


data['score_oof'] = np.nan
fold_of_row = build_oof_fold_map(data, EPISODES, N_OOF_FOLDS)
n_folds_actual = fold_of_row.nunique()
for fold in sorted(fold_of_row.unique()):
    train_idx = data.index[fold_of_row != fold]
    test_idx = data.index[fold_of_row == fold]
    y_tr = data.loc[train_idx, 'label']
    if y_tr.nunique() < 2:
        print(f'  WARNING: OOF fold {fold} has a degenerate training label set '
              f'({y_tr.nunique()} class(es)) -- skipping, rows left unscored.')
        continue
    oof_model = fit_lgbm(data.loc[train_idx, FEATURES], y_tr)
    data.loc[test_idx, 'score_oof'] = oof_model.predict_proba(data.loc[test_idx, FEATURES])[:, 1]
n_scored = data['score_oof'].notna().sum()
print(f'  {n_folds_actual} time-block folds (target K={N_OOF_FOLDS}), '
      f'{n_scored}/{len(data)} rows scored out-of-fold')


print('\n[5] Placebo-in-time significance test (Nashik-hub price) ...')

def detected_intensity(df, end_date):
    win_start = end_date - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS)
    mask = ((df['week_start'] >= win_start) & (df['week_start'] < end_date) &
            (df['price_vs_seasonal_norm'] >= DEV_THRESHOLD))
    return float(df.loc[mask, 'score_oof'].mean()) if mask.sum() else 0.0

placebo_rows = []
placebo_detail = {}
ep_dates = [e['first_action'] for e in EPISODES]
for ep in EPISODES:
    real_intensity = detected_intensity(data, ep['first_action'])

    weeks = data['week_start'].tolist()
    placebo_intensities = []
    i = MAX_LOOKBACK_WEEKS
    while i < len(weeks):
        cand_end = weeks[i]
        too_close = any(abs((cand_end - dt).days) < MAX_LOOKBACK_WEEKS * 2 * 7 for dt in ep_dates)
        if not too_close:
            placebo_intensities.append(detected_intensity(data, cand_end))
        i += MAX_LOOKBACK_WEEKS

    placebo_intensities = np.array(placebo_intensities)
    n_placebo = len(placebo_intensities)
    n_extreme = int((placebo_intensities >= real_intensity).sum())
    # +1/+1 permutation-test adjustment (same fix as Script 40 / Scripts
    # 38/38b): the real/observed window is always at least as extreme as
    # itself, so the theoretical floor is 1/(n_placebo+1), not 0.
    p_value = (n_extreme + 1) / (n_placebo + 1) if n_placebo else np.nan
    placebo_detail[ep['name']] = placebo_intensities
    placebo_rows.append({'episode': ep['name'], 'label': ep['label'],
                          'real_window_intensity': round(float(real_intensity), 5),
                          'n_placebo_windows': n_placebo, 'n_placebo_as_extreme': n_extreme,
                          'p_value': round(float(p_value), 3) if pd.notna(p_value) else None})
    print(f'  [{ep["name"]:12s}] real intensity={real_intensity:.5f}  vs {n_placebo} placebo candidates '
          f'-- {n_extreme} as extreme  p={p_value:.3f}')

placebo_df = pd.DataFrame(placebo_rows)
placebo_path = os.path.join(OUT_DIR, 'table_escalation_signature_nashik_placebo.csv')
placebo_df.to_csv(placebo_path, index=False)
print(f'\n  Saved: {placebo_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON FIGURE — Nashik-hub score trajectory vs. national-level score
# (from Script 40's already-saved output) for all 3 episodes, side by side.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6] Generating Nashik-hub vs. national comparison figure ...')
national_scores = pd.read_csv(os.path.join(OUT_DIR, 'table_escalation_signature_scores_percrop.csv'),
                               parse_dates=['week_start'])
national_scores = national_scores[national_scores['crop'] == 'onion']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, ep in zip(axes, EPISODES):
    win_start = ep['first_action'] - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS * 1.5)
    win_end = ep['first_action'] + pd.Timedelta(weeks=10)

    nat_sub = national_scores[(national_scores['week_start'] >= win_start) &
                               (national_scores['week_start'] <= win_end)].sort_values('week_start')
    hub_sub = data[(data['week_start'] >= win_start) & (data['week_start'] <= win_end)].sort_values('week_start')

    ax.plot(nat_sub['week_start'], nat_sub['score'].clip(lower=1e-6), color=COLOR_NATIONAL,
            linewidth=1.6, linestyle='--', label='National-average price (Script 40)')
    ax.plot(hub_sub['week_start'], hub_sub['score'].clip(lower=1e-6), color=COLOR_NASHIK,
            linewidth=2.0, label='Nashik-hub price only (this script)')
    ax.axvline(ep['first_action'], color='#333333', linewidth=1.1, linestyle='--')
    pval = placebo_df.loc[placebo_df['episode'] == ep['name'], 'p_value'].iloc[0]
    ax.set_title(f"{ep['label']}\n(Nashik-hub placebo p={pval:.3f})", fontsize=9, fontweight='bold')
    ax.set_ylabel('Score (log scale)')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(axis='y', alpha=0.25, which='both')
    ax.legend(fontsize=7, loc='lower left', frameon=False)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_escalation_signature_nashik_vs_national.png')
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {fig_path}')

print('\n' + '=' * 65)
print('Script 41 complete.')
onion_2020_nat_p = None  # printed for reference only; see table_escalation_signature_placebo_test.csv
print('\nCompare table_escalation_signature_nashik_placebo.csv (this script) against')
print('table_escalation_signature_placebo_test.csv (Script 40, national-level) for')
print('onion_2020 specifically -- that comparison is the actual answer to the hypothesis.')
