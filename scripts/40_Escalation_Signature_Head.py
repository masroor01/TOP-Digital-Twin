# -*- coding: utf-8 -*-
"""
Script 40 — Escalation-Signature Head (per-crop prototype, v4)
=============================================================================
Fourth iteration (see git history and the conversation record for the full
design discussion). v4 replaces the fixed 10-week lookback with a DATA-
DRIVEN window, directly motivated by digging into why onion 2020 failed the
v3 placebo test (p=0.198) while 2019 and 2023 passed decisively (p=0.000):

  Checking week-by-week deviation from the crop's own expanding seasonal
  norm showed 2019 and 2023 both have SUSTAINED elevated deviation across
  their entire pre-intervention window, but 2020 has a genuinely different
  SHAPE -- mildly elevated at the start, dipping back to ~0% (even briefly
  negative) in the middle, then re-escalating sharply only in the final
  1-2 weeks before the ban. A fixed 10-week block forces all three episodes
  into the same rigid shape, which dilutes 2020's real (if late and short)
  signal with several genuinely-normal weeks, and also truncates 2019's
  window too early (its deviation was already +50% at what the fixed
  window called "day one").

  Fix: label a week as "escalation" if its price deviates from the crop's
  own seasonal norm by at least DEV_THRESHOLD, anywhere within a generous
  MAX_LOOKBACK_WEEKS ceiling before the intervention -- not a fixed
  contiguous block. Window length and shape now emerge from the data per
  episode instead of being imposed uniformly. The placebo-in-time test
  (Section 5) applies the IDENTICAL detection rule to placebo candidate
  dates, so the comparison stays apples-to-apples.

v3 addition, still in effect: a placebo-in-time significance test (Section
5): for each episode's model, every non-overlapping candidate window across
that crop's FULL available history is scored the same way as the real pre-
intervention window, and the real window's rank against that whole null
distribution gives a genuine permutation-style p-value -- directly
analogous to the in-space placebo test already used for the SDID postban
ATT (Script 38), applied along the time axis instead of across markets.

v2 changes (both requested directly), still in effect:

  1. Every available verified escalation episode is now used, including
     potato's June 2014 MEP ($450/MT, DGFT Notification 85(RE-2013)/
     2009-2014) -- v1 excluded it because it predates the 2017-2026 main
     panel. This version switches to the LONGHISTORY panel (2003-2026,
     all 3 crops) as its single data source instead, so potato's episode
     no longer needs special-casing.
  2. Validation is now done SEPARATELY PER CROP (within-crop leave-one-
     episode-out), not pooled across crops. v1 pooled all 4 episodes into
     one training set regardless of crop, which is exactly what caused
     tomato's held-out scores to calibrate so poorly (its one held-out
     fold had ZERO tomato positive examples in training -- only onion).
     Per-crop separation makes that problem visible rather than hidden
     inside a pooled AUC number, and gives onion (which has 3 episodes)
     a genuine same-crop leave-one-out test for the first time.

v5 addition (2026-08-19, after a deck fact-check found the deck/disclosure
documents were describing tomato and potato's IN-SAMPLE fits with the same
confidence as onion's genuine held-out test -- a real, undisclosed gap):
added 3 more real, primary-source-verified episodes (tomato x2, potato x1),
researched and citation-checked specifically to fix this. The held_out/
in_sample branching in Sections 4-6 was ALREADY keyed off len(crop_episodes)
>= 2, not off crop name, so it required no logic change -- but Section 5's
placebo score_type label and Section 6's figure grouping WERE hardcoded to
'onion' by name; both fixed here to key off episode count like Section 4
already did, since otherwise tomato/potato's new genuine held-out folds
would still have been mislabeled 'in_sample' in the output.

Episode count by crop after v5:
  Onion:  3 episodes (2019, 2020, 2023) -- genuine within-crop leave-one-
          episode-out (3 folds, train on 2, test on 1).
  Tomato: 3 episodes (2023, 2024, 2025) -- NEW: now also a genuine
          within-crop leave-one-out. 2024 actually had TWO candidate
          spikes, checked against the real weekly price series before
          deciding what to do with them: price peaked ~5,275 on
          2024-07-15, fully receded to a trough of ~2,490 by 2024-09-02 (a
          real recovery, not one sustained crisis), then re-spiked to
          ~5,174 on 2024-10-07 -- genuinely two independent crises, not
          one double-counted. Only the July one (2024-07-29, NCCF
          Delhi-NCR retail sale) was added here, deliberately: the two
          intervention dates are 70 days apart, inside MAX_LOOKBACK_WEEKS's
          ~140-day ceiling, so including both risked one episode's lookback
          window swallowing the other's recovery/response tail and
          contaminating the leave-one-out test. The October spike is just
          as real and could become a 4th tomato episode later if that
          overlap risk is examined properly (e.g. shortening the lookback
          ceiling, or checking the actual detected-week spans don't
          intersect) -- left out for now rather than risk a silently
          contaminated fold. 2025-08-04 (NCCF Azadpur Mandi procurement,
          PIB PRID=2154078) added as the 3rd episode; it sits ~370 days
          from both 2023 and 2024's dates, comfortably clear.
  Potato: 2 episodes (2014, 2016) -- NEW: genuine within-crop leave-one-out
          now possible, but only 2 folds (train on 1, test on 1) -- a much
          thinner test than onion/tomato's 3 folds each, honestly reflected
          in n_train_episodes=1 in the output table, not hidden. 2016-07-26
          MEP $360/MT reintroduction (DGFT Notification 15) added; sourced
          from an official APEDA-hosted PDF, a stronger primary source than
          2014's episode has.

A real 4th tomato candidate (Oct-2019, a Centre-directed Mother Dairy price
cap) was investigated and deliberately EXCLUDED: multiple independent news
outlets corroborate it with consistent detail, but no PIB primary-source
citation could be found for it, unlike every other episode here. Left out
to keep this list's verification standard uniform; worth adding later if a
primary citation turns up.

This asymmetry -- onion and tomato now validated properly, potato only
partially -- is itself an informative result: it is a direct, mechanical
consequence of episode count, not a modelling failure, and mirrors the
same n-is-too-small caveat that has applied throughout this project's
causal-inference work (Section 8) and earlier versions of this prototype.

Inputs:
  data/agmarknet_weekly/longhistory/top_weekly_panel_longhistory.csv
  data/satellite_climate/crop_weekly_features.csv

Outputs (Model_Output/):
  table_escalation_signature_loeo_percrop.csv   within-crop LOEO metrics (onion + tomato + potato)
  table_escalation_signature_scores_percrop.csv full weekly held-out/in-sample score series
  fig_escalation_signature_percrop_heldout.png   all held-out folds (any crop with >=2 episodes)
  fig_escalation_signature_percrop_single.png    any crop with exactly 1 episode (in-sample, labelled)

Run: python scripts/40_Escalation_Signature_Head.py
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

LOOKBACK_WEEKS = 10          # kept for the wide plotting window only (display range)
MAX_LOOKBACK_WEEKS = 20      # ceiling within which an escalation week can be detected
DEV_THRESHOLD = 0.15         # price must be >=15% above the crop's own seasonal norm
N_OOF_FOLDS = 8               # time-block CV folds for Section 4.5/5's full-history OOF scoring

EPISODES = [
    dict(name='onion_2019',  crop='onion',  first_action=pd.Timestamp('2019-09-29'),
         label='Onion Sep-2019 ban'),
    dict(name='onion_2020',  crop='onion',  first_action=pd.Timestamp('2020-09-14'),
         label='Onion Sep-2020 ban'),
    dict(name='onion_2023',  crop='onion',  first_action=pd.Timestamp('2023-08-19'),
         label='Onion Aug-2023 duty (first action of 2023-24 sequence)'),
    dict(name='tomato_2023', crop='tomato', first_action=pd.Timestamp('2023-07-20'),
         label='Tomato Jul-2023 NCCF/NAFED procurement'),
    dict(name='tomato_2024', crop='tomato', first_action=pd.Timestamp('2024-07-29'),
         label='Tomato Jul-2024 NCCF Delhi-NCR retail sale (PIB PRID=2038421)'),
    dict(name='tomato_2025', crop='tomato', first_action=pd.Timestamp('2025-08-04'),
         label='Tomato Aug-2025 NCCF Azadpur Mandi procurement (PIB PRID=2154078)'),
    dict(name='potato_2014', crop='potato', first_action=pd.Timestamp('2014-06-26'),
         label='Potato Jun-2014 MEP ($450/MT, DGFT 85(RE-2013)/2009-2014)'),
    dict(name='potato_2016', crop='potato', first_action=pd.Timestamp('2016-07-26'),
         label='Potato Jul-2016 MEP $360/MT (DGFT Notification 15/26.07.2016)'),
]
CROPS = ['tomato', 'onion', 'potato']
CROP_COLORS = {'tomato': '#C0392B', 'onion': '#7B2C8E', 'potato': '#A07020'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                      'axes.spines.top': False, 'axes.spines.right': False})

print('=' * 65)
print('SCRIPT 40 v2: ESCALATION-SIGNATURE HEAD (per-crop, all episodes)')
print('=' * 65)
_counts_preview = {c: len([e for e in EPISODES if e['crop'] == c]) for c in CROPS}
print('\nEpisodes by crop: ' + ', '.join(f'{c} x{n}' for c, n in _counts_preview.items())
      + ' -- within-crop leave-one-out where a crop has >=2 episodes,')
print('reported as an in-sample fit only (clearly labelled) where a crop has just 1.\n')


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD CROP-LEVEL WEEKLY SERIES from the longhistory panel (2003-2026,
# single consistent source for all 3 crops, needed for potato 2014).
# ─────────────────────────────────────────────────────────────────────────────
print('[1] Building crop-level weekly series from the longhistory panel ...')
df = pd.read_csv(LH_FILE, parse_dates=['week_start'])
crop_weekly = (df.groupby(['crop', 'week_start'])
                 .agg(price=('modal_price_weighted', 'mean'),
                      arrivals=('arrivals_tonnes_week', 'mean'))
                 .reset_index())

sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'],
                   usecols=['crop', 'week_start', 'era5_heat_35', 'chirps_rain_mm'])
crop_weekly = crop_weekly.merge(sat, on=['crop', 'week_start'], how='left')
crop_weekly = crop_weekly.sort_values(['crop', 'week_start']).reset_index(drop=True)
print(f'  {len(crop_weekly):,} crop-week rows, {crop_weekly["week_start"].min().date()} '
      f'to {crop_weekly["week_start"].max().date()}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES — identical construction to v1 (all backward-looking only)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Engineering features (rolling acceleration, expanding seasonal norm) ...')

def add_features(g):
    g = g.sort_values('week_start').copy()
    g['price_roll4_pct'] = g['price'].pct_change(4)
    g['price_roll8_pct'] = g['price'].pct_change(8)
    g['arrivals_roll4_pct'] = g['arrivals'].pct_change(4)
    g['era5_heat_35_roll4'] = g['era5_heat_35'].rolling(4, min_periods=2).mean()
    g['chirps_rain_mm_roll4'] = g['chirps_rain_mm'].rolling(4, min_periods=2).mean()

    g['iso_wk'] = g['week_start'].dt.isocalendar().week.astype(int)
    g['iso_yr'] = g['week_start'].dt.isocalendar().year.astype(int)
    yearly = (g.groupby(['iso_wk', 'iso_yr'])['price'].mean()
                .rename('price_yr').reset_index()
                .sort_values(['iso_wk', 'iso_yr']))
    yearly['price_seasonal_norm'] = (yearly.groupby('iso_wk')['price_yr']
                                            .transform(lambda x: x.shift(1).expanding().mean()))
    g = g.merge(yearly[['iso_wk', 'iso_yr', 'price_seasonal_norm']], on=['iso_wk', 'iso_yr'], how='left')
    g['price_vs_seasonal_norm'] = (g['price'] - g['price_seasonal_norm']) / g['price_seasonal_norm']
    return g

crop_weekly = pd.concat([add_features(crop_weekly[crop_weekly['crop'] == c]) for c in CROPS], ignore_index=True)
crop_weekly = crop_weekly.drop(columns=['iso_wk', 'iso_yr', 'price_seasonal_norm'])

FEATURES = ['price_roll4_pct', 'price_roll8_pct', 'price_vs_seasonal_norm',
            'arrivals_roll4_pct', 'era5_heat_35_roll4', 'chirps_rain_mm_roll4']

before = len(crop_weekly)
crop_weekly = crop_weekly.dropna(subset=FEATURES)
print(f'  {len(crop_weekly):,} / {before:,} rows have complete features '
      f'(first ~2 years of each crop lost to the expanding seasonal norm)')


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABEL — data-driven: a week is "escalation" if its price is >=
# DEV_THRESHOLD above the crop's own seasonal norm, anywhere within
# MAX_LOOKBACK_WEEKS of the intervention (not a fixed contiguous block).
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Labelling escalation weeks (data-driven: deviation-from-norm >= '
      f'{DEV_THRESHOLD:.0%}, within a {MAX_LOOKBACK_WEEKS}-week ceiling) ...')
crop_weekly['label'] = 0
crop_weekly['episode'] = ''
for ep in EPISODES:
    win_start = ep['first_action'] - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS)
    mask = ((crop_weekly['crop'] == ep['crop']) &
            (crop_weekly['week_start'] >= win_start) &
            (crop_weekly['week_start'] < ep['first_action']) &
            (crop_weekly['price_vs_seasonal_norm'] >= DEV_THRESHOLD))
    n = mask.sum()
    crop_weekly.loc[mask, 'label'] = 1
    crop_weekly.loc[mask, 'episode'] = ep['name']
    detected_weeks = sorted(crop_weekly.loc[mask, 'week_start'].dt.date.tolist())
    span = f'{detected_weeks[0]} to {detected_weeks[-1]}' if detected_weeks else 'NONE DETECTED'
    print(f'  [{ep["name"]:12s}] {ep["label"]:58s}  {n} weeks detected within {MAX_LOOKBACK_WEEKS}wk '
          f'ceiling ({win_start.date()} to {ep["first_action"].date()}), spanning {span}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. PER-CROP MODELLING — genuine within-crop leave-one-out where >=2
# episodes exist (onion); in-sample fit only, explicitly labelled, where
# only 1 episode exists (tomato, potato).
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] Per-crop modelling ...')

def fit_lgbm(X_train, y_train):
    n_pos = y_train.sum()
    spw = (len(y_train) - n_pos) / max(n_pos, 1)
    model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, num_leaves=15,
                                learning_rate=0.05, scale_pos_weight=spw,
                                min_child_samples=5, verbose=-1)
    model.fit(X_train, y_train)
    return model

loeo_rows = []
crop_weekly['score'] = np.nan
crop_weekly['score_type'] = ''   # 'held_out' or 'in_sample'
episode_models = {}   # episode name -> (model, crop) -- kept for potential external reuse/inspection;
                       # no longer consumed downstream in this script since the 2026-09-02 fix (Section
                       # 4.5/5 now use a separate full-history out-of-fold score, not these per-episode ones)

for crop in CROPS:
    crop_episodes = [ep for ep in EPISODES if ep['crop'] == crop]
    crop_data = crop_weekly[crop_weekly['crop'] == crop]
    print(f'\n  --- {crop} ({len(crop_episodes)} episode(s)) ---')

    if len(crop_episodes) >= 2:
        # Genuine within-crop leave-one-episode-out.
        for ep in crop_episodes:
            test_mask = crop_data['episode'] == ep['name']
            around = ((crop_data['week_start'] >= ep['first_action'] - pd.Timedelta(weeks=LOOKBACK_WEEKS * 4)) &
                      (crop_data['week_start'] <= ep['first_action'] + pd.Timedelta(weeks=LOOKBACK_WEEKS)))
            # Padding negatives near the event window, added to the test set
            # for a fuller precision/recall picture than the episode's own
            # (mostly escalation-week) rows alone would give.
            #
            # FIXED 2026-09-02 (audit finding, confirmed real): these padding
            # rows satisfy `episode != ep['name']`, which is EXACTLY
            # `train_mask`'s condition below -- every one of them used to be
            # trained AND tested on. Only the episode's own held-out rows
            # were genuinely unseen; the AUC/AP reported here were computed
            # on a test set most of which the model had already fit. Now
            # explicitly excluded from train_mask so every row scored below
            # is one the model never saw.
            extra_test_negatives = around & (crop_data['episode'] != ep['name']) & (crop_data['label'] == 0)
            test_mask_full = test_mask | extra_test_negatives
            train_mask = (crop_data['episode'] != ep['name']) & (~extra_test_negatives)

            X_train = crop_data.loc[train_mask, FEATURES]
            y_train = crop_data.loc[train_mask, 'label']
            X_test = crop_data.loc[test_mask_full, FEATURES]
            y_test = crop_data.loc[test_mask_full, 'label']

            model = fit_lgbm(X_train, y_train)
            episode_models[ep['name']] = (model, crop)
            scores = model.predict_proba(X_test)[:, 1]
            crop_weekly.loc[X_test.index, 'score'] = scores
            crop_weekly.loc[X_test.index, 'score_type'] = 'held_out'

            # Also score the WIDE plotting window with this held-out model
            win_start = ep['first_action'] - pd.Timedelta(weeks=LOOKBACK_WEEKS * 3)
            win_end = ep['first_action'] + pd.Timedelta(weeks=LOOKBACK_WEEKS)
            plot_mask = ((crop_data['week_start'] >= win_start) & (crop_data['week_start'] <= win_end))
            X_plot = crop_data.loc[plot_mask, FEATURES]
            plot_scores = model.predict_proba(X_plot)[:, 1]
            crop_weekly.loc[X_plot.index, 'score'] = plot_scores
            crop_weekly.loc[X_plot.index, 'score_type'] = 'held_out'

            if y_test.sum() > 0 and y_test.sum() < len(y_test):
                auc = roc_auc_score(y_test, scores)
                ap = average_precision_score(y_test, scores)
            else:
                auc, ap = np.nan, np.nan
            loeo_rows.append({'crop': crop, 'episode': ep['name'], 'label': ep['label'],
                               'n_test_weeks': len(y_test), 'n_test_positive': int(y_test.sum()),
                               'auc': round(auc, 3) if pd.notna(auc) else None,
                               'avg_precision': round(ap, 3) if pd.notna(ap) else None,
                               'n_train_episodes': len(crop_episodes) - 1})
            print(f'    [{ep["name"]:12s}] within-crop LOEO: test n={len(y_test):3d} '
                  f'(pos={int(y_test.sum())})  AUC={auc:.3f}  AP={ap:.3f}')
    else:
        # Only 1 episode for this crop: no within-crop held-out test is
        # possible. Fit on all available data (including this episode's own
        # labels) and report as an IN-SAMPLE fit only -- not a validation.
        ep = crop_episodes[0]
        X_all = crop_data[FEATURES]
        y_all = crop_data['label']
        model = fit_lgbm(X_all, y_all)
        episode_models[ep['name']] = (model, crop)
        scores = model.predict_proba(X_all)[:, 1]
        crop_weekly.loc[X_all.index, 'score'] = scores
        crop_weekly.loc[X_all.index, 'score_type'] = 'in_sample'
        loeo_rows.append({'crop': crop, 'episode': ep['name'], 'label': ep['label'],
                           'n_test_weeks': None, 'n_test_positive': None,
                           'auc': None, 'avg_precision': None, 'n_train_episodes': 0})
        print(f'    [{ep["name"]:12s}] only 1 episode for {crop} -- IN-SAMPLE fit only, '
              f'no held-out test possible (reported, not hidden).')

loeo_df = pd.DataFrame(loeo_rows)
loeo_path = os.path.join(OUT_DIR, 'table_escalation_signature_loeo_percrop.csv')
loeo_df.to_csv(loeo_path, index=False)
print(f'\n  Saved: {loeo_path}')

for _crop in CROPS:
    _aucs = loeo_df[(loeo_df['crop'] == _crop) & loeo_df['auc'].notna()]['auc']
    if len(_aucs):
        _std = f'{_aucs.std(ddof=1):.3f}' if len(_aucs) > 1 else 'n/a (1 fold)'
        print(f'\n  {_crop.capitalize()} within-crop LOEO mean AUC: {_aucs.mean():.3f} '
              f'(std {_std}, n={len(_aucs)} folds).')
    else:
        print(f'\n  {_crop.capitalize()}: no AUC reported -- honestly not computable with 1 episode.')

scores_path = os.path.join(OUT_DIR, 'table_escalation_signature_scores_percrop.csv')
crop_weekly[['crop', 'week_start', 'label', 'episode', 'score', 'score_type']].to_csv(scores_path, index=False)
print(f'  Saved: {scores_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 4.5 FULL-HISTORY OUT-OF-FOLD SCORING for the placebo-in-time test (Section 5)
# ─────────────────────────────────────────────────────────────────────────────
# FIXED 2026-09-02 (audit finding, confirmed): Section 5 used to reuse
# Section 4's per-episode LOEO models to score EVERY candidate window across
# a crop's full history. Those models exclude only ONE episode's own rows --
# every background (non-episode) week, which is the vast majority of the
# placebo candidate pool, WAS in that model's training set. The real
# episode's own window was genuinely out-of-fold (that's what the LOEO
# models are for), but nearly every placebo candidate was scored in-sample.
# A confident in-sample score on background weeks makes the genuinely
# out-of-sample real window look artificially more "extreme" by comparison
# -- a systematic bias that deflates every placebo p-value below, in the
# same direction every time (never accidentally the other way), which is
# exactly what a real leakage effect looks like rather than noise.
#
# Fix: K-fold TIME-BLOCK cross-validation across each crop's ENTIRE weekly
# history (not just near episodes), with block boundaries nudged so no
# episode's own labelled window is ever split across two folds -- each
# episode's full window (and its positive labels) sits entirely inside one
# fold, mirroring Section 4's per-episode exclusion but extended to the
# background weeks too. Every week in the timeline, episode or background,
# gets scored by a model that never saw that week (or, for episode weeks,
# that whole episode) during training. This single out-of-fold score series
# feeds BOTH the real window's intensity and every placebo candidate's
# intensity in Section 5, so that comparison is finally apples-to-apples.
print(f'\n[4.5] Building full-history out-of-fold scores for the placebo test '
      f'(K={N_OOF_FOLDS}-fold time-block CV, episode-window-safe) ...')


def build_oof_fold_map(crop_data, episodes_for_crop, k):
    """Returns a Series indexed like crop_data, giving each row's fold
    number. Fold boundaries are placed at roughly equal calendar-time
    spacing, then nudged to the nearest point outside every episode's own
    [win_start, first_action) window so no episode is split across folds."""
    weeks = crop_data['week_start'].sort_values().unique()
    protected = [(ep['first_action'] - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS), ep['first_action'])
                 for ep in episodes_for_crop]

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
    return crop_data['week_start'].map(fold_of_week)


crop_weekly['score_oof'] = np.nan
for crop in CROPS:
    crop_episodes = [ep for ep in EPISODES if ep['crop'] == crop]
    crop_data = crop_weekly[crop_weekly['crop'] == crop].sort_values('week_start')
    fold_of_row = build_oof_fold_map(crop_data, crop_episodes, N_OOF_FOLDS)
    n_folds_actual = fold_of_row.nunique()
    for fold in sorted(fold_of_row.unique()):
        train_idx = crop_data.index[fold_of_row != fold]
        test_idx = crop_data.index[fold_of_row == fold]
        y_tr = crop_weekly.loc[train_idx, 'label']
        if y_tr.nunique() < 2:
            print(f'  WARNING: {crop} OOF fold {fold} has a degenerate training label set '
                  f'({y_tr.nunique()} class(es)) -- skipping, rows left unscored.')
            continue
        oof_model = fit_lgbm(crop_weekly.loc[train_idx, FEATURES], y_tr)
        crop_weekly.loc[test_idx, 'score_oof'] = oof_model.predict_proba(
            crop_weekly.loc[test_idx, FEATURES])[:, 1]
    n_scored = crop_weekly.loc[crop_data.index, 'score_oof'].notna().sum()
    print(f'  {crop}: {n_folds_actual} time-block folds (target K={N_OOF_FOLDS}), '
          f'{n_scored}/{len(crop_data)} rows scored out-of-fold')

# Re-save with score_oof included -- makes the out-of-fold scores that feed
# Section 5 directly auditable from disk, not just recomputable.
crop_weekly[['crop', 'week_start', 'label', 'episode', 'score', 'score_type', 'score_oof']].to_csv(
    scores_path, index=False)
print(f'  Re-saved with score_oof: {scores_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLACEBO-IN-TIME SIGNIFICANCE TEST — directly analogous to the in-space
# placebo permutation test used for the SDID postban ATT (Script 38). Uses
# the SAME data-driven detection rule as the real labels (Section 3): for any
# candidate end-date, look at the MAX_LOOKBACK_WEEKS window before it, find
# weeks where price deviated >=DEV_THRESHOLD from seasonal norm, and take the
# mean out-of-fold SCORE over just those detected weeks (0 if none detected).
# Applying the identical rule to real and placebo candidates keeps the
# comparison apples-to-apples -- a candidate with no detected escalation
# weeks at all gets an intensity of 0, exactly like a real week with a flat
# price would. Uses `score_oof` (Section 4.5) throughout, NOT any single
# episode's LOEO model -- every candidate, real or placebo, is scored by a
# model that never saw its own week during training.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Placebo-in-time significance test (data-driven window, real vs. every other candidate) ...')

def detected_intensity(crop_data, end_date):
    """Mean out-of-fold score over weeks in [end_date - MAX_LOOKBACK_WEEKS,
    end_date) whose price deviates >=DEV_THRESHOLD from seasonal norm;
    0.0 if none detected."""
    win_start = end_date - pd.Timedelta(weeks=MAX_LOOKBACK_WEEKS)
    mask = ((crop_data['week_start'] >= win_start) & (crop_data['week_start'] < end_date) &
            (crop_data['price_vs_seasonal_norm'] >= DEV_THRESHOLD))
    if mask.sum() == 0:
        return 0.0
    return float(crop_data.loc[mask, 'score_oof'].mean())

EPISODE_COUNT_BY_CROP = {c: len([e for e in EPISODES if e['crop'] == c]) for c in CROPS}

placebo_rows = []
placebo_detail = {}   # episode name -> array of placebo candidate intensities
for ep in EPISODES:
    crop = ep['crop']
    crop_data = crop_weekly[crop_weekly['crop'] == crop].sort_values('week_start').reset_index(drop=True)

    real_intensity = detected_intensity(crop_data, ep['first_action'])

    # Placebo pool: non-overlapping MAX_LOOKBACK_WEEKS candidate end-dates
    # across the crop's full history, excluding any candidate within
    # 2x MAX_LOOKBACK_WEEKS of ANY real episode for this crop.
    crop_ep_dates = [e['first_action'] for e in EPISODES if e['crop'] == crop]
    weeks = crop_data['week_start'].tolist()
    placebo_intensities = []
    i = MAX_LOOKBACK_WEEKS
    while i < len(weeks):
        cand_end = weeks[i]
        too_close = any(abs((cand_end - d).days) < MAX_LOOKBACK_WEEKS * 2 * 7 for d in crop_ep_dates)
        if not too_close:
            placebo_intensities.append(detected_intensity(crop_data, cand_end))
        i += MAX_LOOKBACK_WEEKS   # non-overlapping windows

    placebo_intensities = np.array(placebo_intensities)
    n_placebo = len(placebo_intensities)
    n_as_extreme = int((placebo_intensities >= real_intensity).sum())
    p_value = n_as_extreme / n_placebo if n_placebo else np.nan
    placebo_detail[ep['name']] = placebo_intensities

    placebo_rows.append({'crop': crop, 'episode': ep['name'], 'label': ep['label'],
                          'real_window_intensity': round(float(real_intensity), 5),
                          'n_placebo_windows': n_placebo,
                          'n_placebo_as_extreme': n_as_extreme,
                          'p_value': round(float(p_value), 3) if pd.notna(p_value) else None,
                          'score_type': 'held_out' if EPISODE_COUNT_BY_CROP[crop] >= 2 else 'in_sample'})
    print(f'  [{ep["name"]:12s}] real intensity={real_intensity:.5f}  vs {n_placebo} placebo candidates '
          f'({crop}\'s full history)  -- {n_as_extreme} as extreme  p={p_value:.3f}')

placebo_df = pd.DataFrame(placebo_rows)
placebo_path = os.path.join(OUT_DIR, 'table_escalation_signature_placebo_test.csv')
placebo_df.to_csv(placebo_path, index=False)
print(f'\n  Saved: {placebo_path}')
in_sample_crops = [c for c in CROPS if EPISODE_COUNT_BY_CROP[c] < 2]
held_out_crops = [c for c in CROPS if EPISODE_COUNT_BY_CROP[c] >= 2]
if in_sample_crops:
    print(f'  NOTE: {", ".join(in_sample_crops)} p-value(s) use an IN-SAMPLE model (the real window\'s')
    print('  own labels were part of its training data), so those p-values are optimistic/biased low')
    print('  and should be read as descriptive, not a genuine significance test.')
print(f'  Genuine held-out p-values (model never saw that episode\'s labels): {", ".join(held_out_crops)}.')

# Placebo distribution figure -- mirrors fig_sdid_inspace_placebo.png's design.
# FIXED 2026-08-19: was a hardcoded 2x3 grid (6 slots) left over from when
# there were exactly 6 episodes -- with 8 episodes now, zip() silently
# truncated to the first 6 in EPISODES order and dropped potato entirely
# from this figure (the CSV was always complete; only this plot was short).
# Sized dynamically off len(EPISODES) so this can't silently drop a crop
# again if more episodes are added later.
n_ep = len(EPISODES)
n_cols = 3
n_rows = -(-n_ep // n_cols)   # ceil division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
axes_flat = axes.flat
for ax, ep in zip(axes_flat, EPISODES):
    intens = placebo_detail[ep['name']]
    real_val = placebo_df.loc[placebo_df['episode'] == ep['name'], 'real_window_intensity'].iloc[0]
    pval = placebo_df.loc[placebo_df['episode'] == ep['name'], 'p_value'].iloc[0]
    ax.hist(intens, bins=25, color='#888888', alpha=0.75, edgecolor='white')
    ax.axvline(real_val, color=CROP_COLORS[ep['crop']], linewidth=2.2,
               label=f'Real episode (p={pval:.3f})')
    ax.set_title(ep['label'], fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Mean score over detected escalation weeks (0 if none)')
    ax.set_ylabel('Count of placebo windows')
    ax.legend(fontsize=7, frameon=False)
for ax in list(axes_flat)[n_ep:]:
    ax.axis('off')
plt.tight_layout()
placebo_fig_path = os.path.join(OUT_DIR, 'fig_escalation_signature_placebo_test.png')
plt.savefig(placebo_fig_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {placebo_fig_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURES — split into (a) onion's 3 genuine held-out folds, (b) tomato/
# potato's single in-sample fits, kept visually SEPARATE so the two are never
# confused for the same kind of evidence.
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6] Generating figures ...')

def plot_episode(ax, ep, score_type_label):
    win_start = ep['first_action'] - pd.Timedelta(weeks=LOOKBACK_WEEKS * 3)
    win_end = ep['first_action'] + pd.Timedelta(weeks=LOOKBACK_WEEKS)
    sub = crop_weekly[(crop_weekly['crop'] == ep['crop']) &
                       (crop_weekly['week_start'] >= win_start) &
                       (crop_weekly['week_start'] <= win_end)].sort_values('week_start')
    y = sub['score'].clip(lower=1e-6)
    ax.plot(sub['week_start'], y, color=CROP_COLORS[ep['crop']], linewidth=1.8)
    # Data-driven detected weeks are not necessarily contiguous (e.g. onion
    # 2020's mid-window dip) -- mark each one individually rather than a
    # single fixed span, so the plot shows the actual (possibly gapped)
    # shape the label now captures.
    detected = sub[sub['episode'] == ep['name']]
    for d in detected['week_start']:
        ax.axvspan(d - pd.Timedelta(days=3), d + pd.Timedelta(days=3),
                   color='#E8A33D', alpha=0.25, zorder=0)
    ax.axvspan(ep['first_action'], ep['first_action'], color='#E8A33D', alpha=0.25,
               label='Detected escalation week')  # zero-width: legend entry only
    ax.axvline(ep['first_action'], color='#333333', linewidth=1.1, linestyle='--', label='First policy action')
    ax.set_title(f"{ep['label']}\n({score_type_label})", fontsize=9, fontweight='bold')
    ax.set_ylabel('Score (log scale)')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(axis='y', alpha=0.25, which='both')
    ax.legend(fontsize=7, loc='lower left', frameon=False)

held_out_crops = [c for c in CROPS if EPISODE_COUNT_BY_CROP[c] >= 2]
single_crops = [c for c in CROPS if EPISODE_COUNT_BY_CROP[c] < 2]

if held_out_crops:
    n_cols = max(EPISODE_COUNT_BY_CROP[c] for c in held_out_crops)
    n_rows = len(held_out_crops)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    for row, crop in enumerate(held_out_crops):
        crop_eps = [ep for ep in EPISODES if ep['crop'] == crop]
        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(crop_eps):
                fold_label = f'within-crop held-out, {len(crop_eps)} folds'
                plot_episode(ax, crop_eps[col], fold_label)
            else:
                ax.axis('off')   # crop has fewer episodes than the widest row
    fig.suptitle('Within-crop leave-one-episode-out (genuine held-out test) -- '
                  + ', '.join(f'{c} x{EPISODE_COUNT_BY_CROP[c]}' for c in held_out_crops),
                  fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig_path1 = os.path.join(OUT_DIR, 'fig_escalation_signature_percrop_heldout.png')
    plt.savefig(fig_path1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fig_path1}')
else:
    print('  No crop has >=2 episodes -- skipping held-out figure.')

single_eps = [ep for ep in EPISODES if ep['crop'] in single_crops]
if single_eps:
    fig, axes = plt.subplots(1, len(single_eps), figsize=(5 * len(single_eps), 4.5), squeeze=False)
    for ax, ep in zip(axes[0], single_eps):
        plot_episode(ax, ep, 'IN-SAMPLE fit -- not held-out, only 1 episode available')
    fig.suptitle(', '.join(c for c in single_crops) + ': single episode each -- '
                  'in-sample fit only, NOT a validated test',
                  fontsize=11, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig_path2 = os.path.join(OUT_DIR, 'fig_escalation_signature_percrop_single.png')
    plt.savefig(fig_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fig_path2}')
else:
    print('  Every crop now has >=2 episodes -- no in-sample-only figure needed.')

print('\n' + '=' * 65)
print('Script 40 v5 complete.')
print('\nHonest summary: ' + ', '.join(f'{c} x{EPISODE_COUNT_BY_CROP[c]}' for c in CROPS) + '.')
print(f'Held-out (genuine same-crop leave-one-out): {", ".join(held_out_crops) or "none"}.')
print(f'In-sample only (not a validated test): {", ".join(in_sample_crops) or "none"}.')
print('The placebo-in-time test adds a second, independent layer on top: for each held-out')
print('crop, does the real pre-intervention window actually stand out against EVERY OTHER')
print('window in that crop\'s 23-year history, not just against a small hand-picked test')
print('set -- see the p-values and table_escalation_signature_placebo_test.csv.')
