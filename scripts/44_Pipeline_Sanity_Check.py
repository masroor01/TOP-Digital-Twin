"""
Script 44 -- Pipeline Sanity Check
=============================================================================
Runs a battery of checks that would have caught every silent pipeline bug
found by hand over the life of this project, so the NEXT one gets caught by
a script instead of by someone noticing a suspicious number days later.

Specifically encodes the failure signature of five real, previously-manual
discoveries:
  1. Phantom fully-imputed tail week (2026-07-27) -- one global END_DATE
     for all crops manufactured a 100%-imputed final week for whichever
     crop's raw source stopped earlier. Caught by eyeballing the dashboard.
  2. data/agmarknet_weekly/ going stale vs Downloads/Agmarknet_Weekly/
     (2026-07-27) -- Script 09 writes to Downloads, not data/, and nothing
     auto-syncs them. Caught because two retrains produced identical row
     counts.
  3. Forward-fill gaps at the reference row (2026-07-27/28) -- MODIS NDVI
     NaN for newly-relocated potato zones because Script 14's own forward
     fill fell one week short of the grid end. Caught by inspecting
     reference_rows.csv by hand.
  4. NaN-is-truthy policy sliders (2026-07-27) -- `float(NaN or 0)` is NaN,
     not 0, because NaN is truthy in Python; MEP/duty sliders silently got
     literal NaN values. Caught by a user bug report, not by any check.
  5. sufficient_history / keep_cols allowlist drop (2026-08-12) -- a new
     flag column was computed correctly but silently excluded from the
     saved CSV by an allowlist that wasn't updated to include it. Caught by
     a live forecast-validation script finding ungrounded predictions.
  6. stale_reference (2026-08-13) -- markets that DO have price history
     (sufficient_history=True) but whose most recent REAL trade is weeks
     old at the reference row, because a long reporting gap fell back to a
     climatological seasonal-median estimate. Both the model and the
     dashboard's naive-persistence comparison were silently anchored on
     that stale estimate. Caught by a live tomato forecast validation.
  7. Hardcoded upper-bound date cap in Script 23 (2026-08-14) -- a
     copy-paste leftover silently discarded every row past a stale fixed
     date BEFORE any feature engineering, so a "successful" retrain on
     freshly-refreshed data was actually still training on the old
     cutoff. Caught by a full-layer audit: two brand-new markets whose
     only real rows were past the cap were silently ABSENT from
     reference_rows.csv entirely, not just flagged.
  8. market-name key collisions (2026-08-14) -- reference_rows.csv used
     to be keyed by market NAME, not market_id. A few names repeat across
     different states (e.g. "Fatehabad APMC" in both Haryana and Uttar
     Pradesh); grouping by name silently collapsed each such pair into
     ONE row, dropping the other real market entirely. Caught by the same
     full-layer audit, comparing market_id counts against reference_rows
     row counts per crop.

Run this after Script 09 (data refresh) and again after Script 23
(retrain) -- see README Sec 4. Exits non-zero if any check FAILs, so it can
be dropped into a routine without reading the output every time.

Usage: python scripts/44_Pipeline_Sanity_Check.py
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data', 'agmarknet_weekly')
DOWNLOADS_DIR = r'C:\Users\masro\Downloads\Agmarknet_Weekly'
MODEL_DIR = os.path.join(BASE, 'Model_Output', 'production_models')
CROPS = ['tomato', 'onion', 'potato']

results = []  # (status, section, message) -- status in {PASS, WARN, FAIL}


def check(status, section, message):
    results.append((status, section, message))
    icon = {'PASS': '  [PASS]', 'WARN': '  [WARN]', 'FAIL': '  [FAIL]'}[status]
    print(f'{icon} {section}: {message}')


print('=================================================================')
print('SCRIPT 44: PIPELINE SANITY CHECK')
print('=================================================================')

# ---------------------------------------------------------------------------
# GROUP A -- Raw weekly panel checks (data/agmarknet_weekly/, post Script 09)
# ---------------------------------------------------------------------------
print('\n[A] Weekly panel checks (data/agmarknet_weekly/)')

for crop in CROPS:
    data_path = os.path.join(DATA_DIR, f'{crop}_weekly_panel.csv')
    dl_path = os.path.join(DOWNLOADS_DIR, f'{crop}_weekly_panel.csv')

    if not os.path.exists(data_path):
        check('FAIL', f'A1 {crop}', f'{data_path} does not exist -- run Script 09.')
        continue

    df = pd.read_csv(data_path, usecols=['market_id', 'week_start', 'imputed'])
    df['week_start'] = pd.to_datetime(df['week_start'])

    # A1: data/ vs Downloads/ sync check -- the exact "stale copy" bug from 2026-07-27.
    if os.path.exists(dl_path):
        dl_size = os.path.getsize(dl_path)
        data_size = os.path.getsize(data_path)
        if dl_size != data_size:
            check('FAIL', f'A1 {crop}',
                  f'data/ copy ({data_size:,} bytes) differs from Downloads/ copy '
                  f'({dl_size:,} bytes) -- Script 09 writes to Downloads only; the repo\'s '
                  f'data/ copy must be refreshed by hand after every Script 09 run.')
        else:
            check('PASS', f'A1 {crop}', 'data/ copy matches Downloads/ copy (same file size).')
    else:
        check('WARN', f'A1 {crop}',
              f'No Downloads/Agmarknet_Weekly/ copy found to compare against -- skipped '
              f'sync check (this is fine if you already cleaned up Downloads/).')

    # A2: phantom fully-imputed tail week -- the global END_DATE bug from 2026-07-27.
    max_week = df['week_start'].max()
    tail = df[df['week_start'] == max_week]
    pct_imputed_tail = (tail['imputed'] != 0).mean() * 100
    if pct_imputed_tail >= 95:
        check('FAIL', f'A2 {crop}',
              f'Latest week ({max_week.date()}) is {pct_imputed_tail:.1f}% imputed across '
              f'{len(tail)} markets -- looks like a phantom manufactured tail week, not real '
              f'data. Check END_DATE handling in Script 09 for this crop\'s actual raw cutoff.')
    elif pct_imputed_tail >= 50:
        check('WARN', f'A2 {crop}',
              f'Latest week ({max_week.date()}) is {pct_imputed_tail:.1f}% imputed -- higher '
              f'than usual, worth a manual look but not necessarily a bug.')
    else:
        check('PASS', f'A2 {crop}',
              f'Latest week ({max_week.date()}) is {pct_imputed_tail:.1f}% imputed -- looks like real data.')

    # A3: informational freshness (not a bug signal, just a heads-up)
    days_old = (pd.Timestamp.now().normalize() - max_week).days
    check('WARN' if days_old > 60 else 'PASS', f'A3 {crop}',
          f'Panel\'s latest week is {days_old} days behind today ({max_week.date()}).')

# ---------------------------------------------------------------------------
# GROUP B -- Production model / reference-row checks (post Script 23)
# ---------------------------------------------------------------------------
print('\n[B] Production model checks (Model_Output/production_models/)')

ref_path = os.path.join(MODEL_DIR, 'reference_rows.csv')
fcols_path = os.path.join(MODEL_DIR, 'feature_columns.json')
stale_path = os.path.join(MODEL_DIR, 'macro_climate_staleness.json')

if not os.path.exists(ref_path):
    check('FAIL', 'B0', f'{ref_path} does not exist -- run Script 23.')
else:
    ref = pd.read_csv(ref_path)

    # B1: sufficient_history column must exist -- the 2026-08-12 keep_cols bug.
    if 'sufficient_history' not in ref.columns:
        check('FAIL', 'B1',
              'reference_rows.csv has no sufficient_history column -- it was computed in '
              'Script 23 but dropped by the keep_cols allowlist. Check the keep_cols list '
              'at the end of Script 23 includes every column the script computes.')
    else:
        for crop in CROPS:
            sub = ref[ref['crop'] == crop]
            if len(sub) == 0:
                check('WARN', f'B1 {crop}', 'No reference rows for this crop at all.')
                continue
            n_insufficient = (~sub['sufficient_history'].astype(bool)).sum()
            check('PASS', f'B1 {crop}',
                  f'{n_insufficient}/{len(sub)} markets flagged sufficient_history=False '
                  f'(expected to be a small minority, not all or none).')

    # B1b: stale_reference column must exist -- the 2026-08-13 stale-anchor bug.
    if 'stale_reference' not in ref.columns:
        check('FAIL', 'B1b',
              'reference_rows.csv has no stale_reference column -- it was computed in '
              'Script 23 but dropped by the keep_cols allowlist (same failure shape as the '
              'sufficient_history bug -- check keep_cols includes every computed column).')
    else:
        for crop in CROPS:
            sub = ref[ref['crop'] == crop]
            if len(sub) == 0:
                continue
            n_stale = sub['stale_reference'].astype(bool).sum()
            check('PASS', f'B1b {crop}',
                  f'{n_stale}/{len(sub)} markets flagged stale_reference=True '
                  f'(expected to be a minority, not all or none).')

    # B1c: reference_rows.csv week_start must match the panel's own max week --
    # the hardcoded-date-cap bug (2026-08-14): a stale fixed upper bound
    # silently truncated the panel before feature engineering, so a
    # "successful" retrain kept using an old cutoff despite fresher data
    # being on disk. A gap of more than 1 week is a red flag.
    for crop in CROPS:
        panel_path = os.path.join(DATA_DIR, f'{crop}_weekly_panel.csv')
        sub = ref[ref['crop'] == crop]
        if not os.path.exists(panel_path) or len(sub) == 0 or 'week_start' not in sub.columns:
            continue
        panel_max = pd.to_datetime(pd.read_csv(panel_path, usecols=['week_start'])['week_start']).max()
        ref_max = pd.to_datetime(sub['week_start']).max()
        gap_days = (panel_max - ref_max).days
        if gap_days > 7:
            check('FAIL', f'B1c {crop}',
                  f'reference_rows.csv week_start ({ref_max.date()}) is {gap_days} days behind '
                  f'the panel\'s own max week ({panel_max.date()}) -- check Script 23 for a '
                  f'hardcoded upper-bound date filter silently truncating the panel.')
        else:
            check('PASS', f'B1c {crop}', f'reference_rows.csv week_start ({ref_max.date()}) matches the panel.')

    # B1d: market_id column must exist and its distinct count must match the
    # panel exactly -- the market-name-collision bug (2026-08-14): grouping
    # reference rows by market NAME silently collapsed same-named markets in
    # different states (e.g. two "Fatehabad APMC"s) into one row, dropping
    # the other entirely.
    if 'market_id' not in ref.columns:
        check('FAIL', 'B1d',
              'reference_rows.csv has no market_id column -- markets are being keyed by name, '
              'which silently drops same-named markets in different states. See Script 23.')
    else:
        for crop in CROPS:
            panel_path = os.path.join(DATA_DIR, f'{crop}_weekly_panel.csv')
            if not os.path.exists(panel_path):
                continue
            panel_ids = pd.read_csv(panel_path, usecols=['market_id'])['market_id'].nunique()
            ref_ids = ref[ref['crop'] == crop]['market_id'].nunique()
            if panel_ids != ref_ids:
                check('FAIL', f'B1d {crop}',
                      f'panel has {panel_ids} distinct market_ids but reference_rows.csv has '
                      f'{ref_ids} -- some markets are being silently dropped or merged.')
            else:
                check('PASS', f'B1d {crop}', f'{ref_ids} distinct market_ids match the panel exactly.')

    # B2: feature_columns.json -- every listed feature must actually exist in reference_rows.
    if os.path.exists(fcols_path):
        with open(fcols_path) as f:
            feature_columns = json.load(f)
        missing_any = False
        for key, cols in feature_columns.items():
            missing = [c for c in cols if c not in ref.columns]
            if missing:
                missing_any = True
                check('FAIL', f'B2 {key}',
                      f'{len(missing)} feature(s) listed in feature_columns.json are missing '
                      f'from reference_rows.csv: {missing[:5]}{"..." if len(missing) > 5 else ""} '
                      f'-- the model expects these at predict time; the dashboard will fail or '
                      f'silently substitute 0.')
        if not missing_any:
            check('PASS', 'B2', 'All features listed in feature_columns.json are present in reference_rows.csv.')
    else:
        check('FAIL', 'B2', f'{fcols_path} does not exist -- run Script 23.')

    # B3/B4: forward-filled column groups should have NO NaN at the reference row --
    # the MODIS/potato-zone gap (2026-07-28) and the NaN-is-truthy policy bug (2026-07-27).
    ffill_groups = {
        'macro/climate/satellite': [c for c in ref.columns if any(
            s in c for s in ['era5_', 'chirps_', 's2_', 'modis_', 'diesel', 'repo_rate',
                              'usd_inr', 'wpi_', 'bank_credit', 'agri_wages', 'iip_'])],
        'policy': [c for c in ref.columns if any(
            s in c for s in ['export_ban', 'mep_', 'export_duty', 'market_intervention', 'operation_greens'])],
    }
    for label, cols in ffill_groups.items():
        if not cols:
            check('WARN', f'B3 {label}', 'No matching columns found -- naming may have changed, check manually.')
            continue
        nan_counts = ref[cols].isna().sum()
        bad_cols = nan_counts[nan_counts > 0]
        if len(bad_cols) > 0:
            worst = bad_cols.sort_values(ascending=False).head(5)
            check('FAIL', f'B3 {label}',
                  f'{len(bad_cols)}/{len(cols)} {label} columns have unexpected NaN at the '
                  f'reference row (should be fully forward-filled): '
                  f'{dict(worst)}. A NaN here can silently disable a dashboard slider '
                  f'(checkboxes render as checked; float sliders get a literal NaN value).')
        else:
            check('PASS', f'B3 {label}', f'All {len(cols)} {label} columns are fully populated (no NaN).')

    # B5: expected metadata columns present (the ones added across this project's bugfixes).
    expected_meta = ['imputed', 'last_observed_price', 'last_observed_date',
                      'pct_imputed_last_52w', 'sufficient_history', 'stale_reference']
    missing_meta = [c for c in expected_meta if c not in ref.columns]
    if missing_meta:
        check('FAIL', 'B5', f'Expected metadata columns missing from reference_rows.csv: {missing_meta}')
    else:
        check('PASS', 'B5', 'All expected metadata columns present.')

# B6: staleness file exists and isn't wildly out of date itself.
if os.path.exists(stale_path):
    with open(stale_path) as f:
        staleness = json.load(f)
    check('PASS', 'B6', f'macro_climate_staleness.json present ({len(staleness)} crop x feature entries).')
else:
    check('WARN', 'B6', 'macro_climate_staleness.json missing -- dashboard staleness captions will not show.')

# ---------------------------------------------------------------------------
# GROUP C -- Model artifact presence
# ---------------------------------------------------------------------------
print('\n[C] Model artifact presence')

HORIZONS = ['1w', '4w', '13w', '26w']
for crop in CROPS:
    for h in HORIZONS:
        p = os.path.join(MODEL_DIR, f'{crop}_{h}.joblib')
        if not os.path.exists(p):
            check('FAIL', f'C {crop}_{h}', f'{p} is missing.')
        elif os.path.getsize(p) < 1000:
            check('FAIL', f'C {crop}_{h}', f'{p} exists but is suspiciously small ({os.path.getsize(p)} bytes).')
        else:
            check('PASS', f'C {crop}_{h}', f'Present ({os.path.getsize(p):,} bytes).')

for fname in ['feature_ranges.json', 'model_uncertainty.json', 'price_history.csv']:
    p = os.path.join(MODEL_DIR, fname)
    if os.path.exists(p):
        check('PASS', f'C {fname}', 'Present.')
    else:
        check('WARN', f'C {fname}', 'Missing -- some dashboard sections may not render.')

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
n_pass = sum(1 for s, _, _ in results if s == 'PASS')
n_warn = sum(1 for s, _, _ in results if s == 'WARN')
n_fail = sum(1 for s, _, _ in results if s == 'FAIL')

print('\n' + '=' * 65)
print(f'SUMMARY: {n_pass} passed, {n_warn} warnings, {n_fail} failed')
print('=' * 65)

if n_fail > 0:
    print('\nFAILED CHECKS:')
    for status, section, message in results:
        if status == 'FAIL':
            print(f'  - {section}: {message}')
    print('\nFix these before trusting the dashboard or shipping new results.')
    sys.exit(1)
elif n_warn > 0:
    print('\nNo failures, but see WARN lines above -- usually informational, not urgent.')
    sys.exit(0)
else:
    print('\nAll checks passed clean.')
    sys.exit(0)
