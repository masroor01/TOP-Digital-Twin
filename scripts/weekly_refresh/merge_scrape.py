# -*- coding: utf-8 -*-
"""
Weekly Refresh -- Step 4: Merge a Validated Scrape Into the Trusted Raw File
===============================================================================
Generalized version of the one-off 09d/09f rebuild scripts, parameterized
instead of hardcoded to a specific week's filenames. ONLY run this after
validate_scrape.py has passed for the same (crop, new_file) pair -- this
script does not re-validate, it trusts its caller.

Strategy: keep everything in the trusted file strictly before the new
scrape's own minimum date untouched, and REPLACE everything from that date
onward with the new scrape's rows. Handles null market_id backfill the same
way every prior refresh in this project has (match by (state, market)
against the trusted file's existing lookup; assign new sequential IDs only
for genuinely new markets).

FIXED 2026-09-02 (audit finding, confirmed live production risk): this
replace-the-suffix strategy silently deletes real trusted data if the new
scrape happens to be thinner than what it's replacing -- validate_scrape.py
now checks for this before merge is ever called (see its 2026-09-02 fix),
but this script had no safety net of its own if it were ever invoked
directly, out of order, or against a file validate_scrape.py didn't see.
Added a second, independent guard right before the overwrite: if the new
rows are less than half the size of the trusted rows they're about to
replace, abort rather than merge (pass --force to override for a
deliberate narrow rebuild, e.g. a cleaner re-pull with fewer duplicate
rows by design, matching the 09c/09d precedent).

Usage:
    python merge_scrape.py <crop> <trusted_csv> <new_scrape_csv> [--force]

Overwrites <trusted_csv> in place after writing a timestamped backup
alongside it (<trusted_csv>.backup_YYYYMMDD_HHMMSS).
"""
import os
import sys
import shutil
from datetime import datetime
import pandas as pd

OUT_COLS = ['source', 'commodity', 'commodity_id', 'state', 'state_id', 'state_code',
            'district', 'district_id', 'market', 'market_id', 'arrival_date',
            'arrival_date_raw', 'year', 'month', 'variety', 'arrivals_tonnes',
            'total_arrivals_tonnes', 'min_price_rs_per_quintal',
            'max_price_rs_per_quintal', 'modal_price_rs_per_quintal']

MAX_BACKUPS = 6  # keep the last N weekly backups per crop, prune older ones


def prune_backups(trusted_path):
    d = os.path.dirname(trusted_path)
    base = os.path.basename(trusted_path)
    backups = sorted(
        f for f in os.listdir(d) if f.startswith(base + '.backup_')
    )
    while len(backups) > MAX_BACKUPS:
        oldest = backups.pop(0)
        os.remove(os.path.join(d, oldest))
        print(f'  Pruned old backup: {oldest}')


def main():
    args = [a for a in sys.argv[1:] if a != '--force']
    force = len(args) != len(sys.argv[1:])
    if len(args) != 3:
        print(__doc__)
        sys.exit(2)
    crop, trusted_path, new_path = args
    crop = crop.lower()

    print(f'=== Merging {crop} ===')
    old = pd.read_csv(trusted_path, low_memory=False)
    new = pd.read_csv(new_path, low_memory=False)

    assert (new['commodity'].str.lower() == crop).all(), \
        f'non-{crop} rows found in {new_path} -- run validate_scrape.py first'

    cutoff = pd.to_datetime(new['arrival_date']).min()
    old_dates = pd.to_datetime(old['arrival_date'])
    old_kept = old[old_dates < cutoff].copy()
    print(f'  Trusted file: {len(old):,} rows -> {len(old_kept):,} rows before '
          f'{cutoff.date()} (kept)')
    max_market_id = int(old['market_id'].max())
    max_district_id = int(old['district_id'].max())
    print(f'  New scrape: {len(new):,} rows, {new["arrival_date"].min()} to '
          f'{new["arrival_date"].max()}')

    null_mask = new['market_id'].isna()
    null_markets = new.loc[null_mask, ['state', 'district', 'market']].drop_duplicates()
    print(f'  {null_mask.sum():,} rows with null market_id, {len(null_markets)} distinct markets')

    old_lookup = old.drop_duplicates(['state', 'market'])[
        ['state', 'market', 'market_id', 'district', 'district_id']
    ]
    resolved = null_markets.merge(old_lookup, on=['state', 'market'], how='left', suffixes=('', '_old'))
    matched = resolved[resolved['market_id'].notna()]
    unmatched = resolved[resolved['market_id'].isna()][['state', 'district', 'market']].reset_index(drop=True)
    print(f'  {len(matched)} matched to an existing market_id, {len(unmatched)} genuinely '
          f'new -> assigning new IDs')

    unmatched['market_id_new'] = max_market_id + 1 + unmatched.index
    unmatched['district_id_new'] = max_district_id + 1 + unmatched.index

    new = new.merge(
        matched[['state', 'market', 'market_id', 'district', 'district_id']]
        .rename(columns={'market_id': 'market_id_fill', 'district': 'district_fill',
                          'district_id': 'district_id_fill'}),
        on=['state', 'market'], how='left')
    new = new.merge(unmatched[['state', 'market', 'market_id_new', 'district_id_new']],
                     on=['state', 'market'], how='left')
    new['market_id'] = new['market_id'].fillna(new['market_id_fill']).fillna(new['market_id_new'])
    new['district'] = new['district'].fillna(new['district_fill'])
    new['district_id'] = new['district_id'].fillna(new['district_id_fill']).fillna(new['district_id_new'])
    new = new.drop(columns=['market_id_fill', 'district_fill', 'district_id_fill',
                             'market_id_new', 'district_id_new'])

    assert new['market_id'].notna().all(), 'unresolved null market_id after backfill'
    new['market_id'] = new['market_id'].astype(int)
    new['district_id'] = new['district_id'].astype(int)
    new['district'] = new['district'].fillna(new['market'])

    new_rows = new[OUT_COLS]
    assert new_rows['modal_price_rs_per_quintal'].notna().all()
    assert new_rows['arrivals_tonnes'].notna().all()

    conflict = new_rows.groupby('market_id')['market'].nunique()
    bad = conflict[conflict > 1]
    if not bad.empty:
        print(f'  NOTE: {len(bad)} market_id(s) map to >1 market name (spelling-variant '
              f'match, as seen before) -- not treated as an error.')

    # Safety net independent of validate_scrape.py: the rows we're ABOUT TO
    # DELETE (everything in the trusted file from cutoff onward) vs. the rows
    # replacing them. A healthy weekly refresh's new_rows is comparable to or
    # larger than what it replaces (same or better coverage, moving forward
    # in time); a truncated/stalled scrape that still has an old-enough min
    # date to trigger a big cutoff would show up here as a large loss.
    replaced = old[old_dates >= cutoff]
    if len(replaced) > 500 and len(new_rows) < 0.5 * len(replaced):
        print(f'  [ABORT] New scrape has {len(new_rows):,} rows but would replace '
              f'{len(replaced):,} existing trusted rows from {cutoff.date()} onward '
              f'(less than 50%) -- this looks like a truncated scrape about to '
              f'delete real trusted data. Refusing to merge.', file=sys.stderr)
        print(f'  If this is a genuine, deliberate narrow rebuild (e.g. a cleaner '
              f're-pull with intentionally fewer duplicate rows), re-run with --force.',
              file=sys.stderr)
        if not force:
            sys.exit(1)
        print(f'  --force given, proceeding despite the row-count drop.')

    combined = pd.concat([old_kept, new_rows], ignore_index=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'{trusted_path}.backup_{ts}'
    shutil.copy(trusted_path, backup_path)
    print(f'  Backed up pre-merge file to: {backup_path}')
    prune_backups(trusted_path)

    combined.to_csv(trusted_path, index=False)
    print(f'  Saved: {trusted_path}')
    print(f'  Total rows: {len(combined):,}  (kept: {len(old_kept):,} + new: {len(new_rows):,})')
    print(f'  Total unique markets: {combined["market_id"].nunique()}')
    print(f'  Date range: {combined["arrival_date"].min()} to {combined["arrival_date"].max()}')


if __name__ == '__main__':
    main()
