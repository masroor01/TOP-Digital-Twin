# -*- coding: utf-8 -*-
"""
Script 09c — Rebuild Onion 2026 From the Native Scraper Source
====================================================================
Supersedes 09b's portal-report merge for onion. A cleaner source turned
up after 09b: `test_onion_2026.csv`, from the SAME scraper as tomato/
potato (source="AGMARKNET 2.0", same 20-column schema), covering
2025-01-01 through 2026-07-25 -- unlike the original onion scraper file
which stopped at 2025-12-31.

Verified before trusting it:
  - Overlapping 2025 dates (e.g. Kurnool APMC, Jun 2025) match the old
    file's prices/arrivals EXACTLY.
  - Its 2025 coverage is far more complete than the old file's own 2025
    tail (259,491 rows / 1,170 markets vs 101,153 rows / 565 markets) --
    the old file's 2025 data was itself thin, not just missing 2026.
  - Only 1,646 rows (14 markets) have a null market_id; 5 of those 14
    match an existing market_id in the old file by (state, market) name,
    the other 9 are genuinely new markets and get freshly assigned IDs
    (continuing from the OLD file's max market_id/district_id, not the
    now-discarded IDs 09b invented for its portal-report merge).

Inputs:
  Downloads/onion_all_india_apmcs_2000_2025.csv   (original, through 2025-12-31)
  Downloads/test_onion_2026 (1).csv               (new, 2025-01-01 to 2026-07-25)

Output:
  Downloads/onion_all_india_apmcs_2000_2026.csv
    (old pre-2025 rows + all of the new file, replacing 09b's output)
"""

import os
import pandas as pd
import numpy as np

DOWNLOADS = os.environ.get('TOP_DOWNLOADS_DIR', r'C:\Users\masro\Downloads')
OLD_FILE = os.path.join(DOWNLOADS, 'onion_all_india_apmcs_2000_2025.csv')
NEW_FILE = os.path.join(DOWNLOADS, 'test_onion_2026 (1).csv')
OUT_FILE = os.path.join(DOWNLOADS, 'onion_all_india_apmcs_2000_2026.csv')

OUT_COLS = ['source', 'commodity', 'commodity_id', 'state', 'state_id', 'state_code',
            'district', 'district_id', 'market', 'market_id', 'arrival_date',
            'arrival_date_raw', 'year', 'month', 'variety', 'arrivals_tonnes',
            'total_arrivals_tonnes', 'min_price_rs_per_quintal',
            'max_price_rs_per_quintal', 'modal_price_rs_per_quintal']


def main():
    print('Loading old onion 2000-2025 file ...')
    old = pd.read_csv(OLD_FILE, low_memory=False)
    old_pre_2025 = old[old['arrival_date'] < '2025-01-01'].copy()
    print(f'  {len(old):,} total rows -> {len(old_pre_2025):,} rows before 2025-01-01 (kept)')
    max_market_id = int(old['market_id'].max())
    max_district_id = int(old['district_id'].max())

    print('Loading new scraper file (test_onion_2026) ...')
    new = pd.read_csv(NEW_FILE, low_memory=False)
    print(f'  {len(new):,} rows, {new["arrival_date"].min()} to {new["arrival_date"].max()}')
    assert (new['commodity'] == 'Onion').all(), 'non-onion rows found in the new file'

    # ---- Backfill the 1,646 null-market_id rows ----
    null_mask = new['market_id'].isna()
    null_markets = new.loc[null_mask, ['state', 'district', 'market']].drop_duplicates()
    print(f'  {null_mask.sum():,} rows with null market_id, {len(null_markets)} distinct markets')

    old_lookup = old.drop_duplicates(['state', 'market'])[
        ['state', 'market', 'market_id', 'district', 'district_id', 'state_id', 'state_code']
    ]
    resolved = null_markets.merge(old_lookup, on=['state', 'market'], how='left', suffixes=('', '_old'))
    matched = resolved[resolved['market_id'].notna()]
    unmatched = resolved[resolved['market_id'].isna()][['state', 'district', 'market']]
    print(f'  {len(matched)} matched to an existing market_id from the old file, '
          f'{len(unmatched)} are genuinely new -> assigning new IDs')

    unmatched = unmatched.reset_index(drop=True)
    unmatched['market_id_new'] = max_market_id + 1 + unmatched.index
    unmatched['district_id_new'] = max_district_id + 1 + unmatched.index

    _n_before = len(new)
    new = new.merge(
        matched[['state', 'market', 'market_id', 'district', 'district_id']]
        .rename(columns={'market_id': 'market_id_fill', 'district': 'district_fill',
                          'district_id': 'district_id_fill'}),
        on=['state', 'market'], how='left')
    assert len(new) == _n_before, f'Merge fan-out detected (matched): {_n_before} -> {len(new)} rows'
    _n_before = len(new)
    new = new.merge(unmatched[['state', 'market', 'market_id_new', 'district_id_new']],
                     on=['state', 'market'], how='left')
    assert len(new) == _n_before, f'Merge fan-out detected (unmatched): {_n_before} -> {len(new)} rows'
    new['market_id'] = new['market_id'].fillna(new['market_id_fill']).fillna(new['market_id_new'])
    new['district'] = new['district'].fillna(new['district_fill'])
    new['district_id'] = new['district_id'].fillna(new['district_id_fill']).fillna(new['district_id_new'])
    new = new.drop(columns=['market_id_fill', 'district_fill', 'district_id_fill',
                             'market_id_new', 'district_id_new'])

    assert new['market_id'].notna().all(), 'still have unresolved null market_id after backfill'
    new['market_id'] = new['market_id'].astype(int)
    new['district_id'] = new['district_id'].astype(int)

    # district name still null for the 9 genuinely-new markets -- fill with market name
    # as a readable placeholder (only used for display, not for any join key)
    new['district'] = new['district'].fillna(new['market'])

    new_rows = new[OUT_COLS]

    # ---- Sanity checks ----
    assert new_rows['modal_price_rs_per_quintal'].notna().all()
    assert new_rows['arrivals_tonnes'].notna().all()
    dupe_ids_conflict = new_rows.groupby('market_id')['market'].nunique()
    bad = dupe_ids_conflict[dupe_ids_conflict > 1]
    if not bad.empty:
        print(f'  NOTE: {len(bad)} market_id(s) map to >1 market name (likely a real '
              f'spelling-variant match, same as seen in the 09b merge) -- not treated as an error.')

    combined = pd.concat([old_pre_2025, new_rows], ignore_index=True)
    combined.to_csv(OUT_FILE, index=False)
    print(f'\nSaved: {OUT_FILE}')
    print(f'Total rows: {len(combined):,} (old pre-2025: {len(old_pre_2025):,} + new: {len(new_rows):,})')
    print(f'Total unique markets: {combined["market_id"].nunique()}')
    print(f'Date range: {combined["arrival_date"].min()} to {combined["arrival_date"].max()}')


if __name__ == '__main__':
    main()
