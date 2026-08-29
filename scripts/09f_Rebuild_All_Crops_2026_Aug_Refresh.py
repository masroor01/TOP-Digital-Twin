# -*- coding: utf-8 -*-
"""
Script 09f — Rebuild Tomato + Onion + Potato 2026 From the 2026-08-21/27 Scraper Pulls
========================================================================================
Same pattern as 09c/09d: fresher scraper exports turned up for all three crops --

  potato_all_india_apmcs_2026_new_edit.csv   (real data through 2026-08-21)
  onion_all_india_apmcs_2026_new_edit.csv    (real data through 2026-08-21)
  tomato_all_india_apmcs_2026_new (1).csv    (through 2026-08-12 -- NOT an extension
                                               beyond the current main file's own
                                               2026-08-12 cutoff, but fills 619 rows
                                               of real gaps inside the existing window,
                                               mostly 2026-07-24 to 07-27)

Verified before trusting them (same rigor as every prior refresh):
  - Correct commodity in every row for all three files.
  - Overlapping 2026 dates match the current main files almost exactly on a
    deduped (market_id, date, variety) key: potato 9/125,661 rows differ by
    >Rs.1 (0.01%), onion 20/139,047 (0.014%), tomato 8/134,367 (0.006%) --
    all consistent with ordinary portal-side revisions, not tampering.
  - Zero genuine market_id conflicts for any of the three crops (a "conflict"
    only ever meant one file had NaN where the other had a real ID).
  - Daily row-volume pattern for all three shows a realistic weekday/weekend
    shape through each file's own cutoff, not a fabrication red flag.

Inputs:
  Downloads/tomato_all_india_apmcs_2000_2026.csv   (existing, through 2026-08-12)
  Downloads/tomato_all_india_apmcs_2026_new (1).csv
  Downloads/onion_all_india_apmcs_2000_2026.csv    (existing, through 2026-08-12)
  Downloads/onion_all_india_apmcs_2026_new_edit.csv
  Downloads/potato_all_india_apmcs_2000_2026.csv   (existing, through 2026-07-27)
  Downloads/potato_all_india_apmcs_2026_new_edit.csv

Outputs (overwritten in place -- same filenames Script 09's INFILES expects):
  Downloads/tomato_all_india_apmcs_2000_2026.csv
  Downloads/onion_all_india_apmcs_2000_2026.csv
  Downloads/potato_all_india_apmcs_2000_2026.csv

Strategy: keep everything strictly before 2026-01-01 from the old file untouched
(the new files all start 2026-01-01), and REPLACE the entire 2026 portion with
the new file's 2026 data. This is a full replace, not an append -- it correctly
picks up tomato's mid-window backfill as well as potato/onion's forward
extension, in one consistent operation across all three crops.
"""

import os
import pandas as pd
import numpy as np

DOWNLOADS = r'C:\Users\masro\Downloads'

OUT_COLS = ['source', 'commodity', 'commodity_id', 'state', 'state_id', 'state_code',
            'district', 'district_id', 'market', 'market_id', 'arrival_date',
            'arrival_date_raw', 'year', 'month', 'variety', 'arrivals_tonnes',
            'total_arrivals_tonnes', 'min_price_rs_per_quintal',
            'max_price_rs_per_quintal', 'modal_price_rs_per_quintal']

CROPS = {
    'tomato': dict(old_file='tomato_all_india_apmcs_2000_2026.csv',
                    new_file='tomato_all_india_apmcs_2026_new (1).csv'),
    'onion':  dict(old_file='onion_all_india_apmcs_2000_2026.csv',
                    new_file='onion_all_india_apmcs_2026_new_edit.csv'),
    'potato': dict(old_file='potato_all_india_apmcs_2000_2026.csv',
                    new_file='potato_all_india_apmcs_2026_new_edit.csv'),
}


def rebuild_crop(crop, old_file, new_file):
    print(f'\n{"="*60}\n{crop.upper()}\n{"="*60}')
    old = pd.read_csv(os.path.join(DOWNLOADS, old_file), low_memory=False)
    old_pre_2026 = old[old['arrival_date'] < '2026-01-01'].copy()
    print(f'  Old file: {len(old):,} rows -> {len(old_pre_2026):,} rows before 2026-01-01 (kept)')
    max_market_id = int(old['market_id'].max())
    max_district_id = int(old['district_id'].max())

    new = pd.read_csv(os.path.join(DOWNLOADS, new_file), low_memory=False)
    print(f'  New file: {len(new):,} rows, {new["arrival_date"].min()} to {new["arrival_date"].max()}')
    assert (new['commodity'].str.lower() == crop).all(), f'non-{crop} rows found in {new_file}'

    null_mask = new['market_id'].isna()
    null_markets = new.loc[null_mask, ['state', 'district', 'market']].drop_duplicates()
    print(f'  {null_mask.sum():,} rows with null market_id, {len(null_markets)} distinct markets')

    old_lookup = old.drop_duplicates(['state', 'market'])[
        ['state', 'market', 'market_id', 'district', 'district_id']
    ]
    resolved = null_markets.merge(old_lookup, on=['state', 'market'], how='left', suffixes=('', '_old'))
    matched = resolved[resolved['market_id'].notna()]
    unmatched = resolved[resolved['market_id'].isna()][['state', 'district', 'market']].reset_index(drop=True)
    print(f'  {len(matched)} matched to an existing market_id, {len(unmatched)} genuinely new -> assigning new IDs')

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
        print(f'  NOTE: {len(bad)} market_id(s) map to >1 market name (spelling-variant match, '
              f'as seen before) -- not treated as an error.')

    combined = pd.concat([old_pre_2026, new_rows], ignore_index=True)
    out_path = os.path.join(DOWNLOADS, old_file)
    combined.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')
    print(f'  Total rows: {len(combined):,}  (old pre-2026: {len(old_pre_2026):,} + new: {len(new_rows):,})')
    print(f'  Total unique markets: {combined["market_id"].nunique()}')
    print(f'  Date range: {combined["arrival_date"].min()} to {combined["arrival_date"].max()}')


if __name__ == '__main__':
    for crop, paths in CROPS.items():
        rebuild_crop(crop, paths['old_file'], paths['new_file'])

    print('\nScript 09f complete.')
    print('Next: bump END_DATE in Script 09 to 2026-08-21, then re-run')
    print('  python scripts/09_Agmarknet_Weekly_Panel.py')
