# -*- coding: utf-8 -*-
"""
Script 09d — Rebuild Tomato + Potato 2026 From an Updated Scraper Pull
====================================================================
Same pattern as 09c (which did this for onion): a newer scraper export
turned up (`test_tomato_2026.csv`, `test_potato_2026.csv`) covering
2025-01-01 through 2026-07-27 -- well past the previous files' real
cutoffs (tomato 2026-06-24, potato 2026-07-01).

Verified before trusting them:
  - Correct commodity in every row (no Banana-under-onion-filename repeat
    of the earlier mislabeled file incident).
  - Overlapping 2025 dates match the old files EXACTLY on a market common
    to both (Kolar APMC for tomato, Agra APMC for potato -- the first
    market tried for potato, Palamaner APMC, turned out to be new-only,
    not a comparison failure).
  - Zero genuine market_id conflicts: every (state, market) present in
    both old and new files maps to the SAME market_id already.

Inputs:
  Downloads/tomato_all_india_apmcs_2000_2026.csv   (existing, through 2026-06-24)
  Downloads/test_tomato_2026.csv                    (new, through 2026-07-27)
  Downloads/potato_all_india_apmcs_2000_2026.csv    (existing, through 2026-07-01)
  Downloads/test_potato_2026.csv                    (new, through 2026-07-27)

Outputs (overwritten in place -- same filenames Script 09's INFILES expects):
  Downloads/tomato_all_india_apmcs_2000_2026.csv
  Downloads/potato_all_india_apmcs_2000_2026.csv
"""

import os
import pandas as pd
import numpy as np

DOWNLOADS = os.environ.get('TOP_DOWNLOADS_DIR', r'C:\Users\masro\Downloads')

OUT_COLS = ['source', 'commodity', 'commodity_id', 'state', 'state_id', 'state_code',
            'district', 'district_id', 'market', 'market_id', 'arrival_date',
            'arrival_date_raw', 'year', 'month', 'variety', 'arrivals_tonnes',
            'total_arrivals_tonnes', 'min_price_rs_per_quintal',
            'max_price_rs_per_quintal', 'modal_price_rs_per_quintal']

CROPS = {
    'tomato': dict(old_file='tomato_all_india_apmcs_2000_2026.csv', new_file='test_tomato_2026.csv'),
    'potato': dict(old_file='potato_all_india_apmcs_2000_2026.csv', new_file='test_potato_2026.csv'),
}


def rebuild_crop(crop, old_file, new_file):
    print(f'\n{"="*60}\n{crop.upper()}\n{"="*60}')
    old = pd.read_csv(os.path.join(DOWNLOADS, old_file), low_memory=False)
    old_pre_2025 = old[old['arrival_date'] < '2025-01-01'].copy()
    print(f'  Old file: {len(old):,} rows -> {len(old_pre_2025):,} rows before 2025-01-01 (kept)')
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

    combined = pd.concat([old_pre_2025, new_rows], ignore_index=True)
    out_path = os.path.join(DOWNLOADS, old_file)
    combined.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')
    print(f'  Total rows: {len(combined):,}  (old pre-2025: {len(old_pre_2025):,} + new: {len(new_rows):,})')
    print(f'  Total unique markets: {combined["market_id"].nunique()}')
    print(f'  Date range: {combined["arrival_date"].min()} to {combined["arrival_date"].max()}')


if __name__ == '__main__':
    for crop, paths in CROPS.items():
        rebuild_crop(crop, paths['old_file'], paths['new_file'])
