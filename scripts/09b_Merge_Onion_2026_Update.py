# -*- coding: utf-8 -*-
"""
Script 09b — Merge Onion 2026 Agmarknet Portal Update
=========================================================
One-off/refresh utility: onion's original raw source
(onion_all_india_apmcs_2000_2025.csv, from a scraper) only covers through
Dec 2025 and has no 2026 successor from that same scraper. Tomato/Potato's
scraper source already extends into mid-2026 (2000_2026.csv files), but
onion's does not, so 2026 onion data comes from a DIFFERENT source: manual
"Daily Price Report" + "Daily Arrival Report" downloads from the Agmarknet
portal (agmarknet.gov.in), which use their own schema and don't carry the
scraper's market_id.

This script merges those two portal reports into the SAME row schema as
the original onion_all_india_apmcs_2000_2025.csv (so it can be appended
and read by Script 09 unchanged), matching markets to existing market_id
by normalized (state, market) name where possible, and assigning new
sequential IDs for markets that genuinely don't appear in the 2000-2025
scrape (~62% of markets in the 2026 portal export -- a real market-
coverage gap between the old scraper and the live portal, confirmed by
checking that normalizing case/punctuation doesn't close the gap).

Inputs:
  Downloads/onion_all_india_apmcs_2000_2025.csv           (existing, 2000-2025)
  Downloads/Daily Price Report-01-01-2026 to 24-07-2026.csv
  Downloads/Daily Arrival Report-01-01-2026 to 24-07-2026.csv

Output:
  Downloads/onion_all_india_apmcs_2000_2026.csv
    (old 2000-2025 rows + new 2026 rows, same 20-column schema,
     ready to be Script 09's INFILES['onion'] path)

Caveat: the portal's price report can have multiple variety/grade rows
per (market, date) (~5% of rows) while the arrivals report has exactly
one row per (market, date) with no variety breakdown. We can't correctly
split one arrivals total across several variety-price rows without an
allocation assumption, so price is averaged to one value per (market,
date) before merging with arrivals -- a simplification, not a precision
loss that matters given Script 09 immediately re-aggregates to a single
weekly arrivals-weighted price per market anyway.
"""

import os
import re
import pandas as pd
import numpy as np

DOWNLOADS = r'C:\Users\masro\Downloads'
OLD_FILE = os.path.join(DOWNLOADS, 'onion_all_india_apmcs_2000_2025.csv')
PRICE_FILE = os.path.join(DOWNLOADS, 'Daily Price Report-01-01-2026 to 24-07-2026.csv')
ARRIVAL_FILE = os.path.join(DOWNLOADS, 'Daily Arrival Report-01-01-2026 to 24-07-2026.csv')
OUT_FILE = os.path.join(DOWNLOADS, 'onion_all_india_apmcs_2000_2026.csv')

OUT_COLS = ['source', 'commodity', 'commodity_id', 'state', 'state_id', 'state_code',
            'district', 'district_id', 'market', 'market_id', 'arrival_date',
            'arrival_date_raw', 'year', 'month', 'variety', 'arrivals_tonnes',
            'total_arrivals_tonnes', 'min_price_rs_per_quintal',
            'max_price_rs_per_quintal', 'modal_price_rs_per_quintal']


def norm(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower().str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip()


def main():
    print('Loading existing onion 2000-2025 file ...')
    old = pd.read_csv(OLD_FILE, low_memory=False)
    print(f'  {len(old):,} rows, {old["market_id"].nunique()} markets')

    # ---- Build market lookup: normalized (state, market) -> full metadata ----
    lookup = old.drop_duplicates(['state', 'market'])[
        ['state', 'district', 'market', 'market_id', 'state_id', 'state_code',
         'district_id', 'commodity_id', 'source']
    ].copy()
    lookup['state_n'] = norm(lookup['state'])
    lookup['market_n'] = norm(lookup['market'])
    lookup = lookup.drop_duplicates(['state_n', 'market_n'])

    max_market_id = int(old['market_id'].max())
    max_district_id = int(old['district_id'].max())
    print(f'  Existing max market_id={max_market_id}, max district_id={max_district_id}')

    # ---- Load and reshape the two portal reports ----
    print('Loading 2026 Daily Price Report ...')
    price = pd.read_csv(PRICE_FILE, skiprows=1)
    price = price.rename(columns={
        'State/UT': 'state', 'District': 'district', 'Market': 'market',
        'Price Date': 'date_raw', 'Modal Price': 'modal_price_rs_per_quintal',
        'Min Price': 'min_price_rs_per_quintal', 'Max Price': 'max_price_rs_per_quintal',
    })
    for c in ['modal_price_rs_per_quintal', 'min_price_rs_per_quintal', 'max_price_rs_per_quintal']:
        price[c] = pd.to_numeric(price[c].astype(str).str.replace(',', ''), errors='coerce')

    # Collapse variety/grade rows to one price per (state, district, market, date):
    # simple mean across varieties -- see module docstring caveat.
    price_agg = (
        price.groupby(['state', 'district', 'market', 'date_raw'], as_index=False)
        .agg(modal_price_rs_per_quintal=('modal_price_rs_per_quintal', 'mean'),
             min_price_rs_per_quintal=('min_price_rs_per_quintal', 'mean'),
             max_price_rs_per_quintal=('max_price_rs_per_quintal', 'mean'))
    )
    print(f'  {len(price):,} raw price rows -> {len(price_agg):,} (market,date) rows after variety collapse')

    print('Loading 2026 Daily Arrival Report ...')
    arrivals = pd.read_csv(ARRIVAL_FILE, skiprows=1)
    arrivals = arrivals.rename(columns={
        'State/UT': 'state', 'District': 'district', 'Market': 'market',
        'Arrival Date': 'date_raw', 'Arrival Quantity': 'arrivals_tonnes',
    })
    assert (arrivals['Arrival Unit'] == 'Metric Tonnes').all(), 'unexpected arrival unit found'
    arrivals['arrivals_tonnes'] = pd.to_numeric(
        arrivals['arrivals_tonnes'].astype(str).str.replace(',', ''), errors='coerce')
    arrivals = arrivals[['state', 'district', 'market', 'date_raw', 'arrivals_tonnes']]

    print('Merging price + arrivals on (state, district, market, date) ...')
    merged = price_agg.merge(arrivals, on=['state', 'district', 'market', 'date_raw'], how='inner')
    print(f'  {len(merged):,} matched (market,date) rows with both price and arrivals')

    # ---- Assign market_id via normalized (state, market) lookup; new IDs for the rest ----
    merged['state_n'] = norm(merged['state'])
    merged['market_n'] = norm(merged['market'])
    merged = merged.merge(
        lookup[['state_n', 'market_n', 'market_id', 'state_id', 'state_code', 'district_id', 'commodity_id']],
        on=['state_n', 'market_n'], how='left'
    )

    unmatched_mask = merged['market_id'].isna()
    n_unmatched_markets = merged.loc[unmatched_mask, ['state_n', 'market_n']].drop_duplicates().shape[0]
    print(f'  {merged.loc[~unmatched_mask, ["state_n","market_n"]].drop_duplicates().shape[0]} markets matched '
          f'to existing market_id; {n_unmatched_markets} markets are new -> assigning new IDs')

    new_market_keys = merged.loc[unmatched_mask, ['state_n', 'market_n']].drop_duplicates().reset_index(drop=True)
    new_market_keys['new_market_id'] = max_market_id + 1 + new_market_keys.index
    new_market_keys['new_district_id'] = max_district_id + 1 + new_market_keys.index  # 1 district assumed per new market_id

    # Use an explicit merge (not index-alignment fillna) so duplicate (state_n, market_n)
    # keys across many date-rows broadcast correctly instead of risking misalignment.
    merged = merged.merge(new_market_keys, on=['state_n', 'market_n'], how='left')
    merged['market_id'] = merged['market_id'].fillna(merged['new_market_id'])
    merged['district_id'] = merged['district_id'].fillna(merged['new_district_id'])
    merged = merged.drop(columns=['new_market_id', 'new_district_id'])

    # state_id/state_code/commodity_id: fill from lookup by state alone where market-level match failed
    state_lookup = old.drop_duplicates('state')[['state', 'state_id', 'state_code']]
    merged = merged.merge(state_lookup, on='state', how='left', suffixes=('', '_bystate'))
    merged['state_id'] = merged['state_id'].fillna(merged['state_id_bystate'])
    merged['state_code'] = merged['state_code'].fillna(merged['state_code_bystate'])
    merged = merged.drop(columns=['state_id_bystate', 'state_code_bystate'])
    merged['commodity_id'] = merged['commodity_id'].fillna(23)  # 23 = Onion, per existing file

    # ---- Fill remaining OUT_COLS ----
    merged['source'] = 'AGMARKNET_PORTAL_2026'
    merged['commodity'] = 'Onion'
    merged['variety'] = 'Mixed'  # averaged across varieties -- see docstring caveat
    merged['total_arrivals_tonnes'] = merged['arrivals_tonnes']
    merged['arrival_date'] = pd.to_datetime(merged['date_raw'], format='%d-%m-%Y', errors='coerce')
    merged = merged.dropna(subset=['arrival_date'])
    merged['arrival_date_raw'] = merged['date_raw']
    merged['year'] = merged['arrival_date'].dt.year
    merged['month'] = merged['arrival_date'].dt.month
    merged['arrival_date'] = merged['arrival_date'].dt.strftime('%Y-%m-%d')

    merged['market_id'] = merged['market_id'].astype(int)
    merged['district_id'] = merged['district_id'].astype(int)
    merged['state_id'] = merged['state_id'].astype('Int64')
    merged['commodity_id'] = merged['commodity_id'].astype(int)

    new_rows = merged[OUT_COLS]

    # ---- Sanity checks before writing ----
    assert new_rows['modal_price_rs_per_quintal'].notna().all()
    assert new_rows['arrivals_tonnes'].notna().all()
    assert new_rows['arrival_date'].between('2026-01-01', '2026-07-24').all()
    dupe_ids_conflict = (
        new_rows.groupby('market_id')['market'].nunique()
    )
    bad = dupe_ids_conflict[dupe_ids_conflict > 1]
    assert bad.empty, f'market_id maps to >1 market name: {bad.index.tolist()}'

    print(f'\n2026 new rows: {len(new_rows):,}  '
          f'markets: {new_rows["market_id"].nunique()}  '
          f'date range: {new_rows["arrival_date"].min()} to {new_rows["arrival_date"].max()}')

    combined = pd.concat([old, new_rows], ignore_index=True)
    combined.to_csv(OUT_FILE, index=False)
    print(f'\nSaved: {OUT_FILE}')
    print(f'Total rows: {len(combined):,} (old {len(old):,} + new {len(new_rows):,})')
    print(f'Total unique markets: {combined["market_id"].nunique()}')
    print(f'Date range: {combined["arrival_date"].min()} to {combined["arrival_date"].max()}')


if __name__ == '__main__':
    main()
