# -*- coding: utf-8 -*-
"""
Portal-report ingestion for the weekly refresh (added 2026-10-28)
=================================================================
Agmarknet's data API now requires a captcha (see README, "Captcha block"), so the automated scrape cannot run.
This module turns the two CSV reports a person downloads from the portal (agmarknet.gov.in -> Price & Arrivals ->
Download: "Daily Price Report" and "Daily Arrival Report", filtered to ONE commodity) into a file in the same
20-column schema the scraper wrote, so the existing gates (validate_scrape.py, merge_scrape.py) and the rest of
the pipeline run unchanged.

Generalised from scripts/09b_Merge_Onion_2026_Update.py (which handled onion 2026 only). Logic kept identical:
  * the price report can hold several variety/grade rows per (market, date); they are averaged to one value
    (Script 09 re-aggregates to a weekly arrival-weighted price anyway);
  * price and arrivals are joined on (state, district, market, date);
  * market_id comes from the trusted file by normalised (state, market) name; markets not yet in the trusted
    file get new sequential ids (new district ids likewise).

Where the reports go: for the CI run, in the raw-data folder on Drive, under
    TOP_Digital_Twin_Raw_Data/portal_reports/<crop>/   (any file names containing "Price" and "Arrival"; the newest
    pair by modified time is used). Locally: pass the paths to the CLI below.

CLI:
    python scripts/weekly_refresh/portal_reports.py --crop onion --price "Daily Price Report-....csv" \
        --arrival "Daily Arrival Report-....csv" --trusted onion_all_india_apmcs_2000_2026.csv --out onion_weekly_scrape.csv
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

OUT_COLS = ['source', 'commodity', 'commodity_id', 'state', 'state_id', 'state_code',
            'district', 'district_id', 'market', 'market_id', 'arrival_date',
            'arrival_date_raw', 'year', 'month', 'variety', 'arrivals_tonnes',
            'total_arrivals_tonnes', 'min_price_rs_per_quintal',
            'max_price_rs_per_quintal', 'modal_price_rs_per_quintal']
SOURCE_LABEL = 'AGMARKNET_PORTAL_REPORT'


def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]+', ' ', regex=True).str.strip()


def find_reports(downloads, crop):
    """Newest (price, arrival) pair under <downloads>/portal_reports/<crop>/, or None."""
    folder = os.path.join(str(downloads), 'portal_reports', crop.lower())
    files = [f for f in glob.glob(os.path.join(folder, '*.csv'))]
    price = sorted([f for f in files if re.search(r'price', os.path.basename(f), re.I)], key=os.path.getmtime)
    arr = sorted([f for f in files if re.search(r'arriv', os.path.basename(f), re.I)], key=os.path.getmtime)
    if price and arr:
        return price[-1], arr[-1]
    return None


def _read_report(path):
    """The portal's CSV has one title line above the header."""
    df = pd.read_csv(path, skiprows=1, low_memory=False)
    if 'Market' not in df.columns:       # no title line in this export
        df = pd.read_csv(path, low_memory=False)
    return df


def build_scrape(crop, price_csv, arrival_csv, trusted_csv, out_csv, log=print):
    crop_name = crop.capitalize()
    old = pd.read_csv(trusted_csv, low_memory=False)
    lookup = old.drop_duplicates(['state', 'market'])[['state', 'district', 'market', 'market_id', 'state_id', 'state_code', 'district_id', 'commodity_id']].copy()
    lookup['state_n'] = norm(lookup['state']); lookup['market_n'] = norm(lookup['market'])
    collisions = lookup.groupby(['state_n', 'market_n'])['market_id'].nunique()
    if (collisions > 1).any():
        log(f'  WARNING: {(collisions > 1).sum()} normalised (state, market) names map to more than one market_id; first-seen kept')
    lookup = lookup.drop_duplicates(['state_n', 'market_n'])
    max_market_id = int(old['market_id'].max()); max_district_id = int(old['district_id'].max())
    commodity_id = int(old['commodity_id'].dropna().iloc[0])

    price = _read_report(price_csv).rename(columns={'State/UT': 'state', 'District': 'district', 'Market': 'market', 'Price Date': 'date_raw',
                                                    'Modal Price': 'modal_price_rs_per_quintal', 'Min Price': 'min_price_rs_per_quintal', 'Max Price': 'max_price_rs_per_quintal'})
    need = ['state', 'district', 'market', 'date_raw', 'modal_price_rs_per_quintal', 'min_price_rs_per_quintal', 'max_price_rs_per_quintal']
    missing = [c for c in need if c not in price.columns]
    if missing:
        raise SystemExit(f'price report is missing columns {missing}; found {list(price.columns)}')
    for c in need[4:]:
        price[c] = pd.to_numeric(price[c].astype(str).str.replace(',', ''), errors='coerce')
    price_agg = price.groupby(['state', 'district', 'market', 'date_raw'], as_index=False).agg(
        modal_price_rs_per_quintal=('modal_price_rs_per_quintal', 'mean'), min_price_rs_per_quintal=('min_price_rs_per_quintal', 'mean'),
        max_price_rs_per_quintal=('max_price_rs_per_quintal', 'mean'))
    log(f'  price report: {len(price):,} rows -> {len(price_agg):,} (market, date) rows after variety average')

    arr = _read_report(arrival_csv).rename(columns={'State/UT': 'state', 'District': 'district', 'Market': 'market', 'Arrival Date': 'date_raw', 'Arrival Quantity': 'arrivals_tonnes'})
    if 'Arrival Unit' in arr.columns and not (arr['Arrival Unit'] == 'Metric Tonnes').all():
        raise SystemExit('unexpected arrival unit in the arrival report (expected Metric Tonnes only)')
    arr['arrivals_tonnes'] = pd.to_numeric(arr['arrivals_tonnes'].astype(str).str.replace(',', ''), errors='coerce')
    arr = arr[['state', 'district', 'market', 'date_raw', 'arrivals_tonnes']]
    m = price_agg.merge(arr, on=['state', 'district', 'market', 'date_raw'], how='inner')
    log(f'  matched (market, date) rows with both price and arrivals: {len(m):,}')

    m['state_n'] = norm(m['state']); m['market_n'] = norm(m['market'])
    m = m.merge(lookup[['state_n', 'market_n', 'market_id', 'state_id', 'state_code', 'district_id']], on=['state_n', 'market_n'], how='left')
    unmatched = m['market_id'].isna()
    keys = m.loc[unmatched, ['state_n', 'market_n']].drop_duplicates().reset_index(drop=True)
    keys['new_market_id'] = max_market_id + 1 + keys.index; keys['new_district_id'] = max_district_id + 1 + keys.index
    log(f'  markets matched to an existing market_id: {m.loc[~unmatched, ["state_n", "market_n"]].drop_duplicates().shape[0]}; new markets: {len(keys)}')
    m = m.merge(keys, on=['state_n', 'market_n'], how='left')
    m['market_id'] = m['market_id'].fillna(m['new_market_id']); m['district_id'] = m['district_id'].fillna(m['new_district_id'])
    st = old.drop_duplicates('state')[['state', 'state_id', 'state_code']]
    m = m.merge(st, on='state', how='left', suffixes=('', '_bystate'))
    m['state_id'] = m['state_id'].fillna(m['state_id_bystate']); m['state_code'] = m['state_code'].fillna(m['state_code_bystate'])

    m['source'] = SOURCE_LABEL; m['commodity'] = crop_name; m['commodity_id'] = commodity_id; m['variety'] = 'Mixed'
    m['total_arrivals_tonnes'] = m['arrivals_tonnes']
    m['arrival_date'] = pd.to_datetime(m['date_raw'], format='%d-%m-%Y', errors='coerce')
    m = m.dropna(subset=['arrival_date', 'modal_price_rs_per_quintal', 'arrivals_tonnes'])
    m['arrival_date_raw'] = m['date_raw']; m['year'] = m['arrival_date'].dt.year; m['month'] = m['arrival_date'].dt.month
    m['arrival_date'] = m['arrival_date'].dt.strftime('%Y-%m-%d')
    m['market_id'] = m['market_id'].astype(int); m['district_id'] = m['district_id'].astype(int); m['state_id'] = m['state_id'].astype('Int64')
    out = m[OUT_COLS]
    # market_ids that the TRUSTED file already shares between different states (found 2026-10-28: 4846 and 4848-4852 in the onion file,
    # a collision between ids assigned by Script 09b and ids from the scraper's directory) are reported, not fatal; a collision that
    # involves an id created by THIS run is fatal.
    st_per_id = old.assign(_s=old['state'].replace({'Keralam': 'Kerala'})).groupby('market_id')['_s'].nunique()
    pre_existing = set(st_per_id[st_per_id > 1].index.astype(int))
    bad = out.groupby('market_id')['market'].nunique()
    bad_ids = set(bad[bad > 1].index.astype(int))
    if bad_ids - pre_existing:
        raise SystemExit(f'market_id maps to more than one market name: {sorted(bad_ids - pre_existing)[:10]}')
    if bad_ids:
        log(f'  WARNING: {len(bad_ids)} market_id(s) are shared by different markets in the trusted file already ({sorted(bad_ids)[:8]}); needs the id repair')
    # every (market_id, normalised name) in the new rows must already exist in the trusted file (any spelling variant recorded for that id),
    # unless the id was created by this run; this catches a report market matched to the wrong id
    _o = old.dropna(subset=['market_id'])
    known = set(zip(_o['market_id'].astype(int), norm(_o['market']))) | set(zip(_o['market_id'].astype(int), norm(_o['market'].astype(str).str.replace(r'\(.*?\)', '', regex=True))))
    fresh = out.assign(_n=norm(out['market']))
    notknown = fresh[(fresh['market_id'] <= max_market_id) & ~pd.Series(list(zip(fresh['market_id'], fresh['_n'])), index=fresh.index).isin(known)]
    if len(notknown):
        raise SystemExit('market(s) matched to an existing market_id under a name never recorded for that id: ' + str(notknown[['market_id', 'market']].drop_duplicates().head(8).values.tolist()))
    out.to_csv(out_csv, index=False)
    log(f'  wrote {out_csv}: {len(out):,} rows, dates {out["arrival_date"].min()} to {out["arrival_date"].max()}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--crop', required=True, choices=['tomato', 'onion', 'potato'])
    ap.add_argument('--price', required=True); ap.add_argument('--arrival', required=True)
    ap.add_argument('--trusted', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    build_scrape(a.crop, a.price, a.arrival, a.trusted, a.out)
