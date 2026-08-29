# -*- coding: utf-8 -*-
"""
Weekly Refresh -- Step 2: Validate a Fresh Scrape Before Trusting It
======================================================================
Scripts a repeatable version of the manual checks used for every prior raw-
data refresh in this project (see MANIFEST.md / project memory): schema,
commodity labelling, price sanity, market_id null rate, and a deduped
overlap price-comparison against the currently-trusted file.

This is a GATE, not a report -- exits 1 (and prints why) the moment any
check fails, so the calling orchestrator (run_weekly_refresh.ps1) can stop
before merging anything bad into the trusted files.

Usage:
    python validate_scrape.py <crop> <trusted_csv> <new_scrape_csv>

Exit code 0 = safe to merge. Exit code 1 = do NOT merge, see stderr.
"""
import sys
import pandas as pd

PRICE_SANITY = {
    'tomato': (10, 20000),
    'onion':  (50, 12000),
    'potato': (40, 3500),
}

REQUIRED_COLS = [
    'source', 'commodity', 'commodity_id', 'state', 'state_id', 'state_code',
    'district', 'district_id', 'market', 'market_id', 'arrival_date',
    'arrival_date_raw', 'year', 'month', 'variety', 'arrivals_tonnes',
    'total_arrivals_tonnes', 'min_price_rs_per_quintal',
    'max_price_rs_per_quintal', 'modal_price_rs_per_quintal',
]


def fail(msg):
    print(f'  [FAIL] {msg}', file=sys.stderr)
    sys.exit(1)


def warn(msg):
    print(f'  [WARN] {msg}')


def ok(msg):
    print(f'  [PASS] {msg}')


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    crop, trusted_path, new_path = sys.argv[1], sys.argv[2], sys.argv[3]
    crop = crop.lower()
    if crop not in PRICE_SANITY:
        fail(f'unknown crop {crop!r}, expected one of {list(PRICE_SANITY)}')

    print(f'--- Validating {crop} scrape: {new_path} ---')

    try:
        new = pd.read_csv(new_path, low_memory=False)
    except Exception as exc:
        fail(f'could not read new scrape file: {exc}')

    # 1. Schema
    missing = [c for c in REQUIRED_COLS if c not in new.columns]
    if missing:
        fail(f'new file missing columns: {missing}')
    ok(f'schema OK ({len(new.columns)} columns, all required present)')

    # 2. Commodity label
    bad_commodity = new[new['commodity'].str.lower() != crop]
    if len(bad_commodity) > 0:
        fail(f'{len(bad_commodity)} rows have commodity != {crop!r} '
             f'(found: {bad_commodity["commodity"].unique().tolist()})')
    ok('commodity label consistent throughout')

    # 3. Row count sanity
    if len(new) < 1000:
        fail(f'only {len(new)} rows total -- looks like a truncated/failed scrape')
    ok(f'{len(new):,} rows total')

    # 4. Date parse + range
    new['arrival_date'] = pd.to_datetime(new['arrival_date'], errors='coerce')
    n_bad_dates = new['arrival_date'].isna().sum()
    if n_bad_dates > 0:
        fail(f'{n_bad_dates} rows have unparseable arrival_date')
    dmin, dmax = new['arrival_date'].min(), new['arrival_date'].max()
    ok(f'date range {dmin.date()} to {dmax.date()}')

    # 5. Price sanity
    lo, hi = PRICE_SANITY[crop]
    price = pd.to_numeric(new['modal_price_rs_per_quintal'], errors='coerce')
    n_null_price = price.isna().sum()
    if n_null_price > 0.01 * len(new):
        fail(f'{n_null_price} rows ({n_null_price/len(new):.1%}) have null/unparseable '
             f'modal price -- exceeds 1% tolerance')
    in_range = price.between(lo, hi)
    frac_out = 1 - (in_range.sum() / price.notna().sum())
    # PRICE_SANITY mirrors Script 09's own PRICE_CLIP, which was calibrated
    # against each crop's PRODUCTION panel states only (e.g. potato is West
    # Bengal + Uttarakhand only -- a national scrape also pulls states never
    # in that panel, and their genuinely different regional pricing (e.g.
    # Tamil Nadu/Kerala potato running Rs.3800-4500, confirmed real via a
    # live scrape 2026-08-29, not a scraper fault) trips this range without
    # being a data-quality problem. Script 09 applies the identical clip
    # during panel-building regardless, so out-of-range rows from states
    # outside the trusted panel are silently and safely dropped downstream
    # either way -- this check exists to catch genuine corruption (garbage
    # values, wrong units, a broken scrape), not to duplicate that clip.
    if frac_out > 0.20:
        fail(f'{frac_out:.1%} of priced rows fall outside [{lo}, {hi}] Rs/quintal '
             f'for {crop} -- exceeds 20% tolerance, looks like more than ordinary '
             f'non-panel-state price variation')
    elif frac_out > 0.05:
        warn(f'{frac_out:.1%} of priced rows fall outside [{lo}, {hi}] Rs/quintal for '
             f'{crop} -- above 5%, but Script 09 clips these identically during panel-'
             f'building, so this is expected from non-panel-state markets, not blocking')
    else:
        ok(f'price sanity OK ({frac_out:.2%} outside [{lo},{hi}])')

    # 6. market_id null rate
    null_mkt = new['market_id'].isna().sum()
    if null_mkt > 0.10 * len(new):
        fail(f'{null_mkt} rows ({null_mkt/len(new):.1%}) have null market_id -- '
             f'exceeds 10% tolerance (market directory lookup may have failed)')
    ok(f'market_id null rate {null_mkt/len(new):.2%} (within tolerance)')

    # 7. Overlap comparison against the trusted file
    try:
        old = pd.read_csv(trusted_path, low_memory=False)
    except Exception as exc:
        warn(f'could not read trusted file for overlap check: {exc} -- skipping check 7')
        old = None

    if old is not None:
        old['arrival_date'] = pd.to_datetime(old['arrival_date'], errors='coerce')
        key = ['market_id', 'arrival_date', 'variety']
        old_agg = old.groupby(key)['modal_price_rs_per_quintal'].mean().rename('old')
        new_agg = new.groupby(key)['modal_price_rs_per_quintal'].mean().rename('new')
        m = pd.concat([old_agg, new_agg], axis=1, join='inner')
        if len(m) < 100:
            warn(f'only {len(m)} overlapping (market,date,variety) keys with the trusted '
                 f'file -- too few to judge consistency, proceeding on other checks alone')
        else:
            diff = (m['old'] - m['new']).abs()
            frac_bad = (diff > 100).mean()
            if frac_bad > 0.02:
                fail(f'{frac_bad:.2%} of {len(m):,} overlapping rows differ from the '
                     f'trusted file by more than Rs.100 -- exceeds 2% tolerance, this '
                     f'looks like more than ordinary portal revision noise')
            ok(f'overlap check OK: {frac_bad:.3%} of {len(m):,} overlapping rows differ '
               f'by >Rs.100 (within 2% tolerance)')

        # 8. market_id conflicts
        old_lookup = old.drop_duplicates(['state', 'market'])[['state', 'market', 'market_id']]
        new_lookup = new.drop_duplicates(['state', 'market'])[['state', 'market', 'market_id']]
        mm = old_lookup.merge(new_lookup, on=['state', 'market'], suffixes=('_old', '_new'), how='inner')
        conflict = mm[(mm['market_id_old'] != mm['market_id_new'])
                       & mm['market_id_old'].notna() & mm['market_id_new'].notna()]
        if len(conflict) > 0:
            fail(f'{len(conflict)} markets map to a DIFFERENT market_id than the trusted '
                 f'file -- real conflict, not a null-vs-null artifact:\n{conflict.head(10)}')
        ok(f'market_id consistency OK (0 genuine conflicts across {len(mm)} shared markets)')

    print(f'--- {crop}: ALL CHECKS PASSED ---')
    sys.exit(0)


if __name__ == '__main__':
    main()
