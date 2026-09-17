# -*- coding: utf-8 -*-
"""
Script 57 — Input Cost Layer: Fertilizer MRP (Department of Fertilizers)
==============================================================
Acquires the Maximum Retail Price (MRP) of P&K fertilizers, and Urea's
fixed price, from the Department of Fertilizers' Monthly Bulletin PDFs
(https://fert.gov.in) -- a single, national/global-level (PAN-India, not
state/district) monthly series, ~53 product variants in recent bulletins
(fewer in older ones, product list has grown over the years).

WHY THIS LAYER IS STRUCTURED LIKE SCRIPT 56 (ONI), NOT 51/53 (DROUGHT):
  Like ONI, this is a single national time series with no spatial
  dimension -- MRPs are centrally fixed (Nutrient-Based Subsidy scheme),
  not state-varying, so there's no crosswalk/join problem. This script
  only acquires and parses; it does NOT join onto the weekly panel.

HOW THIS WAS VALIDATED (2026-09-14 session, acquisition built 2026-09-17):
  - Two listing pages enumerate every bulletin, both plain HTML (no JS
    needed, confirmed via a plain `requests.get` -- matches what the
    browser rendered):
      https://fert.gov.in/en/documents/reports/monthly-bulletin  (recent, ~33)
      https://fert.gov.in/en/archives/monthly_bulletin           (older, ~71, through Jan 2018)
    Each bulletin's own filename is NOT a reliable month/year source --
    inconsistent formats, typos ("Monthy", "Montthly"), some Hindi-titled
    duplicates of the same month. Month/year is instead extracted from
    each PDF's own title text ("Monthly Bulletin for/the month of X, YYYY"),
    which was confirmed present and consistent across 2018/2022/2025
    samples before trusting it as the parsing key.
  - The MRP table's own section header text has changed wording across
    the archive: "Domestic Prices of Fertilizers" (seen 2018/2022) vs.
    "Domestic Prices (MRP) of P&K Fertilizers" (seen 2025) -- the parsing
    regex is deliberately tolerant of both ("(MRP)" and "P&K" optional),
    confirmed to correctly extract all three eras' tables (28 rows/2018,
    33 rows/2022, 53 rows/2025) before trusting a full historical pull.
  - Urea's fixed price is a separate sentence ("The price of Urea is
    fixed at Rs. X (excluding local taxes) per MT"), not a table row --
    confirmed IDENTICAL wording and value (Rs 5360/MT) in both the
    Jan-2018 and May-2025 samples tested; this is a real, genuinely
    long-stationary centrally-fixed price (Nutrient-Based Subsidy keeps
    Urea's price administratively fixed, unlike P&K's more frequently
    revised MRPs), not a parsing bug -- verified by reading the literal
    sentence in both PDFs before trusting it.
  - Some bulletins are duplicated under multiple filenames per month
    (Hindi + English versions, or a re-uploaded "(1)" correction) --
    deduplicated by the (year, month) parsed from the PDF's own title
    text, not by filename, keeping the first successfully-parsed copy.

Output:
  data/fertilizer_mrp/fert_mrp_monthly.csv
    columns: year, month, product, mrp_rs_per_mt, source_url
    (Urea's fixed price appears as a row with product="Urea")
  data/fertilizer_mrp/raw_cache/*.pdf (cached so a re-run doesn't re-download)

Run:
  python scripts/57_Fertilizer_MRP_Layer.py --mode probe
  python scripts/57_Fertilizer_MRP_Layer.py --mode full
"""

import os
import re
import argparse
from urllib.parse import urljoin, unquote

import requests
import pdfplumber

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'data', 'fertilizer_mrp')
CACHE_DIR = os.path.join(OUT_DIR, 'raw_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

LISTING_URLS = [
    'https://fert.gov.in/en/documents/reports/monthly-bulletin',
    'https://fert.gov.in/en/archives/monthly_bulletin',
]
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {'User-Agent': 'Mozilla/5.0'}

TITLE_RE = re.compile(
    r'Monthly Bulletin\s+(?:for\s+)?the\s+month\s+of\s+([A-Za-z]+),?\s+(\d{4})',
    re.IGNORECASE)
UREA_RE = re.compile(r'price of Urea is fixed at Rs\.?\s*([\d,]+\.?\d*)', re.IGNORECASE)
TABLE_RE = re.compile(
    r'Domestic Prices\s*(?:\(MRP\))?\s*of\s*(?:P\s*&?\s*K\s*)?Fertilizers(.*?)Source:',
    re.DOTALL | re.IGNORECASE)
ROW_RE = re.compile(r'(?m)^\s*(\d{1,2})\s+(.+?)\s+([\d,]+\.?\d*)\s*$')

MONTH_NUM = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
}


def discover_bulletin_urls(session):
    """Returns a list of absolute PDF URLs, English-titled only (Hindi
    duplicates are excluded here by filename -- the Latin-script filter --
    not by content, since we haven't downloaded them yet)."""
    urls = []
    seen = set()
    for listing_url in LISTING_URLS:
        resp = session.get(listing_url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        resp.raise_for_status()
        hrefs = re.findall(r'href="([^"]+\.pdf)"', resp.text)
        for href in hrefs:
            abs_url = urljoin(listing_url, href)
            decoded_name = unquote(abs_url)
            if not re.search(r'bulletin', decoded_name, re.IGNORECASE):
                continue  # Hindi-titled duplicate or something unrelated
            if abs_url not in seen:
                seen.add(abs_url)
                urls.append(abs_url)
    return urls


def cache_path(url):
    fname = re.sub(r'[^A-Za-z0-9_.-]', '_', os.path.basename(url))[:150]
    return os.path.join(CACHE_DIR, fname)


def fetch_pdf(url, session):
    path = cache_path(url)
    if os.path.exists(path):
        return path
    resp = session.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    with open(path, 'wb') as f:
        f.write(resp.content)
    return path


def parse_bulletin(path, url):
    """Returns (year, month, urea_price, rows, error). rows is a list of
    (product, price_rs_mt). On failure, year/month/urea_price/rows are
    None/None/None/[] and error explains why."""
    try:
        with pdfplumber.open(path) as pdf:
            first_page = pdf.pages[0].extract_text() or ""
            full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        return None, None, None, [], f'pdf_open_error: {e}'

    title_m = TITLE_RE.search(first_page) or TITLE_RE.search(full_text)
    if not title_m:
        return None, None, None, [], 'title_not_found'
    month_name, year = title_m.group(1).lower(), int(title_m.group(2))
    month = MONTH_NUM.get(month_name)
    if month is None:
        return None, None, None, [], f'unrecognized_month: {month_name}'

    urea_m = UREA_RE.search(full_text)
    urea_price = float(urea_m.group(1).replace(',', '')) if urea_m else None

    table_m = TABLE_RE.search(full_text)
    rows = []
    if table_m:
        for sno, product, price in ROW_RE.findall(table_m.group(1)):
            try:
                price_val = float(price.replace(',', ''))
            except ValueError:
                continue
            rows.append((product.strip(), price_val))

    error = None
    if urea_price is None and not rows:
        error = 'neither urea price nor mrp table found'
    return year, month, urea_price, rows, error


def run_probe():
    session = requests.Session()
    print('=' * 65)
    print('SCRIPT 57 PROBE -- Fertilizer MRP (fert.gov.in) coverage check')
    print('=' * 65)

    print('\n[1] Discovering bulletin URLs from both listing pages ...')
    urls = discover_bulletin_urls(session)
    print(f'  {len(urls)} English-titled bulletin URLs found')

    print('\n[2] Downloading + parsing 3 samples spanning the archive '
          '(oldest, middle, newest) ...')
    sample = [urls[-1], urls[len(urls) // 2], urls[0]]  # archive listed newest-first within each page
    all_ok = True
    for url in sample:
        path = fetch_pdf(url, session)
        year, month, urea, rows, error = parse_bulletin(path, url)
        if error:
            print(f'  FAIL  {os.path.basename(url)[:60]:60s} -- {error}')
            all_ok = False
            continue
        plausible_urea = urea is None or 3000 <= urea <= 15000
        plausible_rows = 10 <= len(rows) <= 100
        # Floor of 100 (not 1000): bulk organic-manure variants (LFOM-Packed
        # etc.) legitimately price far below mineral fertilizers -- confirmed
        # 2026-09-17 by reading the raw PDF text directly, not a parsing bug
        # (2026-07 bulletin genuinely reads "53 LFOM-Packed 500").
        plausible_prices = all(100 <= p <= 100000 for _, p in rows) if rows else True
        ok = plausible_urea and plausible_rows and plausible_prices
        all_ok &= ok
        print(f'  {"OK  " if ok else "WARN"}  {year}-{month:02d}  '
              f'urea={urea}  rows={len(rows)}  '
              f'sample_row={rows[0] if rows else None}')

    print('\n' + '=' * 65)
    if all_ok:
        print('Probe complete -- samples across the full date range parsed '
              'plausibly. Safe to run --mode full.')
    else:
        print('Probe complete -- one or more samples FAILED or looked '
              'implausible. Investigate before running --mode full.')


def run_full():
    session = requests.Session()
    print('=' * 65)
    print('SCRIPT 57 FULL PULL -- Fertilizer MRP, all discovered bulletins')
    print('=' * 65)

    print('\n[1] Discovering bulletin URLs ...')
    urls = discover_bulletin_urls(session)
    print(f'  {len(urls)} English-titled bulletin URLs found')

    print('\n[2] Downloading + parsing each bulletin ...')
    seen_months = {}  # (year, month) -> url of the copy we kept
    all_rows = []
    failures = []
    for i, url in enumerate(urls, 1):
        path = fetch_pdf(url, session)
        year, month, urea, rows, error = parse_bulletin(path, url)
        if error:
            failures.append((url, error))
            continue
        key = (year, month)
        if key in seen_months:
            continue  # duplicate month (Hindi/English pair, re-upload) -- keep first
        seen_months[key] = url
        if urea is not None:
            all_rows.append({'year': year, 'month': month, 'product': 'Urea',
                              'mrp_rs_per_mt': urea, 'source_url': url})
        for product, price in rows:
            all_rows.append({'year': year, 'month': month, 'product': product,
                              'mrp_rs_per_mt': price, 'source_url': url})
        if i % 20 == 0:
            print(f'  [{i}/{len(urls)}] processed, {len(seen_months)} distinct months so far')

    import pandas as pd
    out = pd.DataFrame(all_rows).sort_values(['year', 'month', 'product'])
    out_path = os.path.join(OUT_DIR, 'fert_mrp_monthly.csv')
    out.to_csv(out_path, index=False, encoding='utf-8')

    months_covered = sorted(seen_months.keys())
    print(f'\nSaved: {out_path}  ({len(out):,} rows, '
          f'{len(seen_months)} distinct months, '
          f'{months_covered[0][0]}-{months_covered[0][1]:02d} to '
          f'{months_covered[-1][0]}-{months_covered[-1][1]:02d})')
    if failures:
        print(f'\n{len(failures)} bulletin(s) failed to parse:')
        for url, error in failures:
            print(f'  {os.path.basename(url)[:70]:70s} -- {error}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['probe', 'full'], default='probe')
    args = parser.parse_args()

    if args.mode == 'probe':
        run_probe()
    else:
        run_full()
