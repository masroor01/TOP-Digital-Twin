# -*- coding: utf-8 -*-
"""
Script 58 — Input Cost / Wage Layer: CPI for Agricultural & Rural Labourers
==============================================================
Acquires the Consumer Price Index for Agricultural Labourers (CPI-AL) and
Rural Labourers (CPI-RL), state-wise, from the Labour Bureau's monthly
press-release PDFs (https://labourbureau.gov.in) -- Base 2019=100, 34
States/UTs + All-India, published ~20 days after each reference month.

WHY THIS SERIES, NOT THE OLDER ONE THE USER PULLED FROM RBI'S DBIE:
  Labour Bureau rebased CPI-AL/RL from 1986-87=100 (20 states) to 2019=100
  (34 states/UTs) effective June 2025 -- expanded village sample (787 vs
  600), Geometric Mean instead of Arithmetic Mean, COICOP-2018 basket. The
  RBI DBIE mirror the user downloaded (2026-09-17 session) only carries the
  OLD series and stops dead at May 2025, the last month before the rebase --
  that is not staleness, it is the literal end of that series. This script
  acquires the NEW (2019=100) series directly from the primary source, which
  is confirmed current (August 2026 already published as of 2026-09-21).

WHY THIS IS STILL NOT ITEM-LEVEL CPI:
  This remains a state-level General Index (with a Food sub-group also
  published, not captured here) -- a wage/input-cost proxy, not a
  tomato/onion/potato price series. That granularity was never going to
  exist here; it only exists in MOSPI's item-level API, which has its own
  confirmed live bug (see MANIFEST.md / prior session).

HOW THE ONE-TIME HISTORICAL BACKFILL WAS BUILT (2026-09-21 session):
  labourbureau.gov.in has NO archive/listing page for past releases of the
  new series -- confirmed by direct inspection: "Reports of CPI (AL/RL)" and
  "State wise General Index" (the site's own archive pages) are stale, still
  only listing the OLD-base annual PDFs through 2020-21/2023. The site's own
  "Press Release" notice board (newscontent/pr) only ever shows the CURRENT
  month, with no pagination.
    Workaround: the Wayback Machine has captured the labourbureau.gov.in
  homepage roughly monthly. Each captured snapshot's HTML still names
  whatever press-release PDF was "current" at that time (exact hashed
  filename included). Fetching that filename directly against the LIVE site
  (not the Wayback copy) works -- confirmed 2026-09-21 that every one of the
  10 URLs below returns HTTP 200 today, even the January 2026 one: Labour
  Bureau does not delete old press-release PDFs, it just stops linking them.
    Each PDF's state-wise table covers the reference month AND the one
  before it (2 columns per series), so ~10 releases backfill the entire
  June 2025 (rebase point) -> present range. Cross-validated two of these
  against independent numbers: the Nov-2025 PDF's "October" column
  (All-India AL=136.40) matches a secondary source found independently; the
  Jan-2026 PDF's "December" column (All-India AL=137.12) matches the
  official PIB press release fetched directly. One gap remains: MAY 2026 --
  no Wayback snapshot or web search turned up that release as of 2026-09-21.
  --mode full also tries the site's live "latest" listing each run, so the
  backfill list only needs to be extended if a NEW multi-month gap opens up
  (e.g. this script isn't run for several months).

Output:
  data/cpi_al_rl/cpi_alrl_statewise_monthly.csv
    columns: state, series (AL/RL), year, month, index_value
  data/cpi_al_rl/raw_cache/*.pdf (cached so a re-run doesn't re-download)

Run:
  python scripts/58_CPI_AL_RL_Layer.py --mode probe
  python scripts/58_CPI_AL_RL_Layer.py --mode full
"""

import os
import re
import argparse

import requests
import urllib3
import pdfplumber
import pandas as pd

# labourbureau.gov.in's certificate chain fails Python/OpenSSL verification
# (self-signed intermediate) even though it's fine over curl/Windows schannel
# and in an ordinary browser -- confirmed 2026-09-21, a server-side chain
# quirk on this specific .gov.in host, not a MITM concern (content matches
# curl/browser fetches byte-for-byte). Verification is disabled only for
# requests to this known, already-manually-validated domain.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'data', 'cpi_al_rl')
CACHE_DIR = os.path.join(OUT_DIR, 'raw_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

LATEST_LISTING_URL = 'https://labourbureau.gov.in/newscontent/pr'
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {'User-Agent': 'Mozilla/5.0'}

MONTH_NUM = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
}

# Canonical row order of every CPI-AL/RL (Base 2019=100) state-wise press-release
# table -- confirmed identical across Nov-2025, Jan-2026 and Aug-2026 samples.
# Parsing below does NOT regex-match these names (multi-word names wrap across
# PDF lines unpredictably -- e.g. "Daman and Diu, Dadra Nagar and Haveli" and
# "Andaman and Nicobar Islands" split differently release to release). Instead
# it extracts the flat sequence of numbers from the table body and zips them
# against this fixed, known-good order -- numbers don't wrap ambiguously the
# way multi-word names do.
STATE_ORDER = [
    'Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam',
    'Bihar', 'Chhattisgarh', 'Daman and Diu, Dadra Nagar and Haveli', 'Delhi',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir',
    'Jharkhand', 'Karnataka', 'Kerala', 'Lakshadweep', 'Madhya Pradesh',
    'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
    'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
    'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'All India',
]
# Delhi's AL columns are genuinely "0.00" in every release (CPI-AL isn't
# compiled for Delhi -- no agricultural-labour population) -- confirmed by
# reading the raw PDF text directly, not a parsing artifact.

# One-time historical backfill -- see docstring for how these were found.
# label -> live URL, each covering (label's month - 1) and (label's month).
BACKFILL_RELEASES = [
    # Note: the actual June-2025 press release is the base-revision ANNOUNCEMENT
    # (no state-wise table at all, only a single All-India value) -- confirmed by
    # reading it directly. June's state-wise figures come from Jul-2025 below
    # (col 1 of its table), which is why June has no entry of its own here.
    ('Jul-2025', 'https://labourbureau.gov.in/uploads/public/notice/English-Press-Release-for-CPI---AL-RL-Base-2019-100-July-25pdf-3418bad92953c1a5bf2d82828841cb8d.pdf'),
    ('Aug-2025', 'https://www.labourbureau.gov.in/uploads/public/notice/English-Press-Release-for-CPI---AL--RL-Base-2019-100-August-25pdf-bcdbffb435cfe6621baad890ec2d0fd5.pdf'),
    ('Sep-2025', 'https://labourbureau.gov.in/uploads/public/notice/English-Press-Release-for-CPI---AL-RL-Base-2019-100-September-25pdf-c3e9a3bae5dbc586762060335d672c99.pdf'),
    ('Nov-2025', 'https://www.labourbureau.gov.in/uploads/public/notice/Press-Release-English-Nov-2025pdf-ef5ba3d72d078a4fadf6865323adb916.pdf'),
    ('Dec-2025', 'https://www.labourbureau.gov.in/uploads/public/notice/Press-Release-Eng-Dec-2025pdf-6ce0a6e1cd0a7f5ee3102951b1680e9b.pdf'),
    ('Jan-2026', 'https://labourbureau.gov.in/uploads/public/notice/EnglishPressReleaseCPIALRLJanuary-26pdf-4b3c1d4b9cc26b9ecaa398ac94b98fb2.pdf'),
    ('Mar-2026', 'https://labourbureau.gov.in/uploads/public/notice/English-Press-Release-for-CPI---AL--RL-Base-2019-100-March-2026pdf-751f3b198011fe5aa9bdbd0ad24a66df.pdf'),
    ('Apr-2026', 'https://www.labourbureau.gov.in/uploads/public/notice/English-Press-Release-for-CPI---AL--RL-Base-2019-100-April-2026pdf-0445fea46da460393c7488f9ee621ae6.pdf'),
    ('Jul-2026', 'https://www.labourbureau.gov.in/uploads/public/notice/English-Press-Release-for-CPI---AL-RL-Base-2019-100-July-26pdf-41330b9bce12f952887a3f4b3ca243b8.pdf'),
]
# Known gap: May-2026 -- no release discoverable via Wayback snapshots or web
# search as of 2026-09-21. Left out rather than interpolated/faked.

TABLE_ANCHOR_RE = re.compile(r'All India/States')
NUMBER_RE = re.compile(r'-?\d+\.\d{2}')


def label_to_month_pair(label):
    """'Jul-2025' -> ((6, 2025), (7, 2025)) -- (prior month, release month).
    Deriving the two covered months from the release's own label, rather
    than parsing the header row's month names, sidesteps a real PDF
    rendering quirk: in some releases (e.g. Jul-2025) pdfplumber extracts
    the header text in "June June July July" order -- visually merged
    header cells throwing off left-to-right text order -- even though the
    DATA columns underneath are reliably (AL-prior, AL-current, RL-prior,
    RL-current), confirmed by cross-checking known All-India values."""
    month_abbr, year_s = label.split('-')
    month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month2 = month_names.index(month_abbr.lower()) + 1
    year2 = int(year_s)
    if month2 == 1:
        month1, year1 = 12, year2 - 1
    else:
        month1, year1 = month2 - 1, year2
    return (month1, year1), (month2, year2)


def discover_current_release(session):
    """Returns (label, url) for whatever the site's notice board currently
    lists as the latest CPI-AL/RL press note, or None if not found. Doesn't
    assume href/title attribute order (differs between live site and the
    Wayback-archived copies used to build BACKFILL_RELEASES) -- searches a
    window around the matched title text instead of a single combined regex.
    """
    resp = session.get(LATEST_LISTING_URL, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS, verify=False)
    resp.raise_for_status()
    idx = resp.text.find('Press Note CPI- AL/RL for')
    if idx == -1:
        return None
    window = resp.text[max(0, idx - 400): idx + 400]
    title_m = re.search(r'Press Note CPI- AL/RL for ([A-Za-z]+) (\d{4})', window)
    href_m = re.search(r'href="([^"]+\.pdf)"', window)
    if not (title_m and href_m):
        return None
    month_name, year = title_m.groups()
    return (f'{month_name[:3]}-{year}', href_m.group(1))


def fetch_pdf(label, url, session):
    fname = re.sub(r'[^A-Za-z0-9_.-]', '_', label) + '.pdf'
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        return path
    resp = session.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS, verify=False)
    resp.raise_for_status()
    with open(path, 'wb') as f:
        f.write(resp.content)
    return path


def parse_release(label, path):
    """Returns (year1, month1, year2, month2, rows, error). rows is a list of
    (state, series, year, month, index_value, is_current) tuples for BOTH
    months this PDF's state-wise table covers -- (month1, year1) is the
    prior-month reference column, (month2, year2) is this release's own
    ("current") month, derived from `label` (see label_to_month_pair).
    is_current is True only for the (month2, year2) rows.

    Why is_current matters: cross-checking overlapping releases (2026-09-21
    session) found that a prior-month reference column can genuinely
    disagree with that same month's OWN current-month column in its own
    release -- confirmed via pdfplumber's structured extract_tables(), not a
    parsing artifact, e.g. the Dec-2025 release's "November" column doesn't
    match the Nov-2025 release's own November column for several states
    (looks like an off-by-one regenerated-reference-column error on Labour
    Bureau's end). The current-month column is that month's primary,
    first-published figure, so it's treated as authoritative over any later
    release's repeated reference to it -- see resolve() in run_full."""
    (month1, year1), (month2, year2) = label_to_month_pair(label)

    try:
        with pdfplumber.open(path) as pdf:
            full_text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    except Exception as e:
        return None, None, None, None, [], f'pdf_open_error: {e}'

    anchor_m = TABLE_ANCHOR_RE.search(full_text)
    if not anchor_m:
        return year1, month1, year2, month2, [], 'table_anchor_not_found'

    body_start = anchor_m.end()
    end_m = re.search(r'\*\*\*', full_text[body_start:])
    body = full_text[body_start: body_start + end_m.start()] if end_m else full_text[body_start:]

    numbers = [float(n) for n in NUMBER_RE.findall(body)]
    expected = len(STATE_ORDER) * 4
    if len(numbers) != expected:
        return (year1, month1, year2, month2, [],
                f'expected {expected} numbers (35 states x 4 cols), found {len(numbers)}')

    rows = []
    for i, state in enumerate(STATE_ORDER):
        al_m1, al_m2, rl_m1, rl_m2 = numbers[i * 4: i * 4 + 4]
        rows.append((state, 'AL', year1, month1, al_m1, False))
        rows.append((state, 'AL', year2, month2, al_m2, True))
        rows.append((state, 'RL', year1, month1, rl_m1, False))
        rows.append((state, 'RL', year2, month2, rl_m2, True))
    return year1, month1, year2, month2, rows, None


def run_probe():
    session = requests.Session()
    print('=' * 65)
    print('SCRIPT 58 PROBE -- CPI-AL/RL (labourbureau.gov.in) coverage check')
    print('=' * 65)

    print('\n[1] Discovering the current live release ...')
    current = discover_current_release(session)
    print(f'  {current}' if current else '  NOT FOUND -- notice-board page format may have changed')

    sample = [BACKFILL_RELEASES[0], BACKFILL_RELEASES[len(BACKFILL_RELEASES) // 2]]
    if current:
        sample.append(current)
    print(f'\n[2] Downloading + parsing {len(sample)} samples '
          '(oldest backfill, middle backfill, current live) ...')
    all_ok = True
    for label, url in sample:
        path = fetch_pdf(label, url, session)
        y1, m1, y2, m2, rows, error = parse_release(label, path)
        if error:
            print(f'  FAIL  {label:10s} -- {error}')
            all_ok = False
            continue
        all_india_al = [r[4] for r in rows if r[0] == 'All India' and r[1] == 'AL']
        plausible = all(100 <= v <= 250 for v in all_india_al)
        all_ok &= plausible
        print(f'  {"OK  " if plausible else "WARN"}  {label:10s}  '
              f'covers {m1:02d}/{y1} + {m2:02d}/{y2}  '
              f'All-India AL={all_india_al}')

    print('\n' + '=' * 65)
    if all_ok:
        print('Probe complete -- samples parsed plausibly. Safe to run --mode full.')
    else:
        print('Probe complete -- one or more samples FAILED or looked implausible. '
              'Investigate before running --mode full.')


def run_full():
    session = requests.Session()
    print('=' * 65)
    print('SCRIPT 58 FULL PULL -- CPI-AL/RL, backfill + current')
    print('=' * 65)

    releases = list(BACKFILL_RELEASES)
    current = discover_current_release(session)
    if current and not any(u == current[1] for _, u in releases):
        releases.append(current)
        print(f'\n[1] Added current live release: {current[0]}')
    else:
        print('\n[1] Current live release already covered by backfill list, or not found.')

    print(f'\n[2] Downloading + parsing {len(releases)} releases ...')
    all_rows = []
    failures = []
    for label, url in releases:
        path = fetch_pdf(label, url, session)
        y1, m1, y2, m2, rows, error = parse_release(label, path)
        if error:
            failures.append((label, error))
            print(f'  FAIL  {label:10s} -- {error}')
            continue
        all_rows.extend(rows)
        print(f'  OK    {label:10s}  covers {m1:02d}/{y1} + {m2:02d}/{y2}')

    df = pd.DataFrame(all_rows, columns=['state', 'series', 'year', 'month', 'index_value', 'is_current'])

    print('\n[3] Cross-validating overlapping months across releases and de-duplicating ...')
    revised = []
    conflicts = []

    def resolve(group):
        vals = group['index_value'].unique()
        if len(vals) == 1:
            return group['index_value'].iloc[0]
        current_vals = group.loc[group['is_current'], 'index_value'].unique()
        if len(current_vals) == 1:
            # A prior-month reference column disagreed with this month's own
            # current-month column elsewhere -- trust the current-month one
            # (see parse_release docstring); this is an expected, resolved
            # disagreement, not a data-quality failure.
            revised.append((group.name, vals.tolist(), current_vals[0]))
            return current_vals[0]
        # More than one release both treats this month as "current" and they
        # disagree -- that should not happen with a well-formed release list.
        conflicts.append((group.name, vals.tolist()))
        return group['index_value'].iloc[0]

    dedup = (df.groupby(['state', 'series', 'year', 'month'])
               .apply(resolve, include_groups=False)
               .reset_index(name='index_value')
               .sort_values(['year', 'month', 'state', 'series']))

    if revised:
        print(f'  {len(revised)} cell(s) had a prior-month reference column that disagreed '
              'with that month\'s own current-month figure -- resolved in favor of the '
              'current-month (primary) source:')
        for key, vals, chosen in revised[:5]:
            print(f'    {key} -- saw {vals}, kept {chosen}')
        if len(revised) > 5:
            print(f'    ... and {len(revised) - 5} more')
    if conflicts:
        print(f'  {len(conflicts)} UNRESOLVED conflicting cell(s) '
              '(disagreement between two current-month sources -- investigate):')
        for key, vals in conflicts[:10]:
            print(f'    {key} -- {vals}')
    if not revised and not conflicts:
        print('  No disagreements -- every cell covered by more than one release agreed exactly.')

    out_path = os.path.join(OUT_DIR, 'cpi_alrl_statewise_monthly.csv')
    dedup.to_csv(out_path, index=False, encoding='utf-8')

    months_covered = sorted(dedup[['year', 'month']].drop_duplicates().itertuples(index=False))
    print(f'\nSaved: {out_path}  ({len(dedup):,} rows, '
          f'{dedup["state"].nunique()} states/UTs, '
          f'{months_covered[0].month:02d}/{months_covered[0].year} -> '
          f'{months_covered[-1].month:02d}/{months_covered[-1].year}, '
          f'{len(months_covered)} distinct months)')
    all_months = {(y, m) for y in range(2025, 2027) for m in range(1, 13)}
    expected_range = {(y, m) for (y, m) in all_months
                       if (y, m) >= (2025, 6) and (y, m) <= (months_covered[-1].year, months_covered[-1].month)}
    missing = sorted(expected_range - {(y, m) for y, m in months_covered})
    if missing:
        print(f'Known gap(s) within the covered range: {missing}')
    if failures:
        print(f'\n{len(failures)} release(s) failed to parse:')
        for label, error in failures:
            print(f'  {label:10s} -- {error}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['probe', 'full'], default='probe')
    args = parser.parse_args()

    if args.mode == 'probe':
        run_probe()
    else:
        run_full()
