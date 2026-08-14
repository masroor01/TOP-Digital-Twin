"""
Script 09e -- Merge Tomato/Onion "_new" 2026 Uploads Into the Main Raw Files
=============================================================================
Root cause of the "stale dashboard data despite uploading new files" report
(2026-08-13): `tomato_all_india_apmcs_2026_new.csv` and
`onion_all_india_apmcs_2026_new.csv` were uploaded for the live-forecast-
validation exercise (Scripts 43/45) and only ever wired into those scripts
-- Script 09 (which actually builds the panel the dashboard reads) still
reads the separate MAIN files (`*_all_india_apmcs_2000_2026.csv`), which
were never updated. The "_new" files extend to 2026-08-12; the main files
stopped at 2026-07-27 (tomato) / 2026-07-25 (onion) -- about 2.5 weeks of
real data was sitting unused.

Verified before merging (same rigor as the original onion rebuild):
  tomato overlap match rate: 99.92% on (market_id, arrival_date, variety)
  onion  overlap match rate: 98.24%
Both consistent with prior small merge-order artifacts found elsewhere in
this project (duplicate keys re-ordering on a many-to-many join), not a
real data conflict -- same AGMARKNET 2.0 source, same schema, in both.

Strategy: pure APPEND, not a full re-merge. Keep the main file's already-
verified historical rows completely untouched; append only the "_new"
file's rows strictly AFTER the main file's own current max date. This
avoids re-litigating already-processed history over a <2% artifact rate.

Potato has no "_new" file and is not touched by this script.

Run BEFORE Script 09 (bump END_DATE in Script 09 to match the new max
date first). Overwrites the main files in Downloads/ in place -- back up
if you want to keep the pre-merge version.
"""
import os
import sys
import shutil
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DOWNLOADS = r'C:\Users\masro\Downloads'

MERGES = [
    {
        'crop': 'tomato',
        'main': os.path.join(DOWNLOADS, 'tomato_all_india_apmcs_2000_2026.csv'),
        'new':  os.path.join(DOWNLOADS, 'tomato_all_india_apmcs_2026_new.csv'),
    },
    {
        'crop': 'onion',
        'main': os.path.join(DOWNLOADS, 'onion_all_india_apmcs_2000_2026.csv'),
        'new':  os.path.join(DOWNLOADS, 'onion_all_india_apmcs_2026_new.csv'),
    },
]

print('=================================================================')
print('SCRIPT 09e: MERGE TOMATO/ONION "_NEW" 2026 UPLOADS')
print('=================================================================')

for m in MERGES:
    crop, main_path, new_path = m['crop'], m['main'], m['new']
    print(f'\n[{crop}]')

    main = pd.read_csv(main_path)
    new = pd.read_csv(new_path)
    assert list(main.columns) == list(new.columns), \
        f'{crop}: column mismatch between main and new files -- schema drift, do not merge blindly'

    main['arrival_date'] = pd.to_datetime(main['arrival_date'])
    new['arrival_date'] = pd.to_datetime(new['arrival_date'])

    main_max = main['arrival_date'].max()
    new_max = new['arrival_date'].max()
    print(f'  Main file : {len(main):,} rows, max date {main_max.date()}')
    print(f'  New file  : {len(new):,} rows, max date {new_max.date()}')

    if new_max <= main_max:
        print(f'  New file has no dates beyond the main file -- nothing to append. Skipping.')
        continue

    to_append = new[new['arrival_date'] > main_max].copy()
    print(f'  Appending {len(to_append):,} rows strictly after {main_max.date()} '
          f'(new coverage: {to_append["arrival_date"].min().date()} to {new_max.date()})')

    # Restore the original string date format so the merged file matches
    # what Script 09 expects (it re-parses arrival_date itself).
    to_append['arrival_date'] = to_append['arrival_date'].dt.strftime('%Y-%m-%d')
    main['arrival_date'] = main['arrival_date'].dt.strftime('%Y-%m-%d')

    backup_path = main_path + '.pre_09e_backup'
    if not os.path.exists(backup_path):
        shutil.copy(main_path, backup_path)
        print(f'  Backed up original main file to: {backup_path}')

    merged = pd.concat([main, to_append], ignore_index=True)
    merged.to_csv(main_path, index=False)
    print(f'  Saved: {main_path}  ({len(merged):,} rows, max date now '
          f'{pd.to_datetime(merged["arrival_date"]).max().date()})')

print('\nScript 09e complete.')
print('Next: bump END_DATE in Script 09 to the new max date, then re-run')
print('  python scripts/09_Agmarknet_Weekly_Panel.py')
print('  python scripts/23_Train_Production_Models.py')
