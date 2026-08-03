# -*- coding: utf-8 -*-
"""
Script 10 — CMIE Economic Outlook Macro Series Parser
=======================================================
Parses 7 CMIE Excel downloads into a single monthly macro CSV.

Expected files in C:/Users/masro/Downloads/CMIE_Macro/:
  cmie_agri_credit.xlsx         Bank Credit to Agriculture & Allied Activities
  cmie_export_veg.xlsx          Export of Vegetables
  cmie_import_veg.xlsx          Import of Vegetables
  cmie_crude_oil.xlsx           Crude Oil Price - Indian Basket (USD/barrel)
  cmie_iip_food.xlsx            IIP - Food & Beverages (Base 2011-12=100)
  cmie_pfce_food.xlsx           PFCE - Food & Beverages
  cmie_agri_wages.xlsx          Agricultural / Rural Wages

Output: C:/Users/masro/Downloads/CMIE_Macro/cmie_macro_2017_2025.csv

Run:
  python 10_CMIE_Macro_Parser.py              # parse all found files
  python 10_CMIE_Macro_Parser.py --status     # show which files are present
"""

import io, os, sys, re
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

INDIR   = r'C:\Users\masro\Downloads\CMIE_Macro'
OUTFILE = os.path.join(INDIR, 'cmie_macro_2017_2025.csv')
os.makedirs(INDIR, exist_ok=True)

# Two-Phase Baseline, Stage 1 (2026-08-01): extended from 2017-01-01/
# 2025-12-31 -- each of these six CMIE files' own "From date"/"To date"
# metadata was checked directly before changing this (export_veg/
# import_veg/iip_food/agri_wages: From 2000-01-01; crude_oil (PPAC via
# CMIE): complete Apr 2000 - Mar 2026, every fiscal year, no gaps; only
# bank_credit_agri_cr has a later real floor, ~2006, a genuine per-series
# limit, not a parsing gap). All six also already carry real data through
# 2026 in the file, ahead of where the previous END constant stopped.
START, END = '2000-01-01', '2026-12-31'

# ----------------------------------------------------------------
# Series registry
# ----------------------------------------------------------------
SERIES = {
    'bank_credit_agri_cr': {
        'files':    ['cmie_agri_credit.xlsx', 'cmie_agri_credit.xls'],
        'desc':     'Bank Credit to Agriculture & Allied Activities (Rs Crore)',
        'val_kws':  ['agri', 'agriculture', 'allied', 'credit'],
        'date_kws': ['month', 'date', 'period', 'year'],
        'skip_kws': ['commercial', 'total', 'non-food'],   # reject broad aggregates
    },
    'export_veg_usd_mn': {
        'files':    ['cmie_export_veg.xlsx', 'cmie_export_veg.xls'],
        'desc':     'Export of Vegetables (USD million)',
        'val_kws':  ['vegetable', 'export', 'veg'],
        'date_kws': ['month', 'date', 'period', 'year'],
        'skip_kws': [],
    },
    'import_veg_usd_mn': {
        'files':    ['cmie_import_veg.xlsx', 'cmie_import_veg.xls'],
        'desc':     'Import of Vegetables (USD million)',
        'val_kws':  ['vegetable', 'import', 'veg'],
        'date_kws': ['month', 'date', 'period', 'year'],
        'skip_kws': [],
    },
    'crude_oil_usd_bbl': {
        'files':    ['cmie_crude_oil.xlsx', 'cmie_crude_oil.xls'],
        'desc':     'Crude Oil Price - Indian Basket (USD/barrel)',
        'val_kws':  ['crude', 'oil', 'indian basket', 'price', 'barrel'],
        'date_kws': ['month', 'date', 'period', 'year'],
        'skip_kws': [],
    },
    'iip_food_proc': {
        'files':    ['cmie_iip_food.xlsx', 'cmie_iip_food.xls'],
        'desc':     'IIP - Food & Beverages (Index, Base 2011-12=100)',
        'val_kws':  ['food', 'beverage', 'iip', 'index'],
        'date_kws': ['month', 'date', 'period', 'year'],
        'skip_kws': ['general', 'mining', 'electricity'],
    },
    'pfce_food_bev_cr': {
        'files':    ['cmie_pfce_food.xlsx', 'cmie_pfce_food.xls'],
        'desc':     'PFCE - Food & Beverages (Rs Crore)',
        'val_kws':  ['food', 'beverage', 'pfce', 'consumption', 'private'],
        'date_kws': ['month', 'date', 'period', 'year', 'quarter'],
        'skip_kws': [],
    },
    'agri_wages_rs_day': {
        'files':    ['cmie_agri_wages.xlsx', 'cmie_agri_wages.xls'],
        'desc':     'Agricultural / Rural Wages (Rs/day)',
        'val_kws':  ['wage', 'wages', 'agri', 'rural', 'farm'],
        'date_kws': ['month', 'date', 'period', 'year'],
        'skip_kws': [],
    },
}

# ----------------------------------------------------------------
# Excel reader
# ----------------------------------------------------------------

def read_excel_sheets(path):
    for engine in ('openpyxl', 'xlrd'):
        try:
            xl = pd.ExcelFile(path, engine=engine)
            return {s: xl.parse(s, header=None) for s in xl.sheet_names}
        except Exception:
            pass
    return None


def find_header_row(df_raw, keywords):
    for i, row in df_raw.iterrows():
        row_str = ' '.join(str(v) for v in row.values if pd.notna(v)).lower()
        if any(k in row_str for k in keywords):
            return i
    return 0


# ----------------------------------------------------------------
# CMIE-specific date parser
# CMIE often uses formats like "Apr 2021", "2021-04", "April 2021"
# ----------------------------------------------------------------

def parse_cmie_dates(series):
    # Try standard pandas first
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.notna().sum() > len(series) * 0.7:
        return parsed

    # Try "Mon YYYY" or "Month YYYY" format
    def try_parse(val):
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        for fmt in ('%b %Y', '%B %Y', '%Y-%m', '%m/%Y', '%Y/%m'):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        # Try numeric year + month columns merged as "YYYYMM"
        if re.match(r'^\d{6}$', s):
            try:
                return pd.to_datetime(s, format='%Y%m')
            except Exception:
                pass
        return pd.NaT

    return series.apply(try_parse)


# ----------------------------------------------------------------
# Generic CMIE sheet parser
# ----------------------------------------------------------------

def parse_cmie_sheet(df_raw, col_name, meta):
    date_kws = meta['date_kws']
    val_kws  = meta['val_kws']

    hrow = find_header_row(df_raw, val_kws + date_kws)
    df   = df_raw.iloc[hrow:].reset_index(drop=True)
    if df.empty:
        return None

    raw_header = [str(v).strip() for v in df.iloc[0].values]
    header     = [h.lower() for h in raw_header]
    df         = df.iloc[1:].copy()
    df.columns = header
    df         = df.dropna(how='all')

    print(f'    Header @row {hrow}: {raw_header[:6]}')

    # Locate date column
    date_col = None
    for c in df.columns:
        if any(k in str(c) for k in date_kws):
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            test = parse_cmie_dates(df[c].dropna().head(8))
            if test.notna().sum() >= 4:
                date_col = c
                break

    if date_col is None:
        print(f'    Cannot find date column.')
        return None

    # Locate value column (prefer matching val_kws, avoid skip_kws)
    val_col = None
    for c in df.columns:
        if c == date_col:
            continue
        c_str = str(c).lower()
        if any(k in c_str for k in meta.get('skip_kws', [])):
            continue
        if any(k in c_str for k in val_kws):
            numeric_count = pd.to_numeric(df[c], errors='coerce').notna().sum()
            if numeric_count > 5:
                val_col = c
                break

    # Fallback: first numeric column
    if val_col is None:
        for c in df.columns:
            if c == date_col:
                continue
            c_str = str(c).lower()
            if any(k in c_str for k in meta.get('skip_kws', [])):
                continue
            numeric_count = pd.to_numeric(df[c], errors='coerce').notna().sum()
            if numeric_count > 5:
                val_col = c
                break

    if val_col is None:
        print(f'    Cannot find value column.')
        return None

    print(f'    Date col: "{date_col}"  |  Value col: "{val_col}"')

    out = df[[date_col, val_col]].copy()
    out.columns = ['date_raw', col_name]
    out['date']    = parse_cmie_dates(out['date_raw'])
    out[col_name]  = pd.to_numeric(out[col_name], errors='coerce')
    out = out[['date', col_name]].dropna()

    if len(out) < 5:
        print(f'    Too few valid rows ({len(out)}).')
        return None

    out['date'] = out['date'].dt.to_period('M').dt.to_timestamp()
    out = (out.sort_values('date')
              .drop_duplicates('date')
              .reset_index(drop=True))

    print(f'    Parsed {len(out)} rows: {out["date"].min().date()} to {out["date"].max().date()}')
    return out


def parse_bank_credit(fpath, col_name):
    """Special parser for Bank Credit.xlsx (RBI Priority Sector format)."""
    df = pd.read_excel(fpath, header=None, engine='openpyxl')
    # Row 1: column names, Row 3 onward: data with YYYYMMDD dates
    headers = [str(v).strip().lower() if pd.notna(v) else '' for v in df.iloc[1]]
    data = df.iloc[3:].copy()
    data.columns = range(df.shape[1])
    data['date'] = pd.to_datetime(data[0].astype(str).str.strip(), format='%Y%m%d', errors='coerce')
    data = data.dropna(subset=['date'])
    # Col 3 = Agriculture and allied activities
    data[col_name] = pd.to_numeric(data[3], errors='coerce')
    out = data[['date', col_name]].dropna()
    out['date'] = out['date'].dt.to_period('M').dt.to_timestamp()
    out = out.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    out = out[(out['date'] >= START) & (out['date'] <= END)]
    print('    Bank Credit (Agriculture): ' + str(len(out)) + ' months, mean Rs ' +
          '{:.0f}'.format(out[col_name].mean()) + ' mn')
    return out if len(out) > 0 else None


def parse_crude_oil(fpath, col_name):
    """Special parser for PPAC crude oil file (wide financial-year format)."""
    df = pd.read_excel(fpath, header=None, engine='openpyxl')
    # Col 0 = FY label (e.g. '2017-18'), Cols 1-12 = Apr..Mar
    month_map = {1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:11, 9:12, 10:1, 11:2, 12:3}
    records = []
    for _, row in df.iloc[13:39].iterrows():
        fy = str(row[0]).strip()
        try:
            start_yr = int(fy.split('-')[0])
        except Exception:
            continue
        for col_idx in range(1, 13):
            month_num = month_map[col_idx]
            cal_year = start_yr if month_num >= 4 else start_yr + 1
            val = pd.to_numeric(row[col_idx], errors='coerce')
            if pd.notna(val):
                records.append({
                    'date': pd.Timestamp(year=cal_year, month=month_num, day=1),
                    col_name: round(float(val), 2)
                })
    out = pd.DataFrame(records).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    out = out[(out['date'] >= START) & (out['date'] <= END)]
    print('    Crude Oil (Indian Basket): ' + str(len(out)) + ' months, mean $' +
          '{:.2f}'.format(out[col_name].mean()) + '/bbl')
    return out if len(out) > 0 else None


def parse_one_series(col_name, meta):
    for fname in meta['files']:
        fpath = os.path.join(INDIR, fname)
        if not os.path.exists(fpath):
            continue
        print('\n  Parsing [' + col_name + '] from: ' + fname)

        # Special parsers for non-standard file formats
        if col_name == 'bank_credit_agri_cr':
            return parse_bank_credit(fpath, col_name)
        if col_name == 'crude_oil_usd_bbl':
            return parse_crude_oil(fpath, col_name)

        sheets = read_excel_sheets(fpath)
        if not sheets:
            print('    Cannot open file.')
            continue
        print('    Sheets: ' + str(list(sheets.keys())))
        for sname, df_raw in sheets.items():
            if df_raw.empty:
                continue
            result = parse_cmie_sheet(df_raw, col_name, meta)
            if result is not None and len(result) > 0:
                filtered = result[
                    (result['date'] >= START) & (result['date'] <= END)
                ].reset_index(drop=True)
                if len(filtered) > 0:
                    print('    OK: ' + str(len(filtered)) + ' months (2017-2025)')
                    return filtered
    return None


# ----------------------------------------------------------------
# Merge & save
# ----------------------------------------------------------------

def merge_and_save(results):
    if not results:
        print('\nNo series parsed.')
        return

    dfs = list(results.values())
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='date', how='outer')

    merged = merged.sort_values('date').reset_index(drop=True)
    merged['year']  = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month

    base = ['date', 'year', 'month']
    data = [c for c in merged.columns if c not in base]
    merged = merged[base + data]

    merged.to_csv(OUTFILE, index=False)
    print(f'\n  Saved: {OUTFILE}  ({len(merged)} rows)')
    print(f'  Columns: {list(merged.columns)}')
    print('\n  === Completeness ===')
    for col in data:
        n = merged[col].notna().sum()
        pct = 100 * n / len(merged)
        flag = '  OK' if pct >= 90 else '  WARN: gaps'
        print(f'  {col:<28}: {n:>3}/{len(merged)} months  ({pct:.0f}%){flag}')


# ----------------------------------------------------------------
# Status check
# ----------------------------------------------------------------

def print_status():
    print('\n=== CMIE File Status ===')
    print(f'Directory: {INDIR}\n')
    for col_name, meta in SERIES.items():
        found = [f for f in meta['files'] if os.path.exists(os.path.join(INDIR, f))]
        status = f'FOUND: {found[0]}' if found else 'MISSING'
        print(f'  {col_name:<28}: {status}')
        print(f'    {meta["desc"]}')
    print()


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

if __name__ == '__main__':
    if '--status' in sys.argv:
        print_status()
        sys.exit(0)

    print('=' * 60)
    print('CMIE Macro Series Parser')
    print('=' * 60)
    print_status()

    results = {}
    for col_name, meta in SERIES.items():
        df = parse_one_series(col_name, meta)
        if df is not None:
            results[col_name] = df
        else:
            found = [f for f in meta['files'] if os.path.exists(os.path.join(INDIR, f))]
            if not found:
                print(f'\n  [{col_name}]: file not in {INDIR} -- download and save as {meta["files"][0]}')
            else:
                print(f'\n  [{col_name}]: file found but could not parse -- check sheet structure')

    print(f'\n  Parsed {len(results)}/{len(SERIES)} series successfully.')
    merge_and_save(results)
