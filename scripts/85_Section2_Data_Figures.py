# -*- coding: utf-8 -*-
"""
Script 85 -- additional figures for manuscript Section 2 ("Study area and data"), all drawn from the current files (no hand-typed counts):
  fig_ms_data_architecture.png     data architecture: layers -> weekly panel -> models -> dashboard, and the weekly refresh path
  fig_ms_completeness.png          (a) per-market share of observed weeks, (b) market-weeks by status per year
  fig_ms_layer_coverage.png        share of weeks with a value, by weather/satellite source, crop and year
  fig_ms_zone_distance.png         distance from each market to its nearest production zone, and markets per zone
  fig_ms_weather_series.png        crop-level weekly weather and vegetation series used by the models
  fig_ms_satellite_validation.png  MODIS vs ERA5 temperature and MODIS vs Sentinel-2 NDVI, zone-week level
  fig_ms_climate_correlation.png   correlation among the weather and satellite variables, by crop
Run Scripts 78 and 84 first (they write table_market_locations.csv and refresh the panels' numbers).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from section2_common import load_panel, layer_latest, BASE, CROPS

OUT = os.path.join(BASE, 'Model_Output')
COL = {'tomato': '#DC2626', 'onion': '#7C3AED', 'potato': '#B45309'}
INK, MUTED, GRID = '#1F2933', '#52606D', '#E1E0D9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
LAST = pd.Timestamp('2026-10-26')


def rain_scale_ok(cw):
    """CHIRPS yearly totals (2019-2025) within 0.4-2.5x of each crop's median year. False while the 2000-2024 history and the 2025-26 top-ups are on
    different scales (see scripts/gee/gee_02_CHIRPS_history_2000_2024.js)."""
    t = cw[cw.week_start.dt.year.between(2019, 2025)].groupby([cw.crop, cw.week_start.dt.year]).chirps_rain_mm.sum().unstack()
    r = t.div(t.median(axis=1), axis=0)
    return bool(((r > 0.4) & (r < 2.5)).all().all())


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor='white')
    plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------- data architecture
def fig_architecture():
    lat = layer_latest()
    fmt = lambda k: lat[k].strftime('%b %Y') if lat[k].day > 28 or k not in ('Market prices and arrivals (Agmarknet)',) else lat[k].strftime('%d %b %Y')
    d = lambda k: lat[k].strftime('%d %b %Y')
    m = lambda k: lat[k].strftime('%b %Y')
    L = [  # label, sources, cadence/refresh, latest, fill, in production
        ('Market prices and arrivals', 'Agmarknet: daily modal price and arrivals per APMC market', 'weekly | portal reports, loaded by the weekly workflow', 'to ' + d('Market prices and arrivals (Agmarknet)'), '#DCE9F7', True),
        ('Weather', 'ERA5-Land temperature, CHIRPS rainfall via Earth Engine (17 zones)', 'weekly | automated', 'to ' + d('ERA5-Land temperature') + ' (CHIRPS ' + d('CHIRPS rainfall') + ')', '#DDEFD9', True),
        ('Satellite vegetation', 'Sentinel-2 NDVI/EVI, MODIS vegetation and land temperature (17 zones)', 'weekly | automated', 'to ' + d('Sentinel-2 vegetation') + ' (MODIS ' + d('MODIS vegetation and land temperature') + ')', '#E3F1D2', True),
        ('Macro-economic', 'CMIE, RBI DBIE, PPAC: credit, trade, WPI, wages, fuel, rates', 'monthly | manual', 'to ' + m('Macro (CMIE: credit, trade, WPI, wages, crude, IIP, USD/INR)') + ' (fuel ' + m('Fuel prices (PPAC)') + ')', '#FBE9D0', True),
        ('Infrastructure', 'Labour Bureau wages, cold storage, road density (state level)', 'quarterly to annual | manual', 'wages to ' + m('State agricultural wages (Labour Bureau)'), '#EADCF0', True),
        ('Policy and trade', 'Verified event log (PIB, DGFT): export ban, duty, MEP, schemes', 'event-driven | manual', 'to ' + d('Policy and trade log'), '#F7DADA', True),
        ('Candidate layers', 'Drought (VEDAS, IDM), fertiliser and CPI-AL/RL, ONI (NOAA)', 'monthly to seasonal | manual', 'tested, not in production', '#EEEEEE', False),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(-3.5, 60); ax.axis('off'); fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)

    def box(x, y, w, h, fc, ec='#9AA5B1', ls='-', lw=1.0, r=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, ls=ls, lw=lw))

    ax.text(1, 59.0, 'Data layers (source, cadence, refresh, latest data point)', fontsize=10, fontweight='bold', color=INK)
    y0, hh, gap = 7.0, 6.35, 1.0
    ys = []
    for i, (name, src, cad, latest, fc, prod) in enumerate(L):
        y = 52.4 - i * (hh + gap)
        ys.append(y)
        box(1, y, 44, hh, fc, ls='-' if prod else (0, (4, 3)))
        ax.text(2, y + hh - 1.35, name, fontsize=9.0, fontweight='bold', color=INK, va='center')
        ax.text(2, y + hh - 2.85, src, fontsize=7.1, color=INK, va='center')
        ax.text(2, y + hh - 4.1, cad, fontsize=6.8, color=MUTED, va='center')
        ax.text(2, y + 0.55, latest, fontsize=6.8, color=MUTED, va='center', style='italic')
    # weekly panel
    box(55, 17, 17, 28, '#FFF4C2', ec='#C9A227', lw=1.6, r=1.4)
    ax.text(63.5, 40.5, 'Weekly panel', fontsize=10, fontweight='bold', color=INK, ha='center')
    ax.text(63.5, 36.8, 'market x crop x ISO week', fontsize=7.8, color=INK, ha='center')
    ax.text(63.5, 31.5, 'Layers joined on crop and\nweek (weather, macro) or\nstate (infrastructure).\nValues missing at the\nforecast origin are carried\nforward; their age is stored.', fontsize=7.0, color=INK, ha='center', va='center', linespacing=1.25)
    ax.text(63.5, 21.0, 'Only information dated\non or before the origin', fontsize=7.0, color=MUTED, ha='center', style='italic')
    for (name, src, cad, latest, fc, prod), y in zip(L, ys):
        if prod:
            ax.add_patch(FancyArrowPatch((45.3, y + hh / 2), (54.7, 17 + 28 * (0.15 + 0.7 * (y - 8) / 50)), arrowstyle='-|>', mutation_scale=9, color='#7B8794', lw=0.9))
        else:
            ax.add_patch(FancyArrowPatch((45.3, y + hh / 2), (54.7, 20), arrowstyle='-|>', mutation_scale=9, color='#B0B8C1', lw=0.9, ls=(0, (3, 3))))
    # models, outputs
    box(80, 38, 19, 12, '#E4ECF4', r=1.2)
    ax.text(89.5, 46.5, 'Forecast models', fontsize=9.5, fontweight='bold', color=INK, ha='center')
    ax.text(89.5, 42.4, 'LightGBM per crop x horizon\n(1, 4, 13, 26 weeks)\nprice-only and six-layer variants', fontsize=7.0, color=INK, ha='center', va='center', linespacing=1.3)
    box(80, 20, 19, 13, '#E4ECF4', r=1.2)
    ax.text(89.5, 29.8, 'Evaluation and web twin', fontsize=9.5, fontweight='bold', color=INK, ha='center')
    ax.text(89.5, 24.6, 'Rolling-origin backtest,\nbaseline and scenario\nforecasts, daily curve,\ndashboard', fontsize=7.0, color=INK, ha='center', va='center', linespacing=1.3)
    ax.add_patch(FancyArrowPatch((72.3, 33), (79.7, 43), arrowstyle='-|>', mutation_scale=10, color='#7B8794', lw=1.1))
    ax.add_patch(FancyArrowPatch((89.5, 37.7), (89.5, 33.3), arrowstyle='-|>', mutation_scale=10, color='#7B8794', lw=1.1))
    # weekly refresh
    box(50, 1.0, 49, 11.5, '#F5F7FA', ec='#CBD2D9', ls=(0, (4, 3)), r=1.2)
    ax.text(51.2, 10.6, 'Weekly refresh (GitHub Actions)', fontsize=8.8, fontweight='bold', color=INK)
    ax.text(51.2, 6.6, 'Price reports and raw files pulled from Drive; Earth Engine top-up for the weather layer;\npanel rebuilt, coverage checked, models retrained; pull request reviewed by a person,\nthen the site is redeployed.', fontsize=7.0, color=INK, va='center', linespacing=1.35)
    ax.text(1, -2.2, 'Dashed boxes and arrows: layers assembled and tested but not part of the production models. Dates are read from the data files.', fontsize=7.0, color=MUTED)
    save(fig, 'fig_ms_data_architecture.png')


# ---------------------------------------------------------------- completeness
def fig_completeness(p):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1, 1.1]})
    ax = axs[0]
    bins = np.arange(0.5, 1.0001, 0.05)
    for c in CROPS:
        d = p[p.crop == c]
        sh = (d.imputed == 0).groupby(d.market_id).mean()
        ax.hist(sh, bins=bins, histtype='stepfilled', alpha=0.28, color=COL[c], lw=0)
        ax.hist(sh, bins=bins, histtype='step', color=COL[c], lw=1.6, label='%s (%d markets, median %.0f %%)' % (c.capitalize(), len(sh), 100 * sh.median()))
    ax.axvline(0.70, color=MUTED, lw=1, ls=':')
    ax.text(0.703, ax.get_ylim()[1] * 0.97, 'entry rule: at least 70 % observed', fontsize=7.5, color=MUTED, va='top')
    ax.set_xlabel('Share of a market\'s weeks with an observed price'); ax.set_ylabel('Number of markets')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 0.88))
    ax.set_title('(a) Observed share per market', loc='left', fontsize=10, fontweight='bold', color=INK)
    ax = axs[1]
    p2 = p.copy()
    p2['status'] = np.where(p2.imputed == 0, 'Observed', p2['imputed_method'].map({'linear': 'Interpolated', 'trailing_ffill': 'Carried forward', 'seasonal_median': 'Seasonal median'}).fillna('Missing'))
    order = ['Observed', 'Interpolated', 'Seasonal median', 'Carried forward', 'Missing']
    colors = ['#2a78d6', '#7fb3ea', '#f0a35e', '#9aa5b1', '#d64545']
    t = p2.groupby([p2.week_start.dt.year, 'status']).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    t = t.div(t.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(t))
    for s, cl in zip(order, colors):
        ax.bar(t.index, t[s], bottom=bottom, color=cl, width=0.75, label=s)
        bottom += t[s].values
    ax.set_ylim(0, 100); ax.set_xticks(t.index); ax.set_ylabel('Share of market-weeks (%)'); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1), columnspacing=0.9, handlelength=1.2)
    ax.set_title('(b) Status of market-weeks by year, all crops', loc='left', fontsize=10, fontweight='bold', color=INK)
    fig.text(0.01, 0.01, '2026 runs to the week of %s. Each filled value carries a flag and a method label.' % LAST.strftime('%d %B %Y'), fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'fig_ms_completeness.png')


# ---------------------------------------------------------------- weather/satellite coverage by year
def fig_layer_coverage():
    cw = pd.read_csv(os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv'), parse_dates=['week_start'])
    cw = cw[(cw.week_start >= '2017-01-02') & (cw.week_start <= LAST)]
    src = [('ERA5-Land', 'era5_tmax'), ('CHIRPS', 'chirps_rain_mm'), ('Sentinel-2', 's2_ndvi'), ('MODIS', 'modis_ndvi')]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.1), sharey=True)
    years = list(range(2017, 2027))
    for ax, c in zip(axs, CROPS):
        d = cw[cw.crop == c]
        M = np.array([[100 * d[d.week_start.dt.year == y][col].notna().mean() for y in years] for _, col in src])
        im = ax.imshow(M, vmin=0, vmax=100, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(years))); ax.set_xticklabels([str(y) for y in years], fontsize=7.5, rotation=0)
        ax.set_yticks(range(len(src))); ax.set_yticklabels([s for s, _ in src])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, '%.0f' % M[i, j], ha='center', va='center', fontsize=7, color='white' if M[i, j] > 60 else INK)
        ax.set_title(c.capitalize(), loc='left', fontsize=10, fontweight='bold', color=INK)
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.tick_params(length=0)
    fig.text(0.01, 0.012, 'Share of weeks (%%) with a value in the crop-level weather and satellite table, by year. 2026 runs to the week of %s.\nSentinel-2 is thinner in 2017-18 (early mission coverage); weeks without a cloud-free composite are missing.' % LAST.strftime('%d %B %Y'), fontsize=7.5, color=MUTED, va='bottom')
    fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, 'fig_ms_layer_coverage.png')


# ---------------------------------------------------------------- zone distance
def fig_zone_distance():
    loc = pd.read_csv(os.path.join(OUT, 'table_market_locations.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    for c in CROPS:
        d = np.sort(loc[(loc.crop == c) & loc.dist_km.notna()].dist_km.values)
        ax.plot(d, np.arange(1, len(d) + 1) / len(d) * 100, color=COL[c], lw=1.8, label='%s (median %.0f km)' % (c.capitalize(), np.median(d)))
    ax.axvline(500, color=MUTED, lw=1, ls=':'); ax.text(505, 4, 'assignment refused\nbeyond 500 km', fontsize=7.5, color=MUTED)
    ax.set_xlabel('Distance from the market\'s district centroid to the nearest production zone (km)'); ax.set_ylabel('Share of markets within that distance (%)')
    ax.set_xlim(0, 1500); ax.set_ylim(0, 100); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title('(a) Distance to the nearest production zone', loc='left', fontsize=10, fontweight='bold', color=INK)
    ax = axs[1]
    z = loc[loc.zone.notna()].groupby(['crop', 'zone']).agg(n=('market_id', 'size'), dist=('dist_km', 'mean')).reset_index()
    z['order'] = z.crop.map({c: i for i, c in enumerate(CROPS)})
    z = z.sort_values(['order', 'n'], ascending=[True, False])
    yy = np.arange(len(z))[::-1]
    ax.barh(yy, z.n, color=[COL[c] for c in z.crop], height=0.7)
    for y, n_, dm in zip(yy, z.n, z.dist): ax.text(n_ + 3, y, '%d  (mean %.0f km)' % (n_, dm), va='center', fontsize=7.2, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([s.split('_', 1)[1] for s in z.zone], fontsize=7.5)
    ax.set_xlim(0, z.n.max() * 1.35); ax.set_xlabel('Markets assigned to the zone (within 500 km)'); ax.tick_params(length=0)
    ax.set_title('(b) Markets per zone', loc='left', fontsize=10, fontweight='bold', color=INK)
    fig.text(0.01, 0.01, 'The forecasting models do not use this assignment; they use the crop-level average over the zones. It is shown to document how far the weather layer is from the markets.', fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, 'fig_ms_zone_distance.png')


# ---------------------------------------------------------------- weather series
def fig_weather_series():
    cw = pd.read_csv(os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv'), parse_dates=['week_start'])
    cw = cw[(cw.week_start >= '2017-01-02') & (cw.week_start <= LAST)]
    spec = [('era5_tmax', 'ERA5-Land weekly maximum temperature (deg C)', None), ('chirps_rain_mm', 'CHIRPS weekly rainfall (mm)', None), ('s2_ndvi', 'NDVI: Sentinel-2 (line) and MODIS (grey)', 'modis_ndvi')]
    # the rainfall panel is drawn only when its yearly totals are on one scale (2000-2024 history vs 2025-26 top-ups differ by about 5x for
    # tomato and onion until the CHIRPS history is re-extracted, see scripts/gee/gee_02_CHIRPS_history_2000_2024.js)
    rain_ok = rain_scale_ok(cw)
    if not rain_ok:
        print('WARNING: CHIRPS yearly totals are not on one scale; rainfall panel omitted from fig_ms_weather_series.png')
        spec = [s_ for s_ in spec if s_[0] != 'chirps_rain_mm']
    fig, axs = plt.subplots(3, len(spec), figsize=(11.5 if rain_ok else 8.5, 7.2), sharex=True)
    for i, c in enumerate(CROPS):
        d = cw[cw.crop == c].set_index('week_start')
        for j, (col, title, extra) in enumerate(spec):
            ax = axs[i, j]
            if extra:
                ax.plot(d.index, d[extra], color='#9AA5B1', lw=0.9, label='MODIS')
            ax.plot(d.index, d[col], color=COL[c], lw=0.9 if col != 'chirps_rain_mm' else 0.7, label='Sentinel-2' if extra else None)
            ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
            if i == 0: ax.set_title(title, loc='left', fontsize=8.6, fontweight='bold', color=INK)
            if j == 0: ax.set_ylabel(c.capitalize(), fontsize=9.5, fontweight='bold', color=COL[c])
            if extra and i == 0: ax.legend(frameon=False, fontsize=7.5, loc='upper right', ncol=2)
    for ax in axs[-1]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.text(0.01, 0.005, 'Crop-level weekly averages across the crop\'s production zones (the series the models use). Gaps in Sentinel-2 are weeks without a cloud-free composite.', fontsize=7.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); save(fig, 'fig_ms_weather_series.png')


# ---------------------------------------------------------------- cross-validation of satellite products
def fig_satellite_validation():
    z = pd.read_csv(os.path.join(BASE, 'data', 'satellite_climate', 'zone_weekly_features.csv'), parse_dates=['week_start'])
    z = z[(z.week_start >= '2017-01-02') & (z.week_start <= LAST)]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5))
    pairs = [('era5_tmax', 'modis_lst_mean', 'ERA5-Land maximum air temperature (deg C)', 'MODIS mean land-surface temperature (deg C)', (10, 48)),
             ('s2_ndvi', 'modis_ndvi', 'Sentinel-2 NDVI', 'MODIS NDVI', (-0.05, 1.0))]
    rows = []
    for ax, (x, y, xl, yl, lim) in zip(axs, pairs):
        d = z[[x, y, 'crop']].dropna()
        for c in CROPS:
            q = d[d.crop == c]
            ax.scatter(q[x], q[y], s=5, color=COL[c], alpha=0.35, lw=0, label='%s (r = %.2f)' % (c.capitalize(), q[x].corr(q[y])))
            rows.append((x, y, c, q[x].corr(q[y]), len(q)))
        ax.plot(lim, lim, color=MUTED, lw=0.9, ls='--'); ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        ax.text(0.03, 0.97, 'all crops: r = %.2f, n = %d zone-weeks' % (d[x].corr(d[y]), len(d)), transform=ax.transAxes, fontsize=8.5, va='top', color=INK, fontweight='bold')
        ax.legend(frameon=False, fontsize=8, loc='lower right', markerscale=2.5)
    axs[0].set_title('(a) Temperature: MODIS land surface vs ERA5-Land air', loc='left', fontsize=10, fontweight='bold', color=INK)
    axs[1].set_title('(b) Vegetation: MODIS vs Sentinel-2 NDVI', loc='left', fontsize=10, fontweight='bold', color=INK)
    fig.text(0.01, 0.012, 'Weekly values per production zone, 2017 to %s; dashed line: 1:1. The products measure different things (land surface vs air temperature;\n250 m vs 10 m pixels) and are expected to agree in pattern, not in level.' % LAST.strftime('%B %Y'), fontsize=7.5, color=MUTED, va='bottom')
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig_ms_satellite_validation.png')
    pd.DataFrame(rows, columns=['x', 'y', 'crop', 'r', 'n']).to_csv(os.path.join(OUT, 'table_satellite_validation_r.csv'), index=False, lineterminator='\n')


# ---------------------------------------------------------------- correlation among variables
def fig_correlation():
    cw = pd.read_csv(os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv'), parse_dates=['week_start'])
    cw = cw[(cw.week_start >= '2017-01-02') & (cw.week_start <= LAST)]
    until = LAST if rain_scale_ok(cw) else pd.Timestamp('2024-12-31')   # keep CHIRPS on one scale
    cw = cw[cw.week_start <= until]
    names = [('era5_tmax', 'ERA5 Tmax'), ('era5_tmin', 'ERA5 Tmin'), ('era5_dtr', 'ERA5 diurnal range'), ('era5_heat_35', 'ERA5 days >= 35 C'), ('chirps_rain_mm', 'CHIRPS rain'),
             ('chirps_excess', 'CHIRPS excess rain'), ('s2_ndvi', 'Sentinel-2 NDVI'), ('s2_ndvi_anom', 'Sentinel-2 NDVI anomaly'), ('modis_ndvi', 'MODIS NDVI'), ('modis_lst_mean', 'MODIS LST mean'), ('modis_lst_frac35', 'MODIS LST share >= 35 C')]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True)
    for ax, c in zip(axs, CROPS):
        d = cw[cw.crop == c][[n for n, _ in names]].dropna(how='any')
        C = d.corr().values
        im = ax.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(names))); ax.set_xticklabels([l for _, l in names], rotation=60, ha='right', fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([l for _, l in names], fontsize=7)
        ax.set_title('%s (n = %d weeks)' % (c.capitalize(), len(d)), loc='left', fontsize=10, fontweight='bold', color=INK)
        for s_ in ax.spines.values(): s_.set_visible(False)
        ax.tick_params(length=0)
    fig.subplots_adjust(left=0.13, right=0.90, bottom=0.27, top=0.93, wspace=0.08)
    cax = fig.add_axes([0.925, 0.30, 0.013, 0.60]); cb = fig.colorbar(im, cax=cax); cb.set_label('Pearson correlation', fontsize=8)
    fig.text(0.01, 0.005, 'Weekly crop-level series, 2017 to %s, weeks with all variables present.' % until.strftime('%B %Y'), fontsize=7.5, color=MUTED)
    save(fig, 'fig_ms_climate_correlation.png')


if __name__ == '__main__':
    p = load_panel()
    fig_architecture(); fig_completeness(p); fig_layer_coverage(); fig_zone_distance(); fig_weather_series(); fig_satellite_validation(); fig_correlation()
