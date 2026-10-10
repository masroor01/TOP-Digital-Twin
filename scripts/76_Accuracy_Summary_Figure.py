# -*- coding: utf-8 -*-
"""
Script 76 -- One-page accuracy summary figure (for the PI).
General accuracy (100 - WAPE) and directional accuracy, per crop and horizon:
  * Backtest: production-method M6, five rolling-origin folds, test 2022-2026 (Script 15 predictions;
    directional from the corrected Script 46 table).
  * Live: forecasts the dashboard issued at each weekly refresh since 2026-08-14, scored on later observed
    prices (Script 74). Only the 1- and 4-week horizons have matured.
  * Reference ticks: a "no change" guess (last price carried forward), the bar a forecast has to beat.
Output: Model_Output/fig_accuracy_summary.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')
CROPS = ['tomato', 'onion', 'potato']; HZ = [1, 4, 13, 26]
BLUE, ORANGE, INK, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'


def backtest():
    p = pd.read_csv(os.path.join(OUT, 'dm_market_level_predictions.csv'), parse_dates=['week_start'])
    p = p[p['variant'] == 'M6'].copy()
    pan = pd.read_csv(os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv'),
                      usecols=['crop', 'market_id', 'week_start', 'modal_price_weighted'], parse_dates=['week_start']).dropna(subset=['market_id'])
    pan['market_id'] = pan['market_id'].astype(int); p['market_id'] = p['market_id'].astype(int)
    idx = pan.set_index(['crop', 'market_id', 'week_start'])['modal_price_weighted']
    p['origin_price'] = idx.reindex(pd.MultiIndex.from_frame(p[['crop', 'market_id', 'week_start']])).to_numpy()   # week_start is the origin week
    d = pd.read_csv(os.path.join(OUT, 'table_directional_accuracy.csv')); d = d[d['variant'] == 'M6']
    rows = {}
    for (c, h), g in p.groupby(['crop', 'horizon_weeks']):
        gg = g.dropna(subset=['origin_price'])
        acc = 100 - (g['y_pred'] - g['y_true']).abs().sum() / g['y_true'].sum() * 100
        nac = 100 - (gg['origin_price'] - gg['y_true']).abs().sum() / gg['y_true'].sum() * 100
        dr = d[(d['crop'] == c) & (d['horizon_weeks'] == h)]['directional_accuracy_pct'].iloc[0]
        rows[(c, h)] = dict(acc=acc, naive=nac, dirn=dr)
    return rows


def live():
    s = pd.read_csv(os.path.join(OUT, 'table_live_track_record_summary.csv'))
    return {(r.crop, r.horizon_weeks): dict(acc=100 - r.WAPE_model_pct, naive=100 - r.WAPE_persistence_pct, dirn=r.directional_accuracy_pct,
                                            n=r.n_forecasts, wk=r.n_origin_weeks) for r in s.itertuples()}


def main():
    bt, lv = backtest(), live()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=SURF)
    w = 0.36
    for j, crop in enumerate(CROPS):
        for i, (key, ttl, ylim) in enumerate((('acc', 'General accuracy (100 − average % error)', (0, 100)), ('dirn', 'Directional accuracy (% of moves called right)', (0, 100)))):
            ax = axs[i, j]; ax.set_facecolor(SURF)
            x = np.arange(len(HZ))
            b = [bt[(crop, h)][key] for h in HZ]
            ax.bar(x - w / 2, b, w, color=BLUE, label='Backtest 2022–2026', zorder=3)
            for xi, v in zip(x, b): ax.text(xi - w / 2, 2, '%.0f' % v, ha='center', va='bottom', fontsize=10, color='white', fontweight='bold', zorder=5)
            for k, h in enumerate(HZ):
                if (crop, h) in lv:
                    v = lv[(crop, h)][key]
                    ax.bar(k + w / 2, v, w, color=ORANGE, label='Live, Aug–Oct 2026', zorder=3)
                    ax.text(k + w / 2, 2, '%.0f' % v, ha='center', va='bottom', fontsize=10, color=INK, fontweight='bold', zorder=5)
                else:
                    ax.text(k + w / 2, 3, 'not yet\nmatured', ha='center', va='bottom', fontsize=7.5, color=MUTED)
            if key == 'acc':
                for k, h in enumerate(HZ):
                    ax.plot([k - w, k], [bt[(crop, h)]['naive']] * 2, color=INK, lw=2.2, zorder=4, label='“No change” guess (backtest)')
                    if (crop, h) in lv:
                        ax.plot([k, k + w], [lv[(crop, h)]['naive']] * 2, color=INK, lw=2.2, ls=(0, (2, 1.5)), zorder=4, label='“No change” guess (live)')
            else:
                ax.axhline(50, color=MUTED, lw=1.2, ls='--', zorder=2, label='Coin flip (50%)')
            ax.set_xlim(-0.6, 3.75); ax.set_xticks(x); ax.set_xticklabels(['%d wk' % h for h in HZ]); ax.set_ylim(*ylim)
            ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
            for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
            if j == 0: ax.set_ylabel(ttl, color=INK, fontsize=10)
            if i == 0: ax.set_title(crop.capitalize(), fontsize=14, fontweight='bold', color=INK, loc='left', pad=10)
            ax.set_xlabel('Forecast horizon', color=MUTED, fontsize=9)
    seen = {}
    for a_ in (axs[0, 0], axs[1, 0]):
        h_, l_ = a_.get_legend_handles_labels()
        for hh, ll in zip(h_, l_): seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.925), fontsize=10)
    fig.suptitle('How accurate has the TOP Digital Twin been?', fontsize=18, fontweight='bold', color=INK, x=0.012, ha='left', y=0.985)
    fig.text(0.012, 0.935, 'Price forecasts per market and week. A forecast is only useful if it beats the black “no change” mark.', fontsize=11, color=MUTED, ha='left')
    note = ('Backtest: 5 rolling-origin folds, 2022–2026; forecasts made without seeing the target period (about 0.9 million market-weeks).\n'
            'Live: forecasts the dashboard issued at each weekly refresh from 14 Aug 2026, scored on prices that arrived later. 1-week = 5 origin weeks, 4-week = 2 (potato 3 and 2): indicative only. 13/26-week not yet matured.\n'
            'Onion in the live window: national prices rose about 60% between 10 Aug and 7 Sep 2026; the model forecast falls and missed the rise.  At 1 week no model beats “no change”, and direction is about a coin flip.')
    fig.text(0.012, 0.012, note, fontsize=8.8, color=MUTED, ha='left', va='bottom', linespacing=1.5)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.17, hspace=0.42, wspace=0.12)
    out = os.path.join(OUT, 'fig_accuracy_summary.png'); fig.savefig(out, dpi=160, facecolor=SURF); print('wrote', out)


if __name__ == '__main__':
    main()
