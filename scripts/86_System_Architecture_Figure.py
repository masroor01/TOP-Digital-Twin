# -*- coding: utf-8 -*-
"""
Script 86 -- system architecture figure for manuscript Section 3 (not the data-layer figure of Section 2).
Flow: weekly refresh (GitHub Actions, human-reviewed pull request) -> weekly panel -> twelve LightGBM models exported to JavaScript ->
scenario engine and API (Express) -> web application; offline evaluation and the AI briefing as side parts.
Output: Model_Output/fig_ms_system_architecture.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED = '#1F2933', '#52606D'
plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig, ax = plt.subplots(figsize=(11.5, 6.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis('off')
fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)


def box(x, y, w, h, fc, title, lines, ec='#9AA5B1', ls='-', tfs=9.2, lfs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.1', fc=fc, ec=ec, ls=ls, lw=1.0))
    ax.text(x + w / 2, y + h - 2.0, title, ha='center', va='center', fontsize=tfs, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + (h - 3.6) / 2, '\n'.join(lines), ha='center', va='center', fontsize=lfs, color=INK, linespacing=1.35)


def arrow(p, q, color='#7B8794', ls='-', lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, color=color, lw=lw, ls=ls))


# row 1: sources and refresh
ax.text(1, 54.2, 'Weekly refresh (GitHub Actions, Mondays 21:30 UTC; human-reviewed pull request)', fontsize=9.4, fontweight='bold', color=INK)
box(1, 38, 20, 13, '#DCE9F7', 'Sources', ['Agmarknet portal reports', 'Google Earth Engine', 'CMIE, RBI, PPAC, Labour Bureau', 'Policy event log (PIB, DGFT)', 'raw files kept on cloud storage'])
box(26, 38, 22, 13, '#E7F1DF', 'Checks and build', ['validation gates for new prices', 'id repair; 70 % coverage rule', 'weather coverage and scale checks', 'weekly panel rebuilt', 'row-count assertions on every join'])
box(53, 38, 20, 13, '#FFF4C2', 'Weekly panel', ['market x crop x ISO week', 'six layers; imputed values flagged', 'staleness of each input stored'])
box(78, 38, 21, 13, '#E4ECF4', 'Retrain and export', ['12 LightGBM models', '(3 crops x 4 horizons)', 'exported to JavaScript', 'parity check against Python'])
arrow((21.3, 44.5), (25.7, 44.5)); arrow((48.3, 44.5), (52.7, 44.5)); arrow((73.3, 44.5), (77.7, 44.5))
ax.text(50, 35.6, 'pull request  ->  a person reviews and merges  ->  merge to master redeploys the site', ha='center', fontsize=8, color=MUTED, style='italic')

# row 2: serving
ax.text(1, 32.4, 'Serving (one Node.js process)', fontsize=9.4, fontweight='bold', color=INK)
box(53, 17, 22, 13, '#F7DADA', 'Scenario engine and API', ['holds the latest feature row per market', 'overrides applied, forecast recomputed', 'attribution of each lever', 'band = validated RMSE of the model', 'accuracy at crop, state, market level'])
box(78, 17, 21, 13, '#E3F1D2', 'Web application', ['simulation, attribution, benchmarks', 'multi-market, validation, features', 'AI briefing, audit', 'daily curve from the weekly points'])
box(27, 17, 22, 13, '#EEEEEE', 'Historical analogues', ['onion export-policy actions:', 'price paths around real dated', 'actions (descriptive, not a', 'model effect)'], ls=(0, (4, 3)))
arrow((77.8, 23.5), (75.3, 23.5), color='#7B8794'); arrow((75.0, 28.0), (78.0, 28.0))
arrow((49.3, 23.5), (52.7, 23.5), ls=(0, (3, 3)))
arrow((84, 37.8), (68, 30.4))   # models -> engine

# row 3: evaluation
ax.text(1, 13.6, 'Evaluation (offline)', fontsize=9.4, fontweight='bold', color=INK)
box(1, 1.5, 22, 10.5, '#E4ECF4', 'Rolling-origin backtest', ['5 folds, test years 2022-2026', 'M0 price-only to M6 six-layer', 'skill vs a no-change guess'])
box(27, 1.5, 22, 10.5, '#E4ECF4', 'Layer tests and benchmarks', ['fold-level test of each layer', 'classical benchmarks', 'crisis episodes, direction'])
box(53, 1.5, 22, 10.5, '#E4ECF4', 'Live record', ['forecasts the site issued,', 'scored on later prices', 'frozen onion forward test'])
box(78, 1.5, 21, 10.5, '#FDEBD0', 'AI briefing (optional)', ['short commentary on the', 'user\'s scenario; rate-limited,', 'server-side key'], ls=(0, (4, 3)))
arrow((12, 12.3), (57, 16.7), ls=(0, (3, 3)))
arrow((38, 12.3), (61, 16.7), ls=(0, (3, 3)))
arrow((64, 12.3), (64, 16.7), ls=(0, (3, 3)))
arrow((88.5, 12.2), (88.5, 16.8), ls=(0, (3, 3)))
ax.text(1, -0.7, 'Solid arrows: data flow. Dashed: optional or descriptive parts. Results of the evaluation are shown in the validation and benchmark tabs.', fontsize=7, color=MUTED)
ax.set_ylim(-2, 56)
fig.savefig(os.path.join(BASE, 'Model_Output', 'fig_ms_system_architecture.png'), dpi=200, facecolor='white')
print('wrote fig_ms_system_architecture.png')
