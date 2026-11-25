# -*- coding: utf-8 -*-
"""
Script 86b -- system architecture figure for manuscript Section 3, styled after the author's own infographic (2026-11-25).
Same content as Script 86, corrected: no AI briefing (feature removed 2026-11-13); the model export feeds the scenario engine;
only the four weekly steps are numbered (serving and evaluation are not a sequence); text sized for a 16 cm wide figure.
Output: Model_Output/fig_ms_system_architecture_styled.png (300 dpi), .pdf and .svg (vector)
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})

NAVY, BLUE, TEAL, GREEN = '#0B4F7C', '#0A6C8A', '#0E8A8F', '#5BB75B'
INK, MUTED = '#16435F', '#5A6B78'
W, H = 10.8, 11.5
Z = 6   # icons sit above the card fill

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')


def stroke(color, lw=1.8):
    return dict(color=color, lw=lw, solid_capstyle='round', solid_joinstyle='round', zorder=Z)


def outline(color, lw=1.9):
    return dict(fill=False, ec=color, lw=lw, zorder=Z)


# ------------------------------------------------------------------ icons (box of half-size s around cx, cy)
def icon(kind, cx, cy, color, s=0.27):
    if kind == 'globe':
        ax.add_patch(Circle((cx, cy), s, **outline(color))); ax.add_patch(Ellipse((cx, cy), s * 1.0, s * 2, **outline(color)))
        ax.plot([cx - s, cx + s], [cy, cy], **stroke(color))
        ax.plot([cx - s * .86, cx + s * .86], [cy + s * .5, cy + s * .5], **stroke(color, 1.2))
        ax.plot([cx - s * .86, cx + s * .86], [cy - s * .5, cy - s * .5], **stroke(color, 1.2))
    elif kind == 'check':
        ax.add_patch(Circle((cx, cy), s, **outline(color)))
        ax.plot([cx - s * .5, cx - s * .1, cx + s * .55], [cy, cy - s * .42, cy + s * .45], **stroke(color, 2.6))
    elif kind == 'table':
        ax.add_patch(FancyBboxPatch((cx - s, cy - s * .8), 2 * s, 1.6 * s, boxstyle='round,pad=0,rounding_size=0.04', **outline(color)))
        ax.plot([cx - s, cx + s], [cy + s * .25, cy + s * .25], **stroke(color)); ax.plot([cx - s, cx + s], [cy - s * .3, cy - s * .3], **stroke(color))
        ax.plot([cx - s * .1, cx - s * .1], [cy + s * .8, cy - s * .8], **stroke(color))
    elif kind == 'refresh':
        for a0 in (20, 200):
            ax.add_patch(Arc((cx, cy), 2 * s, 2 * s, theta1=a0, theta2=a0 + 130, ec=color, lw=2.2, zorder=Z))
            t = np.radians(a0 + 130); x, y = cx + s * np.cos(t), cy + s * np.sin(t)
            d = np.array([-np.sin(t), np.cos(t)]); n = np.array([np.cos(t), np.sin(t)])
            ax.add_patch(Polygon([(x + d[0] * .12, y + d[1] * .12), (x - n[0] * .1 - d[0] * .02, y - n[1] * .1 - d[1] * .02),
                                  (x + n[0] * .1 - d[0] * .02, y + n[1] * .1 - d[1] * .02)], closed=True, fc=color, ec=color, zorder=Z))
    elif kind == 'analogues':
        for i, hgt in enumerate((.5, .9, .65)):
            ax.add_patch(Rectangle((cx - s * .9 + i * s * .65, cy - s * .8), s * .45, hgt * s * 1.5, **outline(color, 1.8)))
        ax.plot([cx - s, cx + s], [cy - s * .8, cy - s * .8], **stroke(color))
        ax.plot([cx - s, cx - s * .3, cx + s * .3, cx + s], [cy + s * .5, cy + s * .1, cy + s * .55, cy + s * .2], **stroke(color, 1.3))
    elif kind == 'gear':
        for a in np.arange(0, 360, 45):
            t = np.radians(a)
            ax.plot([cx + s * .85 * np.cos(t), cx + s * 1.1 * np.cos(t)], [cy + s * .85 * np.sin(t), cy + s * 1.1 * np.sin(t)], **stroke(color, 3.2))
        ax.add_patch(Circle((cx, cy), s * .85, **outline(color)))
        ax.text(cx, cy, 'API', ha='center', va='center', fontsize=8.5, fontweight='bold', color=color, zorder=Z + 1)
    elif kind == 'monitor':
        ax.add_patch(FancyBboxPatch((cx - s, cy - s * .55), 2 * s, 1.5 * s, boxstyle='round,pad=0,rounding_size=0.04', **outline(color)))
        ax.plot([cx, cx], [cy - s * .55, cy - s * .95], **stroke(color)); ax.plot([cx - s * .5, cx + s * .5], [cy - s * .95, cy - s * .95], **stroke(color))
        ax.add_patch(Circle((cx, cy + s * .2), s * .4, **outline(color, 1.3)))
    elif kind == 'trend':
        ax.plot([cx - s, cx - s], [cy + s * .8, cy - s * .8], **stroke(color)); ax.plot([cx - s, cx + s], [cy - s * .8, cy - s * .8], **stroke(color))
        ax.plot([cx - s * .7, cx - s * .2, cx + s * .15, cx + s * .8], [cy - s * .3, cy + s * .35, cy - s * .05, cy + s * .7], **stroke(color, 2.2))
    elif kind == 'lens':
        for i, hgt in enumerate((.4, .75, .55)):
            ax.add_patch(Rectangle((cx - s * 1.0 + i * s * .55, cy - s * .8), s * .38, hgt * s * 1.5, **outline(color, 1.6)))
        ax.add_patch(Circle((cx + s * .35, cy + s * .25), s * .45, **outline(color)))
        ax.plot([cx + s * .7, cx + s * 1.0], [cy - s * .1, cy - s * .45], **stroke(color, 2.6))
    elif kind == 'clip':
        ax.add_patch(FancyBboxPatch((cx - s * .75, cy - s), s * 1.5, 2 * s * .95, boxstyle='round,pad=0,rounding_size=0.04', **outline(color)))
        ax.add_patch(Rectangle((cx - s * .3, cy + s * .8), s * .6, s * .3, fc='white', ec=color, lw=1.8, zorder=Z))
        for k in (.3, -.1, -.5):
            ax.plot([cx - s * .45, cx + s * .45], [cy + s * k, cy + s * k], **stroke(color, 1.3))


# ------------------------------------------------------------------ building blocks
def card(x, ytop, w, h, title, lines, color, badge=None, dashed=False, ico=None):
    ax.add_patch(FancyBboxPatch((x + 0.05, ytop - h - 0.05), w, h, boxstyle='round,pad=0,rounding_size=0.16', fc='#9AA8B4', ec='none', alpha=0.18, zorder=1))
    ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle='round,pad=0,rounding_size=0.16', fc='white', ec=color, lw=2.6,
                                ls=(0, (4, 3)) if dashed else '-', zorder=2))
    y = ytop - 0.18
    if badge:
        ax.add_patch(Circle((x + w / 2, ytop), 0.21, fc=color, ec='white', lw=2, zorder=4))
        ax.text(x + w / 2, ytop, badge, ha='center', va='center', color='white', fontsize=11.5, fontweight='bold', zorder=5)
        y = ytop - 0.36
    if ico:
        icon(ico, x + w / 2, y - 0.30, color); y -= 0.66
    ax.text(x + w / 2, y - 0.08, title, ha='center', va='top', fontsize=13, fontweight='bold', color=color, zorder=5, linespacing=1.15)
    y -= 0.34 + 0.22 * title.count('\n')
    for ln in lines:
        y -= 0.205
        ax.text(x + w / 2, y, ln, ha='center', va='center', fontsize=10.0, color=INK, zorder=5)


def arrow(p, q, color=TEAL, ls='-', lw=2.4, rad=0.0, ms=14):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle='arc3,rad=%s' % rad, zorder=3))


def label(x, y, text, subs=(), inline=None):
    ax.text(x, y, text, ha='left', va='center', fontsize=14.5, fontweight='bold', color=NAVY)
    if inline:
        ax.text(x + 0.12 + 0.118 * len(text), y, inline, ha='left', va='center', fontsize=10.5, color=MUTED)
    for k, sub in enumerate(subs):
        ax.text(x, y - 0.3 - 0.23 * k, sub, ha='left', va='center', fontsize=10.2, color=MUTED)


CW, GAP, X0 = 2.36, 0.37, 0.1
col = lambda i: X0 + i * (CW + GAP)

# ---- weekly refresh
label(0.05, 11.2, 'Weekly refresh', inline='GitHub Actions, Mondays 21:30 UTC; a person reviews every pull request')
y1, h1 = 10.7, 2.75
card(col(0), y1, CW, h1, 'Sources', ['Agmarknet portal reports', 'Google Earth Engine', 'CMIE, RBI, PPAC', 'Labour Bureau', 'Policy log (PIB, DGFT)'], NAVY, '01', ico='globe')
card(col(1), y1, CW, h1, 'Checks and build', ['Price validation gates', 'Market id repair', '70 % coverage rule', 'Weather and scale checks', 'Row-count check on joins'], BLUE, '02', ico='check')
card(col(2), y1, CW, h1, 'Weekly panel', ['Market x crop x ISO week', 'Six data layers', 'Imputed values flagged', 'Age of each input stored'], TEAL, '03', ico='table')
card(col(3), y1, CW, h1, 'Retrain and export', ['12 LightGBM models', '(3 crops x 4 horizons)', 'Exported to JavaScript', 'Parity check vs Python'], GREEN, '04', ico='refresh')
for i in range(3):
    arrow((col(i) + CW + 0.04, y1 - 1.3), (col(i + 1) - 0.04, y1 - 1.3), color=[NAVY, BLUE, TEAL][i])

# ---- hand-over to the serving process
yb = y1 - h1
y2, h2 = 6.65, 2.85
ym = yb - 0.55
xe = col(2) + CW / 2
ax.plot([col(3) + CW / 2, col(3) + CW / 2], [yb - 0.05, ym], **stroke(GREEN, 2.4))
ax.plot([col(3) + CW / 2, xe], [ym, ym], **stroke(GREEN, 2.4))
arrow((xe, ym), (xe, y2 + 0.04), color=TEAL)
ax.text(5.0, ym + 0.2, 'pull request  →  a person reviews and merges  →  merge to master redeploys the site',
        ha='center', va='center', fontsize=10.2, color=INK, style='italic')

# ---- serving
label(0.05, y2 - 0.25, 'Serving', ['one Node.js process'])
card(col(1), y2, CW, h2, 'Historical\nanalogues', ['Onion export-policy', 'actions: price paths', 'around real dated events', '(descriptive, not a', 'model effect)'], NAVY, dashed=True, ico='analogues')
card(col(2), y2, CW, h2, 'Scenario engine\nand API', ['Latest row per market', 'Overrides applied,', 'forecast recomputed', 'Attribution per lever', 'Band = validated RMSE'], TEAL, ico='gear')
card(col(3), y2, CW, h2, 'Web application', ['Seven tabs: simulation,', 'attribution, benchmarks,', 'multi-market, validation,', 'feature importance, audit', 'Daily curve (display aid)'], GREEN, ico='monitor')
arrow((col(1) + CW + 0.04, y2 - 1.45), (col(2) - 0.04, y2 - 1.45), color=NAVY, ls=(0, (4, 3)))
arrow((col(2) + CW + 0.03, y2 - 1.15), (col(3) - 0.03, y2 - 1.15), color=TEAL)
arrow((col(3) - 0.03, y2 - 1.75), (col(2) + CW + 0.03, y2 - 1.75), color=GREEN)

# ---- evaluation
y3, h3 = 3.1, 2.3
label(col(3) + 0.05, y3 - 0.5, 'Evaluation', ['offline, from the same panel;', 'results appear in the', 'validation and benchmark', 'tabs'])
card(col(0), y3, CW, h3, 'Rolling-origin\nbacktest', ['5 folds, tested 2022-26', 'M0 price-only to M6', 'Skill vs no-change guess'], NAVY, ico='trend')
card(col(1), y3, CW, h3, 'Layer tests and\nbenchmarks', ['Fold-level test per layer', 'Classical benchmarks', 'Crisis episodes, direction'], BLUE, ico='lens')
card(col(2), y3, CW, h3, 'Live record', ['Forecasts the site made,', 'scored on later prices', 'Frozen onion forward test'], TEAL, ico='clip')
yt = y2 - h2
arrow((col(2) + CW / 2, y3 + 0.04), (col(2) + CW / 2, yt - 0.04), color=TEAL, ls=(0, (4, 3)))
arrow((col(0) + CW / 2 + 0.3, y3 + 0.04), (col(2) + 0.45, yt - 0.04), color=NAVY, ls=(0, (4, 3)))
arrow((col(1) + CW / 2 + 0.3, y3 + 0.04), (col(2) + 0.95, yt - 0.04), color=BLUE, ls=(0, (4, 3)))

ax.text(0.1, 0.62, 'Solid arrows: data flow. The web application sends the user’s overrides to the engine and receives the forecasts.', fontsize=8.8, color=MUTED, va='center')
ax.text(0.1, 0.36, 'Dashed: descriptive or offline links. Numbers 01 to 04 mark the order of the weekly refresh; serving and evaluation run in parallel.', fontsize=8.8, color=MUTED, va='center')
fig.savefig(os.path.join(OUT, 'fig_ms_system_architecture_styled.png'), dpi=300, facecolor='white')
fig.savefig(os.path.join(OUT, 'fig_ms_system_architecture_styled.pdf'), facecolor='white')
fig.savefig(os.path.join(OUT, 'fig_ms_system_architecture_styled.svg'), facecolor='white')
print('wrote fig_ms_system_architecture_styled (png, pdf, svg)')
