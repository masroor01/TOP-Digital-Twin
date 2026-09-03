import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import { rateLimit } from 'express-rate-limit';
import { predictBatch, inferenceHealth } from './inference.js';
import { pchip } from './pchip.js';
import {
  CROPS, HORIZONS, FEATURE_INFO, POLICY_FIELDS, CLIMATE_FIELDS, MACRO_FIELDS,
  SEASON_LABEL, SEASON_COLOR, seasonFor,
} from './config.js';

const WEEK_MS = 7 * 24 * 3600 * 1000;
const DAY_MS = 24 * 3600 * 1000;

// The real, exposed scenario controls in the sidebar top out at ~11 fields
// (4 policy + 3 climate + 3 macro, at most). Anything past a generous
// multiple of that in a single request's `overrides` object isn't a real
// user interaction -- it's either a bug or an attempt to force the server
// into computing thousands of isolated-effect predictions in one synchronous
// request (confirmed exploitable: 500 garbage keys measured at ~1.1s live;
// linear scaling means a 2MB body full of short keys could block the
// single-threaded Node process for minutes). Reject rather than silently
// truncate, so a real bug upstream doesn't get masked.
const MAX_OVERRIDE_KEYS = 40;
function tooManyOverrides(overrides) {
  return overrides && typeof overrides === 'object' && Object.keys(overrides).length > MAX_OVERRIDE_KEYS;
}

// FIXED 2026-09-02 (audit finding): /multi-history's marketIds query param
// had no cap, unlike every other list-type input in this file -- a
// comma-separated list of thousands of ids would filter store.history that
// many times over in one synchronous request. Same defensive-cap pattern
// as MAX_OVERRIDE_KEYS above.
const MAX_MARKET_IDS = 40;

// /ai-brief calls the real Anthropic API with a server-side key -- unlike
// every other endpoint here, each call has a real dollar cost. Rate-limited
// much tighter than the general API limit for that reason.
const aiBriefLimiter = rateLimit({ windowMs: 60 * 60 * 1000, limit: 12, standardHeaders: true, legacyHeaders: false });

export function buildRouter(store) {
  const router = express.Router();

  // ── Metadata ────────────────────────────────────────────────────────────
  router.get('/meta', (req, res) => {
    res.json({
      crops: CROPS,
      horizons: HORIZONS,
      featureInfo: FEATURE_INFO,
      featureRanges: store.featureRanges,
      staleness: store.staleness,
      policyFields: POLICY_FIELDS,
      climateFields: CLIMATE_FIELDS,
      macroFields: MACRO_FIELDS,
      seasonLabel: SEASON_LABEL,
      seasonColor: SEASON_COLOR,
      marketCounts: countByCrop(store.reference),
    });
  });

  router.get('/health', async (req, res) => {
    res.json({ ok: true, inference: await inferenceHealth() });
  });

  // Script 46's per-market directional (up/down) accuracy test -- static
  // validation results, not computed live. See Model_Output/MANIFEST.md.
  router.get('/directional-accuracy', (req, res) => {
    res.json({ rows: store.directionalAccuracy });
  });

  // Script 25's SHAP feature-importance analysis -- static, crop+horizon
  // level (the shared model doesn't vary by market/state, so there is no
  // per-market SHAP to serve here; see the tab's own disclosure copy).
  router.get('/shap', (req, res) => {
    res.json({ layers: store.shapLayers, features: store.shapFeatures });
  });

  // ── Market selection ────────────────────────────────────────────────────
  router.get('/states', (req, res) => {
    const { crop } = req.query;
    if (!CROPS.includes(crop)) return res.status(400).json({ error: 'invalid crop' });
    const states = [...new Set(
      store.reference.filter((r) => r.crop === crop && r.state).map((r) => r.state)
    )].sort();
    res.json(states);
  });

  router.get('/markets', (req, res) => {
    const { crop, state } = req.query;
    if (!CROPS.includes(crop)) return res.status(400).json({ error: 'invalid crop' });
    const rows = store.reference.filter((r) => r.crop === crop && (!state || r.state === state));
    const markets = rows
      .map((r) => ({ marketId: r.market_id, market: r.market, state: r.state }))
      .sort((a, b) => a.market.localeCompare(b.market));
    res.json(markets);
  });

  router.get('/reference', (req, res) => {
    const row = findBaseRow(store.reference, req.query.crop, req.query.marketId);
    if (!row) return res.status(404).json({ error: 'no baseline row for this crop/market' });
    res.json(row);
  });

  router.get('/history', (req, res) => {
    const { crop, marketId } = req.query;
    const mid = Number(marketId);
    const rows = store.history
      .filter((r) => r.crop === crop && r.market_id === mid)
      .sort((a, b) => new Date(a.week_start) - new Date(b.week_start))
      .map((r) => ({ weekStart: r.week_start, price: r.modal_price_weighted }));
    res.json(rows);
  });

  router.get('/benchmarks', (req, res) => {
    const { crop } = req.query;
    const rows = store.reference.filter((r) => r.crop === crop);
    const TOP_N = 15;
    const priceTop = rows
      .filter((r) => r.last_observed_price != null)
      .sort((a, b) => b.last_observed_price - a.last_observed_price)
      .slice(0, TOP_N)
      .map((r) => ({ marketId: r.market_id, market: r.market, state: r.state, value: r.last_observed_price }))
      .reverse();
    const arrTop = rows
      .filter((r) => r.log_arr != null)
      .map((r) => ({ ...r, arrivalsTonnes: Math.expm1(r.log_arr) }))
      .sort((a, b) => b.arrivalsTonnes - a.arrivalsTonnes)
      .slice(0, TOP_N)
      .map((r) => ({ marketId: r.market_id, market: r.market, state: r.state, value: r.arrivalsTonnes }))
      .reverse();
    res.json({ priceTop: disambiguate(priceTop), arrTop: disambiguate(arrTop) });
  });

  router.get('/multi-history', (req, res) => {
    const { crop, marketIds } = req.query;
    const ids = new Set(String(marketIds || '').split(',').filter(Boolean).map(Number));
    if (ids.size > MAX_MARKET_IDS) {
      return res.status(400).json({ error: `too many marketIds (max ${MAX_MARKET_IDS})` });
    }
    const nameById = new Map();
    for (const r of store.reference) {
      if (r.crop === crop) nameById.set(r.market_id, { market: r.market, state: r.state });
    }
    const series = [...ids].map((mid) => {
      const rows = store.history
        .filter((r) => r.crop === crop && r.market_id === mid)
        .sort((a, b) => new Date(a.week_start) - new Date(b.week_start));
      const meta = nameById.get(mid) || { market: '?', state: '?' };
      return {
        marketId: mid,
        market: meta.market,
        state: meta.state,
        points: rows.map((r) => ({ weekStart: r.week_start, price: r.modal_price_weighted })),
      };
    });
    res.json(series);
  });

  // ── Core simulation ─────────────────────────────────────────────────────
  router.post('/simulate', async (req, res) => {
    try {
      const { crop, marketId, horizon, overrides = {} } = req.body;
      if (!CROPS.includes(crop)) return res.status(400).json({ error: 'invalid crop' });
      if (!HORIZONS.includes(horizon)) return res.status(400).json({ error: 'invalid horizon' });
      if (tooManyOverrides(overrides)) return res.status(400).json({ error: `too many override fields (max ${MAX_OVERRIDE_KEYS})` });

      const baseRow = findBaseRow(store.reference, crop, marketId);
      if (!baseRow) return res.status(404).json({ error: 'no baseline row for this crop/market' });

      const scenario = { ...baseRow, ...overrides };
      const diffCols = Object.keys(overrides).filter((c) => !valuesEqual(overrides[c], baseRow[c]));

      const asOf = new Date(baseRow.week_start);

      // Build every predict() call this simulation needs, batched in one
      // round-trip to the inference service.
      const items = [];
      for (const h of HORIZONS) {
        items.push({ id: `base_${h}`, crop, horizon: h, features: baseRow });
        items.push({ id: `scenario_${h}`, crop, horizon: h, features: scenario });
      }
      for (const col of diffCols) {
        items.push({ id: `isolated_${col}`, crop, horizon, features: { ...baseRow, [col]: scenario[col] } });
      }
      const preds = await predictBatch(items);

      // Ticker: 4-horizon baseline forecasts
      const recentHist = store.history
        .filter((r) => r.crop === crop && r.market_id === baseRow.market_id && new Date(r.week_start) <= asOf)
        .sort((a, b) => new Date(a.week_start) - new Date(b.week_start))
        .slice(-12)
        .map((r) => r.modal_price_weighted)
        .filter((v) => v != null);

      const ticker = HORIZONS.map((h) => {
        const fdate = new Date(asOf.getTime() + h * WEEK_MS);
        const price = preds[`base_${h}`];
        const err = store.uncertainty[`${crop}_${h}w`] || {};
        return {
          horizon: h,
          date: fdate.toISOString().slice(0, 10),
          price,
          rmse: err.rmse ?? null,
          wape: err.wape ?? null,
          season: seasonFor(crop, fdate),
          spark: recentHist.length ? [...recentHist, price] : null,
        };
      });

      const baselinePred = preds[`base_${horizon}`];
      const scenarioPred = preds[`scenario_${horizon}`];
      const delta = scenarioPred - baselinePred;
      const deltaPct = baselinePred ? (100 * delta) / baselinePred : 0;
      const err = store.uncertainty[`${crop}_${horizon}w`] || {};
      // This market's OWN backtested accuracy (Script 47), distinct from the
      // crop+horizon-wide figure above -- see the KPI's help text for why
      // they can differ. Every market gets a number now (no hidden cells) --
      // thin-history markets get a shrinkage-blended figure (pulled toward
      // their state's own estimate) instead of a noisy raw one or nothing.
      // WAPE, not MAPE, since 2026-09-02 -- see Script 47's docstring.
      const marketErr = store.marketAccuracy.get(`${crop}_${baseRow.market_id}_${horizon}`);
      const marketWape = marketErr ? marketErr.marketWapeShrunk : null;
      const marketWapeN = marketErr ? marketErr.marketN : null;
      const marketWapeRaw = marketErr ? marketErr.marketWapeRaw : null;
      const stateWape = marketErr ? marketErr.stateWapeShrunk : null;
      const stateWapeN = marketErr ? marketErr.stateN : null;

      const isolatedEffects = diffCols
        .map((col) => {
          const effect = preds[`isolated_${col}`] - baselinePred;
          const direction = effect > 0 ? 'higher' : effect < 0 ? 'lower' : 'about the same';
          return {
            field: col,
            label: FEATURE_INFO[col]?.label || col,
            before: baseRow[col],
            after: scenario[col],
            effect,
            mechanism: FEATURE_INFO[col]?.mechanism ? FEATURE_INFO[col].mechanism.replace('{dir}', direction) : null,
          };
        })
        .sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));
      const sumIsolated = isolatedEffects.reduce((s, e) => s + e.effect, 0);
      const interactionGap = delta - sumIsolated;

      // Main chart series: full historical line + baseline/scenario forecast
      // curves across all 4 horizons + RMSE uncertainty band.
      const mktHistory = store.history
        .filter((r) => r.crop === crop && r.market_id === baseRow.market_id)
        .sort((a, b) => new Date(a.week_start) - new Date(b.week_start));
      const lastActual = mktHistory.length ? mktHistory[mktHistory.length - 1] : null;
      const lastActualDate = lastActual ? new Date(lastActual.week_start) : asOf;
      const lastActualPrice = lastActual ? lastActual.modal_price_weighted : baselinePred;

      const fcDates = [lastActualDate, ...HORIZONS.map((h) => new Date(asOf.getTime() + h * WEEK_MS))];
      const baselineCurve = [lastActualPrice, ...HORIZONS.map((h) => preds[`base_${h}`])];
      const scenarioCurve = [lastActualPrice, ...HORIZONS.map((h) => preds[`scenario_${h}`])];
      const bandUpper = [lastActualPrice, ...HORIZONS.map((h) => preds[`base_${h}`] + (store.uncertainty[`${crop}_${h}w`]?.rmse || 0))];
      const bandLower = [lastActualPrice, ...HORIZONS.map((h) => preds[`base_${h}`] - (store.uncertainty[`${crop}_${h}w`]?.rmse || 0))];

      const targetDate = new Date(asOf.getTime() + horizon * WEEK_MS);

      const dataWeeksStale = Math.floor((Date.now() - asOf.getTime()) / WEEK_MS);
      const dataQuality = {
        dataWeeksStale,
        sufficientHistory: baseRow.sufficient_history ?? null,
        staleReference: baseRow.stale_reference ?? null,
        pctImputedLast52w: baseRow.pct_imputed_last_52w ?? null,
      };

      res.json({
        baseRow,
        asOf: asOf.toISOString().slice(0, 10),
        targetDate: targetDate.toISOString().slice(0, 10),
        dataQuality,
        ticker,
        kpis: {
          baseline: baselinePred,
          scenario: scenarioPred,
          delta,
          deltaPct,
          rmse: err.rmse ?? null,
          wape: err.wape ?? null,
          marketWape,
          marketWapeN,
          marketWapeRaw,
          stateWape,
          stateWapeN,
          lastObservedPrice: baseRow.last_observed_price ?? null,
          lastObservedDate: baseRow.last_observed_date ?? null,
        },
        isolatedEffects,
        sumIsolated,
        interactionGap,
        diffCols,
        chart: {
          history: mktHistory.map((r) => ({ weekStart: r.week_start, price: r.modal_price_weighted })),
          dates: fcDates.map((d) => d.toISOString().slice(0, 10)),
          baselineCurve,
          scenarioCurve,
          bandUpper,
          bandLower,
        },
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: String(err.message || err) });
    }
  });

  // ── Daily disaggregation (PCHIP through the weekly ticker points) ──────
  router.post('/daily-curve', async (req, res) => {
    try {
      const { crop, marketId, overrides = {} } = req.body;
      if (tooManyOverrides(overrides)) return res.status(400).json({ error: `too many override fields (max ${MAX_OVERRIDE_KEYS})` });
      const baseRow = findBaseRow(store.reference, crop, marketId);
      if (!baseRow) return res.status(404).json({ error: 'no baseline row' });
      if (!(crop in store.dailyNoise)) return res.json({ points: [], note: 'no daily-noise factor for this crop' });

      const scenario = { ...baseRow, ...overrides };
      const asOf = new Date(baseRow.week_start);
      const items = HORIZONS.map((h) => ({ id: `h${h}`, crop, horizon: h, features: scenario }));
      const preds = await predictBatch(items);

      const basePrice = baseRow.log_price != null ? Math.expm1(baseRow.log_price) : null;
      const xs = [0, ...HORIZONS.map((h) => h * 7)];
      const ys = [basePrice, ...HORIZONS.map((h) => preds[`h${h}`])].map((p) => Math.log1p(p));
      if (ys.some((v) => v == null || Number.isNaN(v))) return res.json({ points: [] });

      const interp = pchip(xs, ys);
      const maxDay = HORIZONS[HORIZONS.length - 1] * 7;
      const today = new Date();
      const noise = store.dailyNoise[crop];

      const points = [];
      for (let day = 0; day <= maxDay; day++) {
        const date = new Date(asOf.getTime() + day * DAY_MS);
        if (date < (today > asOf ? today : asOf)) continue;
        const price = Math.expm1(interp(day));
        points.push({
          date: date.toISOString().slice(0, 10),
          price,
          band: price * noise,
          season: seasonFor(crop, date),
        });
      }
      res.json({ points });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: String(err.message || err) });
    }
  });

  // ── AI policy briefing (Anthropic proxy, server-side key) ──────────────
  // Calls a real, billed Anthropic API with a server-side key, so this
  // route is deliberately more defensive than the others: rate-limited
  // separately and tighter (aiBriefLimiter, above), crop/horizon checked
  // against known lists, and market/state resolved server-side from
  // `marketId` via the trusted reference data rather than trusted as raw
  // client-supplied strings (both used to be free-text fields interpolated
  // straight into the LLM prompt -- a real prompt-injection surface: a
  // crafted `market`/`state` string could try to override the system-style
  // instructions above it). isolatedEffects is still client-supplied
  // (recomputing baselinePred/scenarioPred/isolatedEffects server-side from
  // a stored simulate result would close that too -- not done here since
  // this feature isn't live on any deployment yet; worth doing before it is).
  router.post('/ai-brief', aiBriefLimiter, async (req, res) => {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) return res.status(503).json({ error: 'ANTHROPIC_API_KEY not configured on the server' });
    try {
      const { crop, marketId, horizon, baselinePred, scenarioPred, deltaPct, delta, rmse, wape, isolatedEffects } = req.body;

      if (!CROPS.includes(crop)) return res.status(400).json({ error: 'invalid crop' });
      if (!HORIZONS.includes(horizon)) return res.status(400).json({ error: 'invalid horizon' });
      const baseRow = findBaseRow(store.reference, crop, marketId);
      if (!baseRow) return res.status(404).json({ error: 'no baseline row for this crop/market' });
      const market = baseRow.market;
      const state = baseRow.state;

      const numericFields = { baselinePred, scenarioPred, deltaPct, delta, rmse, wape };
      for (const [key, val] of Object.entries(numericFields)) {
        if (val != null && (typeof val !== 'number' || !Number.isFinite(val))) {
          return res.status(400).json({ error: `${key} must be a finite number` });
        }
      }
      if (!Array.isArray(isolatedEffects) || isolatedEffects.length > MAX_OVERRIDE_KEYS) {
        return res.status(400).json({ error: `isolatedEffects must be an array of at most ${MAX_OVERRIDE_KEYS} items` });
      }
      for (const e of isolatedEffects) {
        if (typeof e?.effect !== 'number' || !Number.isFinite(e.effect)) {
          return res.status(400).json({ error: 'each isolatedEffects entry needs a finite numeric effect' });
        }
      }

      const changesText = isolatedEffects
        .map((e) => `- ${String(e.label ?? '').slice(0, 120)}: ${String(e.before).slice(0, 40)} -> ${String(e.after).slice(0, 40)} (isolated effect: ${e.effect >= 0 ? '+' : ''}${e.effect.toFixed(0)} Rs/quintal)`)
        .join('\n');
      const prompt = `You are a policy-analysis assistant embedded in an agricultural price forecasting dashboard for Indian APMC markets (Tomato/Onion/Potato, HADP-04, SKUAST-K). A user ran a what-if scenario. Ground your answer STRICTLY in the numbers given below — do not invent statistics, events, or data you were not given.

Crop: ${crop}
Market: ${market}, ${state}
Forecast horizon: ${horizon} weeks ahead
Baseline prediction: Rs ${Math.round(baselinePred).toLocaleString()}/quintal
Scenario prediction: Rs ${Math.round(scenarioPred).toLocaleString()}/quintal (${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(1)}%, ${delta >= 0 ? '+' : ''}${delta.toFixed(0)} Rs/quintal)
Model's typical error at this horizon: ${rmse ? `±Rs ${Math.round(rmse).toLocaleString()} (${wape.toFixed(0)}% WAPE)` : 'not available'}

Changes made in this scenario, with the isolated effect of each:
${changesText}

Write ONE paragraph (120-160 words) of plain-language policy commentary for an agricultural-market analyst. Cover: (1) what this price move would mean for farmers vs consumers, (2) which lever is doing most of the work and whether that matches known market structure for this crop, (3) one caveat about relying on this scenario (it is a what-if from a single model, not a validated forecast; thin-data markets and feature interactions add uncertainty). Plain prose only — no bullet points, headers, or markdown.`;

      const client = new Anthropic({ apiKey });
      const resp = await client.messages.create({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 400,
        messages: [{ role: 'user', content: prompt }],
      });
      res.json({ text: resp.content[0].text });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: String(err.message || err) });
    }
  });

  return router;
}

function findBaseRow(reference, crop, marketId) {
  const mid = Number(marketId);
  return reference.find((r) => r.crop === crop && r.market_id === mid) || null;
}

function valuesEqual(a, b) {
  if (a == null && b == null) return true;
  return a === b;
}

function countByCrop(reference) {
  const counts = {};
  for (const r of reference) counts[r.crop] = (counts[r.crop] || 0) + 1;
  return counts;
}

function disambiguate(rows) {
  const counts = {};
  for (const r of rows) counts[r.market] = (counts[r.market] || 0) + 1;
  return rows.map((r) => ({ ...r, label: counts[r.market] > 1 ? `${r.market} (${r.state})` : r.market }));
}
