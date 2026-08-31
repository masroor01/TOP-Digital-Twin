import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { MODEL_DIR, DOW_PATTERN_FILE, DIRECTIONAL_ACCURACY_FILE, MARKET_ACCURACY_FILE } from './config.js';

function readJSON(file) {
  const p = path.join(MODEL_DIR, file);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf-8')) : {};
}

function readCSV(fullPath) {
  const raw = fs.readFileSync(fullPath, 'utf-8');
  return parse(raw, { columns: true, skip_empty_lines: true });
}

// Numeric/boolean coercion — csv-parse gives everything as strings.
const NUMERIC_HINT = /^-?\d+(\.\d+)?(e-?\d+)?$/i;
function coerceRow(row) {
  const out = {};
  for (const [k, v] of Object.entries(row)) {
    if (v === '' || v === undefined) { out[k] = null; continue; }
    if (v === 'True') { out[k] = true; continue; }
    if (v === 'False') { out[k] = false; continue; }
    if (NUMERIC_HINT.test(v)) { out[k] = Number(v); continue; }
    out[k] = v;
  }
  return out;
}

export function loadAll() {
  if (!fs.existsSync(MODEL_DIR)) {
    throw new Error(`Production models not found at ${MODEL_DIR}. Run scripts/23_Train_Production_Models.py first.`);
  }

  const featureColumns = readJSON('feature_columns.json');
  const featureRanges = readJSON('feature_ranges.json');
  const uncertainty = readJSON('model_uncertainty.json');
  const staleness = readJSON('macro_climate_staleness.json');

  const referenceRaw = readCSV(path.join(MODEL_DIR, 'reference_rows.csv'));
  const reference = referenceRaw.map(coerceRow);

  const historyRaw = readCSV(path.join(MODEL_DIR, 'price_history.csv'));
  const history = historyRaw.map(coerceRow);

  let dailyNoise = {};
  if (fs.existsSync(DOW_PATTERN_FILE)) {
    const dowRaw = readCSV(DOW_PATTERN_FILE).map(coerceRow);
    const sums = {}, counts = {};
    for (const r of dowRaw) {
      if (r.crop == null || r.factor_std == null) continue;
      sums[r.crop] = (sums[r.crop] || 0) + r.factor_std;
      counts[r.crop] = (counts[r.crop] || 0) + 1;
    }
    for (const crop of Object.keys(sums)) dailyNoise[crop] = sums[crop] / counts[crop];
  }

  let directionalAccuracy = [];
  if (fs.existsSync(DIRECTIONAL_ACCURACY_FILE)) {
    directionalAccuracy = readCSV(DIRECTIONAL_ACCURACY_FILE).map(coerceRow);
  }

  // Script 47's hierarchical per-market/state MAPE (M6 variant, from real
  // backtest predictions -- distinct from model_uncertainty.json's
  // crop+horizon-wide figure, which is the same for every market by
  // construction since one shared model serves all markets for a given
  // crop/horizon). Two shrinkage tiers: market shrinks toward its own
  // state's (already-shrunk) estimate, state shrinks toward the crop-wide
  // mean -- see scripts/47_Market_Level_Accuracy.py for the formula.
  // Indexed for O(1) lookup per /simulate request.
  const marketAccuracy = new Map();
  if (fs.existsSync(MARKET_ACCURACY_FILE)) {
    const rows = readCSV(MARKET_ACCURACY_FILE).map(coerceRow);
    for (const r of rows) {
      marketAccuracy.set(`${r.crop}_${r.market_id}_${r.horizon_weeks}`, {
        state: r.state,
        marketMapeRaw: r.market_mape_raw,
        marketN: r.market_n,
        marketMapeShrunk: r.market_mape_shrunk,
        stateMapeRaw: r.state_mape_raw,
        stateN: r.state_n,
        stateMapeShrunk: r.state_mape_shrunk,
        cropMapeRaw: r.crop_mape_raw,
      });
    }
  }

  console.log(
    `[data] Loaded ${reference.length} reference rows, ${history.length} history rows, ` +
    `${Object.keys(featureColumns).length} feature-column specs, ${directionalAccuracy.length} directional-accuracy rows, ` +
    `${marketAccuracy.size} market-accuracy cells.`
  );

  return { featureColumns, featureRanges, uncertainty, staleness, reference, history, dailyNoise, directionalAccuracy, marketAccuracy };
}
