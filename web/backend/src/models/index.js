// Local, pure-JS LightGBM inference -- no Python process, no network hop.
// Each model was exported from its trained .joblib via m2cgen (a direct
// if/else transliteration of the tree logic, not a semantic re-encoding
// like ONNX's TreeEnsemble op) and verified bit-for-bit identical to the
// original Python predictions across 200 real reference rows per model,
// including NaN/missing-value routing -- see web/generate_js_models.py and
// backend/src/models/__fixtures__/verify.mjs. Re-run both after any model
// retrain (Script 23) to regenerate these files and re-confirm parity
// before trusting a new set.
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const featureColumns = JSON.parse(fs.readFileSync(path.join(__dirname, 'feature_columns.json'), 'utf-8'));

const modelCache = new Map();

async function loadModel(key) {
  if (modelCache.has(key)) return modelCache.get(key);
  const modPath = path.join(__dirname, `${key}.js`);
  if (!fs.existsSync(modPath)) return null;
  const mod = await import(`file://${modPath.replace(/\\/g, '/')}`);
  const fn = mod[`predict_${key}`];
  modelCache.set(key, fn);
  return fn;
}

function toNumeric(v) {
  if (v === null || v === undefined || v === '') return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function buildInput(cols, features) {
  return cols.map((c) => toNumeric(features[c]));
}

async function predictOne(crop, horizon, features) {
  const key = `${crop}_${horizon}w`;
  const predict = await loadModel(key);
  if (!predict) throw new Error(`No model loaded for ${key}`);
  const cols = featureColumns[key];
  if (!cols) throw new Error(`No feature columns for ${key}`);
  const input = buildInput(cols, features);
  const logPred = predict(input);
  return Math.expm1(logPred);
}

/**
 * items: [{ id, crop, horizon, features }]
 * returns: { [id]: price }
 */
export async function predictBatch(items) {
  const out = {};
  for (const item of items) {
    out[item.id] = await predictOne(item.crop, item.horizon, item.features);
  }
  return out;
}

export async function inferenceHealth() {
  // Warm-load every model once so a health check also verifies every file
  // is present and parses -- cheap after the first call (cached).
  try {
    const keys = Object.keys(featureColumns);
    let loaded = 0;
    for (const key of keys) {
      const fn = await loadModel(key);
      if (fn) loaded += 1;
    }
    return { ok: loaded === keys.length, status: 'ok', models_loaded: loaded };
  } catch (err) {
    return { ok: false, status: String(err.message || err) };
  }
}
