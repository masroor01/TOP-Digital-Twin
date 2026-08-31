// Cross-language parity check: loads each generated JS model, runs it over
// the Python-dumped fixture rows, and compares against Python's own
// LightGBM predictions (ground truth). This is the check that actually
// matters -- it catches JS-vs-Python float/NaN comparison semantics
// differences that an in-Python-only test (m2cgen's export_to_python) can't.
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = path.resolve(__dirname, '..');

const CROPS = ['tomato', 'onion', 'potato'];
const HORIZONS = [1, 4, 13, 26];
const TOLERANCE = 1e-6;

let allOk = true;
const summary = [];

for (const crop of CROPS) {
  for (const h of HORIZONS) {
    const key = `${crop}_${h}w`;
    const fixturePath = path.join(__dirname, `${key}.json`);
    if (!fs.existsSync(fixturePath)) continue;

    const mod = await import(pathToFileURL(path.join(MODELS_DIR, `${key}.js`)));
    const predict = mod[`predict_${key}`];
    if (typeof predict !== 'function') {
      console.error(`FAIL ${key}: export predict_${key} not found`);
      allOk = false;
      continue;
    }

    const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
    let maxDiff = 0;
    let sumDiff = 0;
    let worstIdx = -1;
    for (let i = 0; i < fixture.length; i++) {
      const { input, expected } = fixture[i];
      // Python dumped null for NaN; m2cgen JS comparisons (`x > threshold`)
      // treat JS NaN the same way Python treats float('nan') -- always false.
      const jsInput = input.map((v) => (v === null ? NaN : v));
      const got = predict(jsInput);
      const diff = Math.abs(got - expected);
      sumDiff += diff;
      if (diff > maxDiff) { maxDiff = diff; worstIdx = i; }
    }
    const meanDiff = sumDiff / fixture.length;
    const ok = maxDiff < TOLERANCE;
    if (!ok) allOk = false;
    summary.push({ key, n: fixture.length, maxDiff, meanDiff, ok });
    console.log(
      `${ok ? 'OK  ' : 'FAIL'} ${key.padEnd(12)} n=${fixture.length}  max_diff=${maxDiff.toExponential(3)}  mean_diff=${meanDiff.toExponential(3)}` +
      (ok ? '' : `  (worst row index ${worstIdx})`)
    );
  }
}

console.log('\n=== SUMMARY ===');
console.log(`All models within tolerance (${TOLERANCE}): ${allOk}`);
fs.writeFileSync(path.join(__dirname, 'node_parity_report.json'), JSON.stringify(summary, null, 1));
process.exit(allOk ? 0 : 1);
