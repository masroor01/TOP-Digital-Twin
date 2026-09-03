# TOP Digital Twin — Web Dashboard

React/Node rebuild of `scripts/24_Simulation_Dashboard.py` (the Streamlit app). Deployable as a **single Node.js process** — no Python at runtime.

## Architecture

- `frontend/` — React (Vite) SPA. Builds to `frontend/dist/`.
- `backend/` — Express API + static file server. Serves the built frontend from the same origin/port, and exposes `/api/*`.
- `backend/src/models/` — the 12 production LightGBM models, each exported to plain JavaScript via [m2cgen](https://github.com/BayesWitnesses/m2cgen) (a direct if/else transliteration of the trained tree logic — not a semantic re-encoding like ONNX). Runs in-process, no Python, no network hop.
- `data/` — a **bundled copy** of the reference data the backend needs (`production_models/reference_rows.csv`, `price_history.csv`, `feature_columns.json`, `feature_ranges.json`, `model_uncertainty.json`, `macro_climate_staleness.json`, `table_dow_pattern.csv`, `table_directional_accuracy.csv`/`_naive.csv`, `table_market_level_accuracy.csv`, and `table_shap_by_layer.csv`/`table_shap_top_features.csv`), ~13MB total.

`GET /api/shap` (backend) serves Script 25's SHAP feature-importance analysis to the frontend's "Feature Importance" tab (`FeatureImportanceTab.jsx`). It is **crop + horizon level only** — computed once per (crop, horizon) on a training-time sample, since one shared model serves every state/market for a given crop+horizon. There is no per-state or per-market SHAP breakdown to serve; the UI discloses this rather than implying a finer granularity than actually exists.

`table_market_level_accuracy.csv` (Script 47) is a **hierarchical, shrinkage-blended** WAPE table (switched from MAPE 2026-09-02, see "Correctness notes" below) — crop-wide -> state -> market, each tier pooling from and shrinking toward its parent so every market/state shows a trustworthy number, not a hidden cell or a noisy raw one for thin history. See the script's docstring for the exact formula and pseudo-count constants.

There is no Python service in this deployment. An earlier version of this app (`web/inference/app.py`, superseded) called out to a FastAPI microservice for inference — that approach was abandoned once we needed to deploy on a Node-only host (Hostinger) that can't run Python. See "Why not ONNX" below.

**Everything the deployed app needs lives inside `web/`.** This is deliberate, not incidental: `web/backend/src/config.js` used to resolve `MODEL_DIR` by walking up to the wider `TOP_Digital_Twin` repo's `Model_Output/` folder — which broke the first real deploy attempt with a 503, because Hostinger's "Deploy Web App" (root directory set to `web`) never checks out anything outside that subtree, so the process crashed on startup trying to read files that weren't there. Fixed by bundling a copy of the needed files into `web/data/` and pointing `config.js` at that instead. Keep it this way — don't reintroduce a path that reaches above `web/`.

### Updating the bundled data

After retraining production models (`scripts/23_Train_Production_Models.py`) or refreshing `table_dow_pattern.csv` (`scripts/26_Weekly_To_Daily_Disaggregation.py`), refresh the bundled copy:

```bash
cp Model_Output/production_models/{feature_columns.json,feature_ranges.json,model_uncertainty.json,macro_climate_staleness.json,reference_rows.csv,price_history.csv} web/data/production_models/
cp Model_Output/table_dow_pattern.csv web/data/
```

Commit the updated files in `web/data/` along with the rest of the retrain.

## Regenerating the models

Whenever the production models are retrained (`scripts/23_Train_Production_Models.py`), regenerate the JS versions and re-validate before trusting them:

```bash
cd web
python generate_js_models.py                          # writes backend/src/models/*.js + fixtures
node backend/src/models/__fixtures__/verify.mjs        # cross-language parity check (must show all OK)
```

`generate_js_models.py` also dumps 200 real reference-row feature vectors per model with their Python ground-truth predictions as fixtures. `verify.mjs` loads each generated JS model and checks it reproduces those predictions — this is the check that matters, since it catches JS-vs-Python float/NaN comparison differences, not just whether the Python-side logic itself was translated correctly. **Do not deploy new models without a clean `verify.mjs` run.**

### Why not ONNX

The first attempt converted models via `onnxmltools` (LightGBM → ONNX) run with `onnxruntime-node`. This was rejected after validation: even at float64 precision, 5 of 12 models showed real (non-precision-noise) prediction drift up to ~0.011 log-price-points, most likely from imperfect missing-value/NaN split-routing conversion in the ONNX TreeEnsemble encoding. `m2cgen`'s direct code transliteration doesn't have this problem — it reproduces LightGBM's own comparison logic verbatim, and validated bit-for-bit identical (`max_diff = 0`) across 200 samples per model, including every case with missing features.

## Local development

```bash
cd web
npm run build   # builds frontend/dist, installs backend deps
npm start        # serves the built app on :4000 (frontend + API, one process)
```

Or for hot-reload during development: `npm run dev` in `frontend/` (Vite, proxies `/api` to `:4000`) alongside `npm run dev` in `backend/`.

## Deployment

Deployed as a single Node.js app (build command: `npm run build`, start command: `npm start`, both run from this `web/` directory). `ANTHROPIC_API_KEY` (optional — powers the AI Briefing tab) is set as an environment variable, not committed; see `backend/.env.example`. Hostinger auto-redeploys on every push to `master` (confirmed 2026-08-31 — no manual redeploy click needed).

## Security notes (from the 2026-08-31 audit)

- **Rate limiting**: general API limit 120 req/min/IP; `/api/ai-brief` specifically limited to 12 req/hour/IP since it calls a real, billed Anthropic API. `app.set('trust proxy', 1)` is required for this to key on the real client IP rather than Hostinger's edge — don't remove it.
- **`overrides` size cap** (`/api/simulate`, `/api/daily-curve`): capped at `MAX_OVERRIDE_KEYS` (40). Confirmed exploitable before this fix — 500 garbage override keys measured at ~1.1s live; a 2MB request body full of short keys (Node's single-threaded, this blocks everyone) would have scaled to minutes.
- **`/api/ai-brief` hardening**: `market`/`state` are now resolved server-side from `marketId` via the trusted reference data, not accepted as free-text client strings (they used to be interpolated directly into the LLM prompt — a real prompt-injection surface). All numeric fields are type/finiteness-checked; `isolatedEffects` is length-capped and each entry's text fields are truncated. **Not fully closed**: `baselinePred`/`scenarioPred`/`isolatedEffects` are still client-supplied rather than recomputed server-side from a stored `/simulate` result — the gold-standard fix, not done here since this feature isn't live on any deployment yet (no `ANTHROPIC_API_KEY` configured on Hostinger as of this audit). Do this before adding a real key.
- Basic headers: `X-Powered-By` disabled, `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: strict-origin-when-cross-origin`.
- **Not done, lower priority**: CORS is still wide open (`cors()` with no origin restriction) — acceptable for now since this is a public, read-mostly tool with no accounts/cookies, but worth tightening to the deployed origin if that ever changes. Dependency audit (`npm audit`) is clean on both `frontend/` and `backend/` as of this date.

## Correctness notes

- **"Model Accuracy" switched from MAPE to WAPE, 2026-09-02** (user-requested, after an empirical comparison — see `scripts/50_MAPE_vs_WAPE_Comparison.py` and `scripts/47_Market_Level_Accuracy.py`'s docstring). Plain MAPE (mean of per-row percentage errors) let a single anomalous near-zero-price row dominate a market's whole accuracy figure. WAPE (`sum(|error|)/sum(actual)`) uses the same errors weighted by each row's own value instead of counted equally. Every field touched by this is renamed, not just relabeled — `wape`/`marketWape`/`marketWapeN`/`marketWapeRaw`/`stateWape`/`stateWapeN` throughout `routes.js`/`data.js`/`SimulationTab.jsx`/`AITab.jsx` (was `mape`/`marketMape`/...), and `model_uncertainty.json` / `table_market_level_accuracy.csv`'s columns are `wape`/`*_wape_*` (was `mape`/`*_mape_*`) — old MAPE-named fields no longer exist anywhere in this data flow, by design, so a stale client caching the old field names fails loudly (`undefined`) rather than silently mixing metrics.
- **`pchip.js` (daily-curve interpolation) fixed 2026-09-02** (audit finding, confirmed): the endpoint derivatives (`d[0]`/`d[n-1]`) used a plain secant slope of just the first/last interval, diverging from scipy's actual `PchipInterpolator` behavior — which uses a non-centered, shape-preserving three-point (Fritsch-Carlson) edge formula instead. This could overshoot specifically in the daily curve's first (0-7 day) and last (13-26 week) segments, the most-viewed part of the chart. Fixed to match scipy's edge-case formula exactly. Verified numerically against real `scipy.interpolate.PchipInterpolator` (5 random test cases, 25 evaluation points each): max deviation ~9e-13, i.e. floating-point noise, not a real difference — this now genuinely matches scipy, not just "closely enough" as the file's docstring previously claimed.
