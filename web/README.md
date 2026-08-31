# TOP Digital Twin — Web Dashboard

React/Node rebuild of `scripts/24_Simulation_Dashboard.py` (the Streamlit app). Deployable as a **single Node.js process** — no Python at runtime.

## Architecture

- `frontend/` — React (Vite) SPA. Builds to `frontend/dist/`.
- `backend/` — Express API + static file server. Serves the built frontend from the same origin/port, and exposes `/api/*`.
- `backend/src/models/` — the 12 production LightGBM models, each exported to plain JavaScript via [m2cgen](https://github.com/BayesWitnesses/m2cgen) (a direct if/else transliteration of the trained tree logic — not a semantic re-encoding like ONNX). Runs in-process, no Python, no network hop.

There is no Python service in this deployment. An earlier version of this app (`web/inference/app.py`, superseded) called out to a FastAPI microservice for inference — that approach was abandoned once we needed to deploy on a Node-only host (Hostinger) that can't run Python. See "Why not ONNX" below.

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

Deployed as a single Node.js app (build command: `npm run build`, start command: `npm start`, both run from this `web/` directory). `ANTHROPIC_API_KEY` (optional — powers the AI Briefing tab) is set as an environment variable, not committed; see `backend/.env.example`.
