# -*- coding: utf-8 -*-
"""
Script 17 — Temporal Fusion Transformer (TFT) Secondary Model
==============================================================
Trains TFT (Lim et al., 2021) as a secondary model alongside LightGBM
(Scripts 12 & 15). Provides multi-horizon forecasting with built-in
attention weights and interpretability.

Why TFT as secondary?
  - Handles temporal dependencies natively (no manual lag engineering)
  - Quantile outputs → prediction intervals (10th / 50th / 90th)
  - Variable-selection networks show which features matter per step
  - Attention weights visualise which historical weeks drive each forecast

Inputs (same as Script 15):
  data/agmarknet_weekly/top_weekly_panel.csv
  data/satellite_climate/crop_weekly_features.csv
  data/cmie_macro/cmie_macro_2017_2025.csv
  data/rbi_dbie/rbi_dbie_macro_2017_2025.csv
  data/ppac_macro/ppac_diesel_lpg_2017_2025.csv

Outputs (Model_Output/):
  tft_raw_results.csv          — per fold × horizon × crop metrics
  table_tft_vs_lgbm.csv        — TFT vs LightGBM M4 comparison (mean folds)
  fig_tft_vs_lgbm.png          — side-by-side R² and MAPE bar chart
  fig_tft_intervals.png        — prediction intervals for onion (fold 4)
  fig_tft_importance.png       — variable importance from TFT encoder

CV:   4-fold rolling-origin (same folds as Script 15)
Horizons: h = 1, 4, 13, 26 weeks (max encoder → 52 weeks lookback)
Crops:  tomato, onion, potato (one model per crop per fold)

Run: python scripts/17_TFT_Model.py
Estimated runtime: 3–5 hours on CPU (14 threads). Use FAST_MODE=True
for a quick test (~30 min) with reduced markets and shorter training.
"""

import io, os, sys, time, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
pl.seed_everything(42, workers=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CMIE_FILE= os.path.join(BASE, 'data', 'cmie_macro',      'cmie_macro_2017_2025.csv')
RBI_FILE = os.path.join(BASE, 'data', 'rbi_dbie',        'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE= os.path.join(BASE, 'data', 'ppac_macro',      'ppac_diesel_lpg_2017_2025.csv')
SAT_FILE = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
LGBM_FILE= os.path.join(BASE, 'Model_Output', 'ablation_raw_results.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato']
# TFT multi-step: model outputs 26 steps; we evaluate at these steps
HORIZONS = [1, 4, 13, 26]
MAX_PREDICTION_LENGTH = 26   # decoder length (longest horizon)
MAX_ENCODER_LENGTH    = 52   # lookback window (1 year)
SEED = 42

# SMOKE_TEST: 1 crop, 1 fold, ~15 markets, 3 epochs, tiny network, short
# encoder — purely to measure real sec/batch throughput on this machine
# before committing to any larger run. Overrides FAST_MODE when True.
SMOKE_TEST = False

# TIMING_TEST: full FAST_MODE architecture but restricted to 1 crop x 1
# fold — measures real epoch time at production settings.
TIMING_TEST = False

# FAST_MODE: fewer representative markets + shorter encoder + early
# stopping so a 4-fold x 3-crop run finishes in hours, not days.
# 5.5 min/epoch was measured at hidden_size=64/encoder=52w/50 markets —
# scaled down here to keep the full run in a ~4-6h budget.
FAST_MODE = True
MARKETS_PER_CROP_FAST = {'tomato': 35, 'onion': 30, 'potato': 35}
MAX_ENCODER_LENGTH = 26          # was 52w; halves sequence compute
EARLY_STOP_PATIENCE = 5
MAX_EPOCHS_CAP = 20              # ceiling; early stopping usually ends sooner

if SMOKE_TEST:
    CROPS = ['onion']
    FOLDS_LIMIT = 1
    MARKETS_PER_CROP_FAST = {'onion': 15}
    MAX_ENCODER_LENGTH = 26
    SMOKE_EPOCHS = 3
elif TIMING_TEST:
    CROPS = ['onion']
    FOLDS_LIMIT = 1
else:
    FOLDS_LIMIT = None

FOLDS = [
    {'fold': 1, 'train_end': '2021-06-30',
     'val_start': '2021-07-01', 'val_end': '2021-12-31',
     'test_start': '2022-01-01', 'test_end': '2022-12-31', 'test_year': 2022},
    {'fold': 2, 'train_end': '2022-06-30',
     'val_start': '2022-07-01', 'val_end': '2022-12-31',
     'test_start': '2023-01-01', 'test_end': '2023-12-31', 'test_year': 2023},
    {'fold': 3, 'train_end': '2023-06-30',
     'val_start': '2023-07-01', 'val_end': '2023-12-31',
     'test_start': '2024-01-01', 'test_end': '2024-12-31', 'test_year': 2024},
    {'fold': 4, 'train_end': '2024-06-30',
     'val_start': '2024-07-01', 'val_end': '2024-12-31',
     'test_start': '2025-01-01', 'test_end': '2025-12-31', 'test_year': 2025},
]
if FOLDS_LIMIT is not None:
    FOLDS = FOLDS[:FOLDS_LIMIT]

# TFT hyperparameters — smaller network for the smoke test to isolate
# whether slowness is dataloading (fixed cost) or model compute (scales
# with hidden_size/lstm_layers)
TFT_PARAMS = dict(
    hidden_size           = 16 if SMOKE_TEST else 64,
    lstm_layers            = 1 if SMOKE_TEST else 2,
    dropout               = 0.1,
    attention_head_size   = 2 if SMOKE_TEST else 4,
    learning_rate         = 1e-3,
    log_interval          = -1,
    reduce_on_plateau_patience = 3,
)
TRAINER_PARAMS = dict(
    max_epochs        = SMOKE_EPOCHS if SMOKE_TEST else MAX_EPOCHS_CAP,
    gradient_clip_val = 0.1,
    enable_progress_bar = True,
    enable_model_summary = False,
    devices           = 1,
    accelerator       = 'cpu',
    logger            = False,
    enable_checkpointing = False,
)
BATCH_SIZE  = 128 if SMOKE_TEST else 64
NUM_WORKERS = 0      # Windows: DataLoader must use 0 workers

QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
MEDIAN_IDX = 3  # index of 0.5 quantile in QUANTILES

CROP_COLORS = {'tomato': '#E63946', 'onion': '#F4A261', 'potato': '#457B9D'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD & PREPARE PANEL
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 65)
print('SCRIPT 17: TEMPORAL FUSION TRANSFORMER (TFT)')
print('=' * 65)
print(f'  Smoke test     : {SMOKE_TEST}')
print(f'  Timing test    : {TIMING_TEST}')
print(f'  Fast mode      : {FAST_MODE}')
print(f'  Crops          : {CROPS}')
print(f'  Horizons       : {HORIZONS}')
print(f'  Encoder length : {MAX_ENCODER_LENGTH}w')
print(f'  Decoder length : {MAX_PREDICTION_LENGTH}w')
print(f'  Folds          : {len(FOLDS)}')
print(f'  Batch size     : {BATCH_SIZE}')
print(f'  Hidden size    : {TFT_PARAMS["hidden_size"]}')
print(f'  Max epochs     : {TRAINER_PARAMS["max_epochs"]}\n')

print('[1] Loading panel ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2025-12-31')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)
df['year']  = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month
print(f'   Panel: {len(df):,} rows')

# Macro join (monthly features)
macro_dfs = []
for fpath in [CMIE_FILE, RBI_FILE, PPAC_FILE]:
    if os.path.exists(fpath):
        macro_dfs.append(pd.read_csv(fpath))

MACRO_COLS = []
if macro_dfs:
    macro = macro_dfs[0]
    for m in macro_dfs[1:]:
        macro = macro.merge(m, on=['year', 'month'], how='outer',
                            suffixes=('', '_dup'))
        macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
    drop_cols = [c for c in ['date', 'date_x', 'date_y'] if c in macro.columns]
    df = df.merge(macro.drop(columns=drop_cols, errors='ignore'),
                  on=['year', 'month'], how='left')
    MACRO_COLS = [c for c in macro.columns if c not in ('date', 'year', 'month')]
    print(f'   Macro: {len(MACRO_COLS)} series joined')

# Satellite / climate join (crop × week)
print('[2] Loading satellite/climate features ...')
sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'])
ERA5_COLS  = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr',
              'era5_heat_35', 'era5_heat_38']
CHIRPS_COLS= ['chirps_rain_mm', 'chirps_rain_max', 'chirps_excess']
S2_COLS    = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']
MODIS_COLS = ['modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max',
              'modis_lst_frac35']
SAT_ALL_COLS = [c for c in ERA5_COLS + CHIRPS_COLS + S2_COLS + MODIS_COLS
                if c in sat.columns]
df = df.merge(sat[['crop', 'week_start'] + SAT_ALL_COLS],
              on=['crop', 'week_start'], how='left')
print(f'   Satellite features: {len(SAT_ALL_COLS)}')

# Time index (integer, required by TimeSeriesDataSet)
week_map = {w: i for i, w in enumerate(sorted(df['week_start'].unique()))}
df['time_idx'] = df['week_start'].map(week_map)

# Unique series ID = crop_market
df['series_id'] = df['crop'] + '__' + df['market']

# Forward-fill price within each series (non-trading weeks have NaN;
# TFT rejects NaN targets — ffill replicates last known price, same as Script 15)
df['modal_price_filled'] = (df.groupby('series_id')['modal_price_weighted']
                              .transform(lambda x: x.ffill().bfill()))
df['arrivals_filled'] = (df.groupby('series_id')['arrivals_tonnes_week']
                           .transform(lambda x: x.fillna(0)))
df['log_price']    = np.log(df['modal_price_filled'].clip(lower=1))
df['log_arrivals'] = np.log(df['arrivals_filled'].clip(lower=1))

# Seasonality features (known in future — safe for decoder)
df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(float)
df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

FUTURE_KNOWN = ['week_sin', 'week_cos', 'month_sin', 'month_cos']

# Past-only features (not known in future)
PAST_FEATURES = [c for c in MACRO_COLS + SAT_ALL_COLS if c in df.columns]
PAST_FEATURES += ['log_arrivals']

# Fill remaining NaN with 0 (already imputed in panel; satellite has ~5% missing)
for col in PAST_FEATURES + FUTURE_KNOWN:
    if col in df.columns:
        df[col] = df[col].fillna(0.0).astype(float)

df = df.sort_values(['series_id', 'time_idx']).reset_index(drop=True)
print(f'   Panel ready: {len(df):,} rows  |  {df["series_id"].nunique()} series')
print(f'   Past features : {len(PAST_FEATURES)}')
print(f'   Future known  : {len(FUTURE_KNOWN)} (seasonality)')


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    """Return RMSE, MAE, MAPE, R² for log-price predictions (back-transformed)."""
    y_true = np.exp(np.array(y_true, dtype=float))
    y_pred = np.exp(np.array(y_pred, dtype=float))
    mask   = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 5:
        return np.nan, np.nan, np.nan, np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return rmse, mae, mape, r2


def build_dataset(data, train_cutoff_idx, val_size, predict=False):
    """Build TimeSeriesDataSet from filtered data."""
    return TimeSeriesDataSet(
        data,
        time_idx                  = 'time_idx',
        target                    = 'log_price',
        group_ids                 = ['series_id'],
        max_encoder_length        = MAX_ENCODER_LENGTH,
        max_prediction_length     = MAX_PREDICTION_LENGTH,
        time_varying_known_reals  = FUTURE_KNOWN,
        time_varying_unknown_reals= PAST_FEATURES + ['log_price'],
        target_normalizer         = None,  # we normalise manually via log
        allow_missing_timesteps   = True,
        predict_mode              = predict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN CV LOOP
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Running TFT: 4 folds × 4 horizons × 3 crops ...')
print(f'    Using pytorch-forecasting {__import__("pytorch_forecasting").__version__}')
print(f'    Torch {torch.__version__}  |  CPU threads: {torch.get_num_threads()}\n')

all_results = []
# Store last fold's predictions for onion (for interval plot)
onion_interval_store = None

t_total = time.time()

for crop in CROPS:
    df_crop = df[df['crop'] == crop].copy()

    # FAST_MODE: subsample representative markets by coverage
    if FAST_MODE:
        n = MARKETS_PER_CROP_FAST[crop]
        coverage = df_crop.groupby('market')['modal_price_weighted'].count()
        top_mkts = coverage.nlargest(n).index.tolist()
        df_crop  = df_crop[df_crop['market'].isin(top_mkts)].copy()
        print(f'  [{crop}] FAST_MODE: {n} markets')

    n_series = df_crop['series_id'].nunique()
    print(f'\n  ── {crop.upper()} ({n_series} markets) ──')

    for fold in FOLDS:
        t_fold = time.time()
        fnum   = fold['fold']

        # Time-index boundaries
        train_end_ts  = pd.Timestamp(fold['train_end'])
        val_end_ts    = pd.Timestamp(fold['val_end'])
        test_start_ts = pd.Timestamp(fold['test_start'])
        test_end_ts   = pd.Timestamp(fold['test_end'])

        train_end_idx = week_map.get(
            df_crop[df_crop['week_start'] <= train_end_ts]['week_start'].max(),
            max(week_map.values()))
        val_end_idx   = week_map.get(
            df_crop[df_crop['week_start'] <= val_end_ts]['week_start'].max(),
            max(week_map.values()))

        # Train+val slice (include encoder lookback before train_start)
        data_tv = df_crop[df_crop['week_start'] <= val_end_ts].copy()

        if len(data_tv) == 0:
            print(f'  {crop} fold{fnum}: no data, skipping')
            continue

        val_size_weeks = len(df_crop[(df_crop['week_start'] > train_end_ts) &
                                     (df_crop['week_start'] <= val_end_ts)
                                     ]['week_start'].unique())

        # Build training dataset: windows must end at/before train_end_idx
        # so no validation-period target ever appears in a training sample
        try:
            training = TimeSeriesDataSet(
                data_tv[data_tv['time_idx'] <= train_end_idx],
                time_idx                  = 'time_idx',
                target                    = 'log_price',
                group_ids                 = ['series_id'],
                max_encoder_length        = MAX_ENCODER_LENGTH,
                max_prediction_length     = MAX_PREDICTION_LENGTH,
                time_varying_known_reals  = FUTURE_KNOWN,
                time_varying_unknown_reals= PAST_FEATURES + ['log_price'],
                target_normalizer         = None,
                allow_missing_timesteps   = True,
                predict_mode              = False,
            )
            # Validation: same series/params, windows starting after train cutoff,
            # encoder history drawn from data_tv (may reach back into train period)
            validation = TimeSeriesDataSet.from_dataset(
                training, data_tv, min_prediction_idx=train_end_idx + 1,
                stop_randomization=True)
        except Exception as e:
            print(f'  {crop} fold{fnum}: dataset build failed — {e}')
            continue

        train_dl = training.to_dataloader(
            train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
        val_dl = validation.to_dataloader(
            train=False, batch_size=BATCH_SIZE * 4, num_workers=NUM_WORKERS)

        # Build TFT model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            **TFT_PARAMS,
            loss=QuantileLoss(quantiles=QUANTILES),
        )
        n_params = sum(p.numel() for p in tft.parameters())

        # Train (early stopping on val_loss so plateaued folds don't burn
        # the full epoch budget)
        early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE,
                                    mode='min')
        trainer = pl.Trainer(**TRAINER_PARAMS, callbacks=[early_stop])
        try:
            trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
        except Exception as e:
            print(f'  {crop} fold{fnum}: training failed — {e}')
            continue

        # Predict on test period
        # Build test dataset: include enough history (encoder) + test weeks
        lookback_start = test_start_ts - pd.Timedelta(weeks=MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH)
        data_test_ctx  = df_crop[df_crop['week_start'] >= lookback_start].copy()
        data_test_ctx  = data_test_ctx[data_test_ctx['week_start'] <= test_end_ts].copy()

        try:
            test_dataset = TimeSeriesDataSet.from_dataset(
                training, data_test_ctx, predict=True, stop_randomization=True)
            test_dl = test_dataset.to_dataloader(
                train=False, batch_size=BATCH_SIZE * 4, num_workers=NUM_WORKERS)
            pred_result = tft.predict(test_dl, mode='raw', return_index=True)
            raw_preds = pred_result.output   # dict with 'prediction' key
            index     = pred_result.index    # DataFrame with series_id, time_idx
            # raw_preds['prediction'] shape: (n_series_cutpoints, max_pred_len, n_quantiles)
            median_preds = raw_preds['prediction'][:, :, MEDIAN_IDX]  # (N, 26)
        except Exception as e:
            print(f'  {crop} fold{fnum}: prediction failed — {e}')
            continue

        # Evaluate at each horizon h
        for h in HORIZONS:
            h_idx = h - 1  # 0-indexed step

            # Match predictions to actuals
            y_pred_list, y_true_list = [], []
            for i, (sid, cutoff_idx) in enumerate(zip(index['series_id'],
                                                        index['time_idx'])):
                target_idx = cutoff_idx + h
                actual_rows = df_crop[
                    (df_crop['series_id'] == sid) &
                    (df_crop['time_idx']  == target_idx)
                ]
                if actual_rows.empty:
                    continue
                actual_week = actual_rows['week_start'].iloc[0]
                if not (test_start_ts <= actual_week <= test_end_ts):
                    continue
                y_true_list.append(actual_rows['log_price'].iloc[0])
                y_pred_list.append(float(median_preds[i, h_idx]))

            if len(y_true_list) < 5:
                continue

            rmse, mae, mape, r2 = metrics(y_true_list, y_pred_list)
            elapsed = time.time() - t_fold
            print(f'   TFT {crop:6s}  fold{fnum} h={h:2d}w | '
                  f'RMSE={rmse:7.1f}  MAPE={mape:5.1f}%  R²={r2:6.3f}  '
                  f'[{elapsed:.0f}s]')

            all_results.append({
                'variant':       'TFT',
                'crop':          crop,
                'fold':          fnum,
                'horizon_weeks': h,
                'test_year':     fold['test_year'],
                'n_params':      n_params,
                'RMSE':          round(rmse, 1) if rmse is not None else np.nan,
                'MAE':           round(mae,  1) if mae  is not None else np.nan,
                'MAPE':          round(mape, 2) if mape is not None else np.nan,
                'R2':            round(r2,   4) if r2   is not None else np.nan,
                'N':             len(y_true_list),
                'fit_sec':       round(elapsed, 1),
            })

            # Store onion fold 4 predictions for interval plot
            if crop == 'onion' and fnum == 4 and h == 13:
                onion_interval_store = {
                    'y_true': y_true_list,
                    'y_pred': y_pred_list,
                    'y_lo':   [float(raw_preds['prediction'][i, h_idx, 1])
                               for i in range(len(index['series_id']))
                               if len(y_true_list) > 0][:len(y_true_list)],
                    'y_hi':   [float(raw_preds['prediction'][i, h_idx, 5])
                               for i in range(len(index['series_id']))
                               if len(y_true_list) > 0][:len(y_true_list)],
                }

        fold_time = time.time() - t_fold
        print(f'  {crop} fold{fnum} done in {fold_time/60:.1f} min')

    print(f'  {crop} complete')

total_time = time.time() - t_total
print(f'\n  Total TFT time: {total_time/60:.1f} min')

# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE RAW RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if not all_results:
    print('\nNo results collected. Check errors above.')
    sys.exit(1)

results_df = pd.DataFrame(all_results)
raw_path   = os.path.join(OUT_DIR, 'tft_raw_results.csv')
results_df.to_csv(raw_path, index=False)
print(f'\n[4] Saved: {raw_path}  ({os.path.getsize(raw_path)/1024:.1f} KB)')

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON TABLE: TFT vs LightGBM M4
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] Building comparison table (TFT vs LightGBM M4) ...')

tft_mean = (results_df.groupby(['crop', 'horizon_weeks'])[['RMSE','MAPE','R2']]
            .mean().round({'RMSE':1,'MAPE':2,'R2':4}).reset_index())
tft_mean['variant'] = 'TFT'

lgbm_m4 = None
if os.path.exists(LGBM_FILE):
    lgbm_raw = pd.read_csv(LGBM_FILE)
    lgbm_m4  = (lgbm_raw[lgbm_raw['variant'] == 'M4']
                .groupby(['crop', 'horizon_weeks'])[['RMSE','MAPE','R2']]
                .mean().round({'RMSE':1,'MAPE':2,'R2':4}).reset_index())
    lgbm_m4['variant'] = 'LightGBM_M4'

comparison = pd.concat([tft_mean, lgbm_m4], ignore_index=True) if lgbm_m4 is not None else tft_mean
comp_path  = os.path.join(OUT_DIR, 'table_tft_vs_lgbm.csv')
comparison.to_csv(comp_path, index=False)
print(f'   Saved: {comp_path}')

# Print summary tables
for crop in CROPS:
    print(f'\n  {crop.upper()}')
    sub = comparison[comparison['crop'] == crop].sort_values(['horizon_weeks', 'variant'])
    header = f"  {'Variant':<16} {'h=1w':>6} {'MAPE':>7} {'R²':>7}  " \
             f"{'h=4w':>6} {'MAPE':>7} {'R²':>7}  " \
             f"{'h=13w':>6} {'MAPE':>7} {'R²':>7}  " \
             f"{'h=26w':>6} {'MAPE':>7} {'R²':>7}"
    print('  ' + '─' * (len(header) - 2))
    for var in ['TFT', 'LightGBM_M4']:
        row = sub[sub['variant'] == var]
        if row.empty:
            continue
        parts = []
        for h in [1, 4, 13, 26]:
            r = row[row['horizon_weeks'] == h]
            if r.empty:
                parts += ['     N/A', '    N/A%', '    N/A']
            else:
                parts += [f"{r['RMSE'].iloc[0]:>7.0f}",
                          f"{r['MAPE'].iloc[0]:>6.1f}%",
                          f"{r['R2'].iloc[0]:>7.3f}"]
        print(f"  {var:<16} " + '  '.join(parts))
    print('  ' + '─' * (len(header) - 2))

# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6] Generating figures ...')

# ── Fig 1: TFT vs LightGBM M4 — R² and MAPE side by side ──
if lgbm_m4 is not None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('TFT vs LightGBM M4 — R² and MAPE by Horizon',
                 fontsize=13, fontweight='bold')

    for col, crop in enumerate(CROPS):
        tft_r  = tft_mean[tft_mean['crop'] == crop].set_index('horizon_weeks')
        lgb_r  = lgbm_m4[lgbm_m4['crop'] == crop].set_index('horizon_weeks')

        x  = np.arange(len(HORIZONS))
        w  = 0.35
        hs = [str(h) + 'w' for h in HORIZONS]

        # R² row
        ax = axes[0, col]
        tft_r2  = [tft_r.loc[h, 'R2']  if h in tft_r.index  else np.nan for h in HORIZONS]
        lgbm_r2 = [lgb_r.loc[h, 'R2']  if h in lgb_r.index  else np.nan for h in HORIZONS]
        ax.bar(x - w/2, tft_r2,  w, label='TFT',           color='#6c5ce7', alpha=0.85)
        ax.bar(x + w/2, lgbm_r2, w, label='LightGBM M4',   color=CROP_COLORS[crop], alpha=0.85)
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_xticks(x); ax.set_xticklabels(hs)
        ax.set_ylabel('R²') if col == 0 else None
        ax.set_title(crop.capitalize())
        if col == 0: ax.legend(fontsize=8)
        ax.set_ylim(-0.3, 1.0)

        # MAPE row
        ax = axes[1, col]
        tft_mp  = [tft_r.loc[h,  'MAPE'] if h in tft_r.index  else np.nan for h in HORIZONS]
        lgbm_mp = [lgb_r.loc[h,  'MAPE'] if h in lgb_r.index  else np.nan for h in HORIZONS]
        ax.bar(x - w/2, tft_mp,  w, label='TFT',         color='#6c5ce7', alpha=0.85)
        ax.bar(x + w/2, lgbm_mp, w, label='LightGBM M4', color=CROP_COLORS[crop], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(hs)
        ax.set_ylabel('MAPE (%)') if col == 0 else None
        ax.set_ylim(0, None)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig_tft_vs_lgbm.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {p}')

# ── Fig 2: Prediction intervals for onion ──
if onion_interval_store is not None:
    n_plot = min(80, len(onion_interval_store['y_true']))
    y_t = np.exp(np.array(onion_interval_store['y_true'][:n_plot]))
    y_p = np.exp(np.array(onion_interval_store['y_pred'][:n_plot]))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(n_plot), y_t, color='black', linewidth=1.2,
            label='Actual price', zorder=3)
    ax.plot(range(n_plot), y_p, color='#F4A261', linewidth=1.2,
            linestyle='--', label='TFT median (h=13w)', zorder=3)
    ax.fill_between(range(n_plot),
                    np.exp(np.array(onion_interval_store['y_lo'][:n_plot])),
                    np.exp(np.array(onion_interval_store['y_hi'][:n_plot])),
                    alpha=0.25, color='#F4A261', label='10th–90th pct interval')
    ax.set_xlabel('Test observation')
    ax.set_ylabel('Price (Rs/quintal)')
    ax.set_title('TFT Prediction Intervals — Onion, h=13w, Fold 4 (2025)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig_tft_intervals.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {p}')

# ── Fig 3: Variable importance from TFT encoder (last trained model) ──
try:
    # Get feature importance from variable selection weights
    # This uses the last trained TFT model (potato fold 4)
    interpretation = tft.interpret_output(
        raw_preds, reduction='sum')
    encoder_vars = interpretation.get('encoder_variables', None)
    if encoder_vars is not None:
        vi = encoder_vars.cpu().numpy()
        feat_names = (PAST_FEATURES + ['log_price'])[:len(vi)]
        top_n = 15
        sorted_idx = np.argsort(vi)[-top_n:][::-1]
        top_feats  = [feat_names[i] for i in sorted_idx]
        top_vals   = vi[sorted_idx]
        top_vals   = top_vals / top_vals.sum() * 100

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#6c5ce7' if 's2_' in f or 'modis_' in f
                  else '#00b894' if 'era5_' in f or 'chirps_' in f
                  else '#fdcb6e' if any(m in f for m in ['repo','wpi','diesel','usdinr','crude'])
                  else '#636e72' for f in top_feats]
        ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f.replace('_', ' ') for f in top_feats[::-1]], fontsize=9)
        ax.set_xlabel('Variable selection weight (%)')
        ax.set_title('TFT Encoder Variable Importance (Top 15)', fontweight='bold')
        from matplotlib.patches import Patch
        legend_els = [Patch(color='#6c5ce7', label='Satellite (S2/MODIS)'),
                      Patch(color='#00b894', label='Climate (ERA5/CHIRPS)'),
                      Patch(color='#fdcb6e', label='Macro-logistics'),
                      Patch(color='#636e72', label='Price/arrivals')]
        ax.legend(handles=legend_els, fontsize=8, loc='lower right')
        plt.tight_layout()
        p = os.path.join(OUT_DIR, 'fig_tft_importance.png')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {p}')
except Exception as e:
    print(f'  Variable importance figure skipped: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print(f'Script 17 complete. Total elapsed: {total_time/60:.1f} min\n')
print('Key outputs:')
for fname in ['tft_raw_results.csv', 'table_tft_vs_lgbm.csv',
              'fig_tft_vs_lgbm.png', 'fig_tft_intervals.png',
              'fig_tft_importance.png']:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        print(f'  {fname:<40} {os.path.getsize(fpath)/1024:>7.1f} KB')
print()
print('Next: Script 18 — Diebold-Mariano statistical tests')
