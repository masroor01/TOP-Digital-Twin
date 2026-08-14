# -*- coding: utf-8 -*-
"""
Script 15c — LSTM + Transformer Comparison at M6 (Full Feature Set)
====================================================================
Extends Script 15b's model-choice comparison (LightGBM/XGBoost/CatBoost/
RandomForest) with two sequence models: a plain LSTM and a plain
(encoder-only) Transformer, both built fresh here in pure PyTorch --
deliberately NOT the existing Temporal Fusion Transformer (Script 17),
which is a separate, more complex architecture kept on its own deferred
track. Same M6 feature set, same rolling-origin CV folds/horizons/crops
as Scripts 15/15b (2 models x 4 folds x 4 horizons x 3 crops = 96 fits).

CPU-only (no CUDA available on this machine) -- kept deliberately small
and fast to fit within a reasonable runtime given that constraint:
  - LOOKBACK = 26 weeks of M6 features per sequence (fixed window)
  - Small models: 1-layer LSTM (hidden=64) / 1-layer Transformer encoder
    (d_model=64, 4 heads) -- this is NOT a claim that bigger wouldn't do
    better, just what's tractable on CPU in this run
  - Training sequences capped at MAX_TRAIN_SEQ per fit (random sample)
    to bound runtime -- without this, folds with hundreds of thousands
    of valid sequences would make full CV impractical on CPU
  - Early stopping on validation loss (patience=3), max 15 epochs

Outputs (Model_Output/):
  table_lstm_transformer_comparison.csv        all 96 rows
  table_lstm_transformer_comparison_mean.csv   mean across folds
  fig_lstm_transformer_comparison.png          R2/MAPE by model, per crop x horizon

Run: python scripts/15c_LSTM_Transformer_Comparison.py
     python scripts/15c_LSTM_Transformer_Comparison.py --pilot   (tiny sanity run)
"""

import io, os, sys, time, warnings, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true',
                     help='Tiny sanity run: 1 crop, 1 fold, 1 horizon, 2 epochs, capped sequences.')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONFIG (identical join to Scripts 15/15b)
# ─────────────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGM_FILE = os.path.join(BASE, 'data', 'agmarknet_weekly', 'top_weekly_panel.csv')
CMIE_FILE= os.path.join(BASE, 'data', 'cmie_macro',      'cmie_macro_2017_2025.csv')
RBI_FILE = os.path.join(BASE, 'data', 'rbi_dbie',        'rbi_dbie_macro_2017_2025.csv')
PPAC_FILE= os.path.join(BASE, 'data', 'ppac_macro',      'ppac_diesel_lpg_2017_2025.csv')
SAT_FILE = os.path.join(BASE, 'data', 'satellite_climate', 'crop_weekly_features.csv')
WAGE_FILE  = os.path.join(BASE, 'data', 'labour_wages',   'wage_agri_state_monthly.csv')
COLD_FILE  = os.path.join(BASE, 'data', 'infrastructure', 'cold_storage_by_state.csv')
ROAD_FILE  = os.path.join(BASE, 'data', 'infrastructure', 'road_density_state_annual.csv')
POLICY_FILE= os.path.join(BASE, 'data', 'policy_trade',   'policy_weekly_features.csv')
OUT_DIR  = os.path.join(BASE, 'Model_Output')
os.makedirs(OUT_DIR, exist_ok=True)

CROPS    = ['tomato', 'onion', 'potato'] if not args.pilot else ['tomato']
HORIZONS = [1, 4, 13, 26] if not args.pilot else [1]
SEED     = 42
LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINS = [4, 8, 13]

LOOKBACK = 26
MAX_TRAIN_SEQ = 2000 if args.pilot else 8_000
MAX_EPOCHS = 2 if args.pilot else 15
PATIENCE = 2 if args.pilot else 3
BATCH_SIZE = 256
DEVICE = torch.device('cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

FOLDS_ALL = [
    {'fold': 1, 'train_end': '2021-06-30',
     'val_start': '2021-07-01', 'val_end': '2021-12-31',
     'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'fold': 2, 'train_end': '2022-06-30',
     'val_start': '2022-07-01', 'val_end': '2022-12-31',
     'test_start': '2023-01-01', 'test_end': '2023-12-31'},
    {'fold': 3, 'train_end': '2023-06-30',
     'val_start': '2023-07-01', 'val_end': '2023-12-31',
     'test_start': '2024-01-01', 'test_end': '2024-12-31'},
    {'fold': 4, 'train_end': '2024-06-30',
     'val_start': '2024-07-01', 'val_end': '2024-12-31',
     'test_start': '2025-01-01', 'test_end': '2025-12-31'},
]
FOLDS = FOLDS_ALL[:1] if args.pilot else FOLDS_ALL

MODEL_COLORS = {'LSTM': '#5c7cfa', 'Transformer': '#e64980'}

print('=' * 65)
print(f'SCRIPT 15c: LSTM + TRANSFORMER COMPARISON AT M6{"  [PILOT MODE]" if args.pilot else ""}')
print('=' * 65)
print(f'  2 models x {len(FOLDS)} folds x {len(HORIZONS)} horizons x '
      f'{len(CROPS)} crops = {2*len(FOLDS)*len(HORIZONS)*len(CROPS)} fits\n')

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD + JOIN ALL LAYERS (identical to Scripts 15/15b)
# ─────────────────────────────────────────────────────────────────────────────
print('[1] Loading panel ...')
df = pd.read_csv(AGM_FILE, parse_dates=['week_start'])
df = df[(df['week_start'] >= '2017-01-01') & (df['week_start'] <= '2026-07-27')]
df = df.sort_values(['crop', 'market', 'week_start']).reset_index(drop=True)
df['year']  = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month
print(f'   Panel: {len(df):,} rows')

macro_dfs = []
for fpath in [CMIE_FILE, RBI_FILE, PPAC_FILE]:
    if os.path.exists(fpath):
        macro_dfs.append(pd.read_csv(fpath))
MACRO_COLS = []
if macro_dfs:
    macro = macro_dfs[0]
    for m in macro_dfs[1:]:
        macro = macro.merge(m, on=['year', 'month'], how='outer', suffixes=('', '_dup'))
        macro = macro[[c for c in macro.columns if not c.endswith('_dup')]]
    drop_cols = [c for c in ['date', 'date_x', 'date_y'] if c in macro.columns]
    df = df.merge(macro.drop(columns=drop_cols, errors='ignore'), on=['year', 'month'], how='left')
    MACRO_COLS = [c for c in macro.columns if c not in ('date', 'year', 'month')]
print(f'   Macro joined: {len(MACRO_COLS)} series')

print('[2] Loading satellite/climate features ...')
sat = pd.read_csv(SAT_FILE, parse_dates=['week_start'])
sat = sat.sort_values(['crop', 'week_start']).reset_index(drop=True)

ERA5_COLS   = ['era5_tmax', 'era5_tmin', 'era5_tmean', 'era5_dtr', 'era5_heat_35', 'era5_heat_38']
CHIRPS_COLS = ['chirps_rain_mm', 'chirps_rain_max', 'chirps_excess']
S2_COLS     = ['s2_ndvi', 's2_evi', 's2_valid_frac', 's2_ndvi_anom']
MODIS_COLS  = ['modis_ndvi', 'modis_evi', 'modis_lst_mean', 'modis_lst_max', 'modis_lst_frac35']

roll_specs = [
    ('era5_heat_35',   'sum', [4, 8]),
    ('chirps_rain_mm', 'sum', [4, 8]),
    ('s2_ndvi',        'mean', [4]),
    ('s2_ndvi_anom',   'mean', [4]),
    ('modis_lst_mean', 'mean', [4]),
]
roll_cols = []
for col, func, windows in roll_specs:
    if col not in sat.columns:
        continue
    for w in windows:
        new_col = f'{col}_roll{w}'
        agg = 'sum' if func == 'sum' else 'mean'
        sat[new_col] = (sat.groupby('crop')[col]
                           .transform(lambda x: getattr(x.shift(1).rolling(w, min_periods=2), agg)()))
        roll_cols.append(new_col)

CLIMATE_FEATS  = [c for c in ERA5_COLS + CHIRPS_COLS if c in sat.columns]
CLIMATE_FEATS += [c for c in roll_cols if any(s in c for s in ['era5_heat', 'chirps_rain'])]
SAT_FEATS      = [c for c in S2_COLS + MODIS_COLS if c in sat.columns]
SAT_FEATS     += [c for c in roll_cols if any(s in c for s in ['s2_', 'modis_'])]

join_cols = ['week_start', 'crop'] + CLIMATE_FEATS + SAT_FEATS
df = df.merge(sat[join_cols], on=['crop', 'week_start'], how='left')

print('[2b] Loading infrastructure + policy/trade layers ...')


def assert_unique(frame, keys, label):
    n_dup = frame[keys].duplicated().sum()
    if n_dup:
        raise ValueError(f'{label}: {n_dup} duplicate rows on {keys}')


INFRA_FEATS, POLICY_FEATS = [], []

if os.path.exists(WAGE_FILE):
    wages = pd.read_csv(WAGE_FILE)[['state', 'year', 'month', 'wage_agri_men', 'wage_agri_women']]
    assert_unique(wages, ['state', 'year', 'month'], 'wages')
    n0 = len(df)
    df = df.merge(wages, on=['state', 'year', 'month'], how='left')
    assert len(df) == n0
    INFRA_FEATS += ['wage_agri_men', 'wage_agri_women']

if os.path.exists(COLD_FILE):
    cold = pd.read_csv(COLD_FILE)[['state', 'n_facilities', 'capacity_mt']]
    cold = cold.rename(columns={'n_facilities': 'cold_storage_n_facilities',
                                 'capacity_mt': 'cold_storage_capacity_mt'})
    assert_unique(cold, ['state'], 'cold storage')
    n0 = len(df)
    df = df.merge(cold, on=['state'], how='left')
    assert len(df) == n0
    INFRA_FEATS += ['cold_storage_n_facilities', 'cold_storage_capacity_mt']

if os.path.exists(ROAD_FILE):
    road = pd.read_csv(ROAD_FILE)[['state', 'year', 'road_density_per_100_sqkm']]
    assert_unique(road, ['state', 'year'], 'road density')
    n0 = len(df)
    df = df.merge(road, on=['state', 'year'], how='left')
    assert len(df) == n0
    INFRA_FEATS += ['road_density_per_100_sqkm']

if os.path.exists(POLICY_FILE):
    policy = pd.read_csv(POLICY_FILE, parse_dates=['week_start'])
    policy_cols = ['export_banned', 'mep_usd_per_tonne', 'export_duty_pct',
                   'market_intervention_flag', 'operation_greens_active']
    policy = policy[['crop', 'week_start'] + policy_cols]
    assert_unique(policy, ['crop', 'week_start'], 'policy')
    n0 = len(df)
    df = df.merge(policy, on=['crop', 'week_start'], how='left')
    assert len(df) == n0
    POLICY_FEATS += policy_cols

print(f'   Infrastructure features: {len(INFRA_FEATS)}  |  Policy features: {len(POLICY_FEATS)}')

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING (identical to Scripts 15/15b)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] Engineering features ...')


def build_features(df_in):
    # FIXED 2026-08-14 (full-layer audit): grouping/sorting used to key on
    # 'market' NAME, not market_id -- see Script 23/15's commits for the
    # full discovery story.
    out = {}
    for crop in CROPS:
        sub = df_in[df_in['crop'] == crop].copy()
        sub = sub.sort_values(['market_id', 'week_start'])
        sub['log_price'] = np.log1p(sub['modal_price_weighted'])

        for lag in LAG_WEEKS:
            sub[f'price_lag_{lag}'] = sub.groupby('market_id')['log_price'].shift(lag)
        for w in ROLL_WINS:
            g = sub.groupby('market_id')['log_price']
            sub[f'price_roll_mean_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).mean())
            sub[f'price_roll_std_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=2).std())

        if 'arrivals_tonnes_week' in sub.columns:
            sub['log_arr'] = np.log1p(sub['arrivals_tonnes_week'].clip(lower=0))
            for lag in [1, 2, 4]:
                sub[f'arr_lag_{lag}'] = sub.groupby('market_id')['log_arr'].shift(lag)
            for w in [4, 8]:
                sub[f'arr_roll_mean_{w}'] = sub.groupby('market_id')['log_arr'].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=2).mean())

        sub['price_yoy'] = sub.groupby('market_id')['log_price'].shift(52)
        sub['week_num'] = sub['week_start'].dt.isocalendar().week.astype(int)
        sub['sin_week'] = np.sin(2 * np.pi * sub['week_num'] / 52)
        sub['cos_week'] = np.cos(2 * np.pi * sub['week_num'] / 52)
        sub['sin2_week'] = np.sin(4 * np.pi * sub['week_num'] / 52)
        sub['cos2_week'] = np.cos(4 * np.pi * sub['week_num'] / 52)

        m = sub['week_start'].dt.month
        if crop == 'tomato':
            sub['season_peak_arrival'] = m.isin([11, 12, 1, 2]).astype(int)
            sub['season_lean']         = m.isin([5, 6, 7]).astype(int)
            sub['season_kharif']       = m.isin([8, 9, 10]).astype(int)
        elif crop == 'onion':
            sub['season_rabi_arrival'] = m.isin([2, 3, 4, 5]).astype(int)
            sub['season_lean']         = m.isin([9, 10, 11]).astype(int)
            sub['season_kharif']       = m.isin([8, 9]).astype(int)
        elif crop == 'potato':
            sub['season_harvest']      = m.isin([2, 3, 4]).astype(int)
            sub['season_storage']      = m.isin([5, 6, 7, 8, 9]).astype(int)
            sub['season_lean']         = m.isin([10, 11]).astype(int)

        if 'market_id' in sub.columns:
            sub['market_enc'] = pd.Categorical(sub['market_id']).codes
        for col in ['state']:
            if col in sub.columns:
                sub[f'{col}_enc'] = pd.Categorical(sub[col]).codes
        sub['year_trend'] = sub['week_start'].dt.year - 2017
        out[crop] = sub
    return out


feat = build_features(df)

PRICE_FEATS = (
    [f'price_lag_{lag}' for lag in LAG_WEEKS] +
    [f'price_roll_mean_{w}' for w in ROLL_WINS] +
    [f'price_roll_std_{w}' for w in ROLL_WINS] +
    ['price_yoy', 'sin_week', 'cos_week', 'sin2_week', 'cos2_week',
     'week_num', 'year_trend', 'market_enc', 'state_enc',
     'season_peak_arrival', 'season_lean', 'season_kharif',
     'season_rabi_arrival', 'season_harvest', 'season_storage']
)
ARR_FEATS = (
    ['log_arr'] + [f'arr_lag_{lag}' for lag in [1, 2, 4]] + [f'arr_roll_mean_{w}' for w in [4, 8]]
)
M6_FEATS = PRICE_FEATS + ARR_FEATS + MACRO_COLS + CLIMATE_FEATS + SAT_FEATS + INFRA_FEATS + POLICY_FEATS

for crop in CROPS:
    available = [c for c in M6_FEATS if c in feat[crop].columns]
    print(f'   {crop:8s}: {len(feat[crop]):>8,} rows  |  M6 features available: {len(available)}/{len(M6_FEATS)}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. SEQUENCE CONSTRUCTION
# Each sample: LOOKBACK weeks of M6 features for one market, ending at week
# t (the "as-of" week), target = log_price at week t+h. Sequences whose
# lookback window would cross before the market's own history start just
# get zero-padded at the front (mask not used -- simplicity over full
# padding-awareness, consistent with the "fillna(0)" convention already
# used for missing tabular features in Scripts 15/15b).
# ─────────────────────────────────────────────────────────────────────────────
def build_sequences(df_crop, fcols, h, date_lo, date_hi, max_n=None):
    """Returns (X, y) where X: (n, LOOKBACK, n_features), y: (n,) log-price target,
    for as-of weeks within [date_lo, date_hi] (inclusive)."""
    # FIXED 2026-08-14: grouped by 'market' NAME, not market_id -- same
    # collision bug as Scripts 15/15b/23. Here it's worse than a feature
    # value being wrong: g.sort_values('week_start') on a combined group
    # would interleave two DIFFERENT markets' weeks into one sequence
    # window, feeding the LSTM/Transformer a physically nonsensical
    # lookback (part Haryana, part UP for "Fatehabad APMC").
    Xs, ys = [], []
    for market_id, g in df_crop.groupby('market_id'):
        g = g.sort_values('week_start').reset_index(drop=True)
        vals = g[fcols].fillna(0).to_numpy(dtype=np.float32)
        log_price = g['log_price'].to_numpy(dtype=np.float32)
        dates = g['week_start'].to_numpy()

        n = len(g)
        for i in range(n):
            if dates[i] < date_lo.to_datetime64() or dates[i] > date_hi.to_datetime64():
                continue
            j = i + h
            if j >= n or not np.isfinite(log_price[j]):
                continue
            lo = max(0, i - LOOKBACK + 1)
            window = vals[lo:i + 1]
            if len(window) < LOOKBACK:
                pad = np.zeros((LOOKBACK - len(window), window.shape[1]), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)
            Xs.append(window)
            ys.append(log_price[j])

    if not Xs:
        return np.empty((0, LOOKBACK, len(fcols)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)

    if max_n is not None and len(X) > max_n:
        idx = np.random.RandomState(SEED).choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
    return X, y


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODELS
# ─────────────────────────────────────────────────────────────────────────────
class SimpleLSTM(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class SimpleTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, dim_ff=128):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        pos = torch.zeros(LOOKBACK, d_model)
        position = torch.arange(0, LOOKBACK, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                            batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.proj(x) + self.pos_enc
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def compute_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, R2=np.nan, N=0)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(RMSE=round(rmse, 1), MAE=round(mae, 1), MAPE=round(mape, 2), R2=round(r2, 4), N=len(yt))


def train_eval(model_name, n_features, train_ds, val_ds, X_te, y_te):
    model = (SimpleLSTM(n_features) if model_name == 'LSTM' else SimpleTransformer(n_features)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val, best_state, patience_ct = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = np.mean(val_losses) if val_losses else np.inf

        if val_loss < best_val - 1e-5:
            best_val, best_state, patience_ct = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te).to(DEVICE)
        y_pred = model(X_te_t).cpu().numpy()
    return y_pred, epoch + 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[4] Running LSTM + Transformer comparison at M6 (lookback={LOOKBACK}w, '
      f'max_train_seq={MAX_TRAIN_SEQ:,}, max_epochs={MAX_EPOCHS}) ...')
MODELS = ['LSTM', 'Transformer']
all_rows = []
t0_total = time.time()

for model_name in MODELS:
    print(f'\n  == {model_name} ==')
    m_t0 = time.time()
    for crop in CROPS:
        df_crop = feat[crop].copy()
        fcols = [c for c in M6_FEATS if c in df_crop.columns]
        n_features = len(fcols)

        for fold_info in FOLDS:
            fold = fold_info['fold']
            t_end = pd.Timestamp(fold_info['train_end'])
            v_start = pd.Timestamp(fold_info['val_start'])
            v_end = pd.Timestamp(fold_info['val_end'])
            te_start = pd.Timestamp(fold_info['test_start'])
            te_end = pd.Timestamp(fold_info['test_end'])

            for h in HORIZONS:
                t0 = time.time()

                X_tr, y_tr = build_sequences(df_crop, fcols, h, pd.Timestamp('2017-01-01'), t_end,
                                              max_n=MAX_TRAIN_SEQ)
                X_va, y_va = build_sequences(df_crop, fcols, h, v_start, v_end, max_n=MAX_TRAIN_SEQ // 4)
                X_te, y_te = build_sequences(df_crop, fcols, h, te_start, te_end)

                if len(X_tr) < 100 or len(X_te) < 10 or len(X_va) < 10:
                    print(f'    {crop:8s} fold{fold} h={h:>2d}w  SKIPPED (too few sequences: '
                          f'train={len(X_tr)}, val={len(X_va)}, test={len(X_te)})')
                    continue

                # Standardize features using TRAIN-only stats (fit on train,
                # applied to val/test) -- trees don't need this but NN
                # convergence within a tight epoch budget depends heavily on
                # it; without it the pilot run never learned past a near-
                # constant prediction (R2 << 0 after 2 epochs).
                feat_mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
                feat_std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0)
                feat_std[feat_std < 1e-6] = 1.0
                X_tr = (X_tr - feat_mean) / feat_std
                X_va = (X_va - feat_mean) / feat_std
                X_te = (X_te - feat_mean) / feat_std

                train_ds, val_ds = SeqDataset(X_tr, y_tr), SeqDataset(X_va, y_va)
                y_pred, epochs_run = train_eval(model_name, n_features, train_ds, val_ds, X_te, y_te)
                m = compute_metrics(y_te, y_pred)
                elapsed = round(time.time() - t0, 1)

                all_rows.append(dict(model=model_name, crop=crop, fold=fold, horizon_weeks=h,
                                      epochs=epochs_run, elapsed_s=elapsed,
                                      n_train=len(X_tr), n_test=len(X_te), **m))
                print(f'    {crop:8s} fold{fold} h={h:>2d}w  '
                      f'RMSE={m["RMSE"]:>8.1f}  MAPE={m["MAPE"]:>6.2f}%  R2={m["R2"]:>7.4f}  '
                      f'epochs={epochs_run:>2d}  n_train={len(X_tr):>6d}  ({elapsed:.1f}s)')
    print(f'  {model_name} total time: {(time.time()-m_t0)/60:.1f} min')

print(f'\nTotal runtime: {(time.time()-t0_total)/60:.1f} min')

results = pd.DataFrame(all_rows)
suffix = '_pilot' if args.pilot else ''
results.to_csv(os.path.join(OUT_DIR, f'table_lstm_transformer_comparison{suffix}.csv'), index=False)

if not results.empty:
    mean_tbl = (results.groupby(['model', 'crop', 'horizon_weeks'])[['RMSE', 'MAE', 'MAPE', 'R2']]
                .mean().round(3).reset_index())
    mean_tbl.to_csv(os.path.join(OUT_DIR, f'table_lstm_transformer_comparison_mean{suffix}.csv'), index=False)
    print('\n=== Mean R2 by model x crop x horizon ===')
    print(mean_tbl.pivot_table(index=['crop', 'horizon_weeks'], columns='model', values='R2').to_string())

    if not args.pilot:
        print('\n[5] Generating comparison figure ...')
        fig, axes = plt.subplots(len(CROPS), 2, figsize=(13, 11))
        for i, crop in enumerate(CROPS):
            sub = mean_tbl[mean_tbl['crop'] == crop]
            ax_r2, ax_mape = axes[i, 0], axes[i, 1]
            for model_name in MODELS:
                s = sub[sub['model'] == model_name].sort_values('horizon_weeks')
                if s.empty:
                    continue
                ax_r2.plot(s['horizon_weeks'], s['R2'], 'o-', label=model_name, color=MODEL_COLORS[model_name])
                ax_mape.plot(s['horizon_weeks'], s['MAPE'], 'o-', label=model_name, color=MODEL_COLORS[model_name])
            ax_r2.set_title(f'{crop.capitalize()} — R² by horizon')
            ax_r2.set_xlabel('Horizon (weeks)')
            ax_r2.set_ylabel('R²')
            ax_r2.axhline(0, color='gray', lw=0.5, ls='--')
            ax_r2.legend(fontsize=8)
            ax_mape.set_title(f'{crop.capitalize()} — MAPE by horizon')
            ax_mape.set_xlabel('Horizon (weeks)')
            ax_mape.set_ylabel('MAPE (%)')
            ax_mape.legend(fontsize=8)
        plt.suptitle('LSTM vs Transformer Comparison at M6 (Full Feature Set)', fontsize=14, y=1.0)
        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, 'fig_lstm_transformer_comparison.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {fig_path}')
else:
    print('\nNo results produced.')

print('\nScript 15c complete.')
