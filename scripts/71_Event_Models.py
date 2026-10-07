# -*- coding: utf-8 -*-
"""
Script 71 -- Decision-focused event models: spike / crash classifiers (Track B, Arm D2)
=======================================================================================
Arm D (Script 70) found that regression forecasts rank spike risk well (AUC 0.80-0.90) but the six-layer
model adds almost nothing over the price-only model for the decision. This script trains models whose TARGET IS
THE EVENT, with and without the data layers, so the question "do the layers help the decision?" can be asked
properly.  Protocol frozen before scoring: docs/EVENT_MODELS_PROTOCOL.md.

Events (market-week, origin t, horizon h):  spike = price(t+h)/price(t) - 1 >= +30 %;  crash = ... <= -25 %.
Two feature sets, identical in every other respect:
  M0E  price-only: Script 15's M0 features + price-derived event features (momentum 4/13/26 wk, deviation from the
       52-wk mean, 13-wk z-score, national-median momentum, share of markets already rising, market vs national level);
  LAY  M0E + the layers: Script 15's M6 features + drought + ONI + layer-derived anomalies (arrivals shortfall,
       rainfall and heat anomalies, diesel change).
Training: Script 15's rolling-origin folds (train/validation rows selected by TARGET date, test = next calendar
year), LightGBM binary classifier, early stopping on validation log-loss, isotonic calibration on the validation
rows, 5 seeds (probabilities averaged).  No threshold or hyperparameter is chosen on a test year.

Usage:  python scripts/71_Event_Models.py <h> [crop ...]      (default crops: tomato onion potato)
Output: Model_Output/_event_runs/event_probs_h<h>_<crop>.parquet   (git-ignored)
"""
import io, os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'Model_Output', '_event_runs')
SEEDS = [42, 43, 44, 45, 46]
UP, DN = 0.30, -0.25
PARAMS = dict(objective='binary', metric='binary_logloss', n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1)


def load_features():
    sys.path.insert(0, os.path.join(BASE, 'scripts'))
    import gpu_utils
    gpu_utils._cache['lgbm'] = {}
    path = os.path.join(BASE, 'scripts', '15_Ablation_Study_M0_M4.py')
    src = io.open(path, encoding='utf-8').read()
    cut = src.rfind('print(', 0, src.index('Running ablation: M0'))
    g = {'__name__': '__not_main__', '__file__': path}
    exec(compile(src[:cut], path, 'exec'), g)
    return g


def add_event_features(df):
    """Past-only engineered features. Returns (df, price_derived_cols, layer_derived_cols)."""
    df = df.sort_values(['market_id', 'week_start']).copy()
    lp = df['log_price']
    gm = df.groupby('market_id')
    df['ret4'] = lp - df['price_lag_4']; df['ret13'] = lp - df['price_lag_13']; df['ret26'] = lp - df['price_lag_26']
    df['dev52'] = lp - gm['log_price'].transform(lambda s: s.rolling(52, min_periods=26).mean())
    df['z13'] = (lp - df['price_roll_mean_13']) / df['price_roll_std_13'].replace(0, np.nan)
    wk = df.groupby('week_start')
    df['nat_ret4'] = wk['ret4'].transform('median'); df['nat_ret13'] = wk['ret13'].transform('median')
    df['share_rising13'] = wk['ret13'].transform(lambda s: (s > 0.2).mean())
    df['mkt_vs_nat'] = lp - wk['log_price'].transform('median')
    price_cols = ['ret4', 'ret13', 'ret26', 'dev52', 'z13', 'nat_ret4', 'nat_ret13', 'share_rising13', 'mkt_vs_nat']
    # layer-derived (past-only)
    df['arr_anom52'] = df['log_arr'] - gm['log_arr'].transform(lambda s: s.rolling(52, min_periods=26).mean())
    df['arr_chg4'] = df['log_arr'] - df['arr_lag_4']
    df['rain8_anom'] = df['chirps_rain_mm_roll8'] - gm['chirps_rain_mm_roll8'].transform(lambda s: s.rolling(104, min_periods=52).mean())
    df['heat8_anom'] = df['era5_heat_35_roll8'] - gm['era5_heat_35_roll8'].transform(lambda s: s.rolling(104, min_periods=52).mean())
    df['diesel_chg13'] = df['diesel_delhi_per_L'] / gm['diesel_delhi_per_L'].shift(13) - 1
    layer_cols = ['arr_anom52', 'arr_chg4', 'rain8_anom', 'heat8_anom', 'diesel_chg13']
    return df, price_cols, layer_cols


def main():
    h = int(sys.argv[1]); crops = sys.argv[2:] or ['tomato', 'onion', 'potato']
    os.makedirs(OUT, exist_ok=True)
    g = load_features(); feat = g['feat']; FS = g['MODEL_FEATURE_SETS']; FOLDS = g['FOLDS']
    for crop in crops:
        df, price_cols, layer_cols = add_event_features(feat[crop])
        m0 = [c for c in FS['M0'] if c in df.columns]
        m6 = [c for c in FS['M6'] if c in df.columns]
        extra = [c for c in (g['DROUGHT_FEATS'] + g['ONI_FEATS']) if c in df.columns]
        sets = {'M0E': m0 + price_cols, 'LAY': list(dict.fromkeys(m0 + price_cols + m6 + extra + layer_cols))}
        print('[71] %s h=%d | M0E %d features, LAY %d features' % (crop, h, len(sets['M0E']), len(sets['LAY'])), flush=True)
        df['target'] = df.groupby('market_id')['log_price'].shift(-h)
        df = df.dropna(subset=['target', 'price_lag_1'])
        df['ret_true'] = np.expm1(df['target']) / np.expm1(df['log_price']) - 1
        rows = []
        for fi in FOLDS:
            t_end, v_s, v_e = pd.Timestamp(fi['train_end']), pd.Timestamp(fi['val_start']), pd.Timestamp(fi['val_end'])
            te_s, te_e = pd.Timestamp(fi['test_start']), pd.Timestamp(fi['test_end'])
            tr = df[df['week_start'] + pd.Timedelta(weeks=h) <= t_end]
            va = df[(df['week_start'] + pd.Timedelta(weeks=h)).between(v_s, v_e)]
            te = df[(df['week_start'] >= te_s) & (df['week_start'] <= te_e)]
            if len(tr) < 200 or len(te) < 50: continue
            for side, ev_fn in (('spike', lambda r: r >= UP), ('crash', lambda r: r <= DN)):
                ytr, yva, yte = ev_fn(tr['ret_true']).astype(int).to_numpy(), ev_fn(va['ret_true']).astype(int).to_numpy(), ev_fn(te['ret_true']).astype(int).to_numpy()
                if ytr.sum() < 30 or yva.sum() < 10: continue
                out = te[['market_id', 'week_start']].copy(); out['crop'] = crop; out['horizon_weeks'] = h; out['fold'] = fi['fold']; out['side'] = side; out['event'] = yte
                for name, cols in sets.items():
                    t0 = time.time(); pc, pr = [], []
                    Xtr, Xva, Xte = tr[cols].fillna(0), va[cols].fillna(0), te[cols].fillna(0)
                    for s in SEEDS:
                        mdl = lgb.LGBMClassifier(**PARAMS, random_state=s)
                        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                        pv = mdl.predict_proba(Xva)[:, 1]; pt = mdl.predict_proba(Xte)[:, 1]
                        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip').fit(pv, yva)
                        pc.append(iso.predict(pt)); pr.append(pt)
                    out['p_cal_' + name] = np.mean(pc, axis=0); out['p_raw_' + name] = np.mean(pr, axis=0)
                    auc = roc_auc_score(yte, out['p_raw_' + name]) if 0 < yte.sum() < len(yte) else np.nan
                    print('   %s h=%d fold%d %-5s %-4s | AUC %.3f  base-rate test %.3f  [%.0fs]' % (crop, h, fi['fold'], side, name, auc, yte.mean(), time.time() - t0), flush=True)
                rows.append(out)
        pd.concat(rows, ignore_index=True).to_parquet(os.path.join(OUT, 'event_probs_h%d_%s.parquet' % (h, crop)), index=False)
        print('[71] saved %s h=%d' % (crop, h), flush=True)


if __name__ == '__main__':
    main()
