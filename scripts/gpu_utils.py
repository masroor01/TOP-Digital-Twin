# -*- coding: utf-8 -*-
"""
gpu_utils — shared NVIDIA GPU auto-detection for the TOP Digital Twin scripts.

Every model script (LightGBM, XGBoost, CatBoost, PyTorch) imports this
module and asks it for device params instead of hardcoding CPU/GPU. Each
`*_gpu_params()` function does a real trial fit/alloc on a few rows of
dummy data (availability flags like torch.cuda.is_available() say a CUDA
driver exists, not that this specific library's wheel was built with GPU
support) and silently falls back to CPU params if that trial raises.
Results are cached per-process so the trial only runs once per backend.

Run this file directly to print a summary of what each backend detected
on the current machine.
"""

import warnings

_cache = {}


def _log(msg):
    print(f"[gpu_utils] {msg}")


def lgbm_gpu_params(verbose=True):
    """Return LightGBM constructor kwargs that select the GPU when usable."""
    if 'lgbm' in _cache:
        return _cache['lgbm']

    import numpy as np
    import lightgbm as lgb

    X = np.random.rand(64, 4)
    y = np.random.rand(64)
    params = {}
    for device_type in ('cuda', 'gpu'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lgb.LGBMRegressor(device_type=device_type, n_estimators=2,
                                   verbosity=-1).fit(X, y)
            params = {'device_type': device_type}
            if verbose:
                _log(f"LightGBM: using GPU (device_type='{device_type}')")
            break
        except Exception:
            continue
    else:
        if verbose:
            _log("LightGBM: GPU not usable (driver/build unavailable) -- using CPU")

    _cache['lgbm'] = params
    return params


def xgb_gpu_params(verbose=True):
    """Return XGBoost constructor kwargs that select the GPU when usable."""
    if 'xgb' in _cache:
        return _cache['xgb']

    import numpy as np
    import xgboost as xgb

    X = np.random.rand(64, 4)
    y = np.random.rand(64)
    params = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            xgb.XGBRegressor(device='cuda', tree_method='hist', n_estimators=2,
                              verbosity=0).fit(X, y)
        params = {'device': 'cuda', 'tree_method': 'hist'}
        if verbose:
            _log("XGBoost: using GPU (device='cuda')")
    except Exception:
        if verbose:
            _log("XGBoost: GPU not usable (driver/build unavailable) -- using CPU")

    _cache['xgb'] = params
    return params


def catboost_gpu_params(verbose=True):
    """Return CatBoost constructor kwargs that select the GPU when usable."""
    if 'catboost' in _cache:
        return _cache['catboost']

    import numpy as np
    from catboost import CatBoostRegressor

    X = np.random.rand(64, 4)
    y = np.random.rand(64)
    params = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            CatBoostRegressor(task_type='GPU', iterations=2,
                               verbose=False).fit(X, y)
        params = {'task_type': 'GPU'}
        if verbose:
            _log("CatBoost: using GPU (task_type='GPU')")
    except Exception:
        if verbose:
            _log("CatBoost: GPU not usable (driver/build unavailable) -- using CPU")

    _cache['catboost'] = params
    return params


def torch_device(verbose=True):
    """Return a torch.device pointing at the GPU when usable, else CPU."""
    if 'torch' in _cache:
        return _cache['torch']

    import torch

    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            _log(f"PyTorch: using GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        if verbose:
            _log("PyTorch: CUDA not available -- using CPU")

    _cache['torch'] = device
    return device


if __name__ == '__main__':
    lgbm_gpu_params()
    xgb_gpu_params()
    catboost_gpu_params()
    torch_device()
