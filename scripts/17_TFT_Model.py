# -*- coding: utf-8 -*-
"""
Script 17 — Temporal Fusion Transformer (TFT) Secondary Model
==============================================================
Trains a TFT as a secondary forecasting model alongside LightGBM (Script 12).
Provides attention-based interpretability and temporal pattern decomposition
as a robustness check for the ablation results from Script 15.

Inputs:
  data/agmarknet_weekly/top_weekly_panel.csv    — Agmarknet weekly panel
  data/satellite_climate/crop_weekly_features.csv — satellite + climate (Script 14)
  data/cmie_macro/cmie_macro_2017_2025.csv
  data/rbi_dbie/rbi_dbie_macro_2017_2025.csv
  data/ppac_macro/ppac_diesel_lpg_2017_2025.csv

Outputs (Model_Output/):
  tft_raw_results.csv           — per-fold, per-horizon, per-crop metrics
  table_tft_vs_lgbm.csv         — comparison table TFT vs LightGBM M4
  fig_tft_attention.png         — temporal self-attention weights (onion, h=26w)
  fig_tft_vs_lgbm.png           — side-by-side R² comparison

CV: same 4-fold rolling-origin as Script 15 (test years 2022/2023/2024/2025)
Horizons: h = 1, 4, 13, 26 weeks
Crops: tomato, onion, potato

Run: python scripts/17_TFT_Model.py
Status: PENDING — to be implemented
"""

# TODO: implement TFT model
# Candidate libraries:
#   pytorch-forecasting  (PyTorch-based, native TFT implementation)
#   neuralforecast        (NeuralForecast TFT, simpler API)
#
# Install (choose one):
#   pip install pytorch-forecasting lightning
#   pip install neuralforecast

raise NotImplementedError(
    "Script 17 is not yet implemented.\n"
    "See docstring above for planned inputs, outputs, and architecture."
)
