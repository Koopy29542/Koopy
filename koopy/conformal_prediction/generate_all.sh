#!/usr/bin/env bash
set -euo pipefail

# 1) data_formatting
(
  cd data_formatting
  python txt2npy.py
)


# 3) ETHUCY (all) generators + moveUNIV
(
  cd generate_predictions
  python generate_gp_predictions_all.py

  python generate_linear_predictions_all.py
  python generate_koopman_predictions_all.py
  python moveUNIV.py
)
# 2) Lobby3 generators
(
  cd generate_predictions
  python generate_gp_predictions_lobby3.py

  python generate_linear_predictions_lobby3.py
  python generate_koopman_predictions_lobby3.py
  python generate_koopman_predictions_clu_geo_lobby3.py
)

