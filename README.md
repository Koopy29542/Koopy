# Koopy
Official implementation of Koopy.
Python version 3.13.
## Installation

Clone the repo:

```bash
git clone https://github.com/Koopy29542/Koopy.git
cd Koopy
pip install -r requirements.txt
```

The original datasets generated during our experimentations have been uploaded to here:[Drive](https://drive.google.com/file/d/1HvLqsX4YWHW1jplOqJwZSbFtcbqBBnhw/view?usp=sharing). Please replace the lobby2 and lobby3 folders within the koopy folder when utilizing these datasets.

We have placed the necessary data generation methods for the linear, gp, and Koopman methods within the koopy/conformal_prediction/generate_predictions folder. For easy use, please run the following code:
```
cd koopy
cd conformal_prediction
bash ./generate_all.sh
```
Comparison results utilized within the paper, including the relevant images, can be generated through the related Python files in the compare_predictions folder.



*models explanation

koopy: training and testing at once

koopman_clu_geo.py(koopy-c): training and testing at once

train_koopy.py: training koopy with Lobby datasets > generate koopy.pkl file
koopy.pkl: encoder + Koopman matrix information saved
koopy_predictor.py: use saved koopy.pkl file for prediction without training

train_geo_clu.py
koopman_model_clu_geo.pkl
koopman_predictor_clu_geo.py

*prediction generation

ETH/UCY prediction generation: import koopy (both training & testing at once) / koopman_clu_geo
Lobby prediction generation: import koopy_predictor (after train_koopy.py) / koopman_predictor_clu_geo for fast inference without training phase
For comparison of ADE/FDE with video: Refer to compare_preditions_{dataset}

*ACP-MPC
Control performance metrics: acp_mpc_ethucy_statistics.py
Videos for conformal predictions: conformal_prediction_navi_{datasets}

