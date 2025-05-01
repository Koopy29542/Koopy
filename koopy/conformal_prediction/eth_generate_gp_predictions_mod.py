import os
import re
import numpy as np
from models.gp_predictor import GaussianProcessPredictor
from function import mathismathing

def main():
    prediction_len = 12
    history_len = 8

    dt = 0.1

    # model
    prediction_model = GaussianProcessPredictor(prediction_len=prediction_len, history_len=history_len-2, dt=dt)
    # load dataset
    idx_begin = 40
    idx_end = 160

    # build a calibration dataset

    pattern = r'biwi_eth.npy$'
    mathismathing(prediction_model,prediction_len,history_len,pattern,'./lobby2/biwi_eth')
    return

if __name__ == '__main__':
    main()