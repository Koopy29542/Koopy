import os
import re
import numpy as np
from ..models.gp_predictor import GaussianProcessPredictor
from ..utils.function import mathismathing

def main():
    prediction_len = 12
    history_len = 8

    dt = 0.1

    pattern = [r'biwi_eth.npy$',r'^.*\d{3}\.npy$',r'biwi_hotel.npy$',r'^.*\d{2}\.npy$',r'^.*\d{2}\.npy$']
    directories=  ['lobby2/biwi_eth','lobby2/univ','lobby2/biwi_hotel','lobby2/crowds_zara01','lobby2/crowds_zara02']

    # model
    for k,i in zip(pattern,directories):
        prediction_model = GaussianProcessPredictor(prediction_len=prediction_len, history_len=history_len-2, dt=dt)
        # build a calibration dataset

        mathismathing(prediction_model,prediction_len,history_len,k,i,namer='_gp_predictions.npy')
    return

if __name__ == '__main__':
    main()