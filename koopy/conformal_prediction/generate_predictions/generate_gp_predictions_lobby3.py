import os
import re
import numpy as np
from models.gp_predictor import GaussianProcessPredictor
from utils.function import mathismathing 

def main():
    prediction_len = 12
    history_len = 8
    
    pattern = r'^\d+\.npy$'

    # model
    prediction_model = GaussianProcessPredictor(prediction_len=prediction_len, history_len= history_len-2, dt = 0.1)
    # load dataset
    for type in ['test']:
        dirpath = os.path.join('../lobby3', type)
        for file in os.listdir(dirpath):
            if re.match(pattern, file):
               #filepath_to_load = os.path.join(dirpath, file)
                name, _ = os.path.splitext(file)
               # data = np.load(filepath_to_load)
               # time_steps, n_pedestrians, _ = data.shape
                mathismathing(prediction_model,prediction_len,history_len,pattern,'../lobby3',namer='_gp_predictions.npy')
    return


if __name__ == '__main__':
    main()