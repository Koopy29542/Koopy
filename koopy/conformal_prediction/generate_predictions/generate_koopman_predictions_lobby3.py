import os
import re
import numpy as np
from models.koopy_predictor import KoopmanPredictor
import matplotlib.pyplot as plt
import time

def main():
    prediction_len = 12
    history_len = 8
    
    pattern = r'^\d+\.npy$'

    # model
    prediction_model = KoopmanPredictor(prediction_len=prediction_len,data_dir='../lobby3',pattern = r'^\d+\.npy$',dt = 0.1)
    #prediction_model.train(num_epochs=50, lr=1e-3, data_dir='lobby2/biwi_eth')
    #loaded_model = KoopmanPredictor.load_model('koopman_model_encoder_geo2.pkl')
    

    for type in ['test']:
        dirpath = os.path.join('../lobby3', type)
        for file in os.listdir(dirpath):
            if re.match(pattern, file):
                #filepath_to_load = os.path.join(dirpath, file)
                name, _ = os.path.splitext(file)
                filepath_to_save_predictions = os.path.join(dirpath, name + '_koopy_predictions.npy')
                filepath_to_save_targets = os.path.join(dirpath, name + '_targets.npy')
               # data = np.load(filepath_to_load)
               # time_steps, n_pedestrians, _ = data.shape
                ##target, prediction = prediction_model.testtraj2('lobby3/test')
                start_time = time.time()
                target, prediction = prediction_model.testtraj2('../lobby3/test')
                elapsed_time = time.time() - start_time
                print(f"Elapsed time for testtraj2: {elapsed_time:.6f} seconds")
                np.save(filepath_to_save_predictions, prediction)      # shape = (# effective time steps, prediction length, # pedestrians, 2)
                np.save(filepath_to_save_targets, target)
    return


if __name__ == '__main__':
    main()