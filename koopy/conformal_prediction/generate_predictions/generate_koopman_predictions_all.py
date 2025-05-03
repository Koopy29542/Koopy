import os
import re
import numpy as np
from ..models.koopy_justmul import KoopmanPredictor


def main():
    prediction_len = 12
    history_len = 8


    pattern = [r'biwi_eth.npy$',r'students001.npy$',r'students003.npy$',r'biwi_hotel.npy$',r'^.*\d{2}\.npy$',r'^.*\d{2}\.npy$']
    directories=  ['lobby2/biwi_eth','lobby2/univ','lobby2/univ','lobby2/biwi_hotel','lobby2/crowds_zara01','lobby2/crowds_zara02']
    test_dirpaths = ['lobby2/biwi_eth/test','lobby2/univ/test','lobby2/univ/test','lobby2/biwi_hotel/test','lobby2/crowds_zara01/test','lobby2/crowds_zara02/test']
    for i,k,j in zip(pattern,directories,test_dirpaths):
    # model
        print(k)
        prediction_model = KoopmanPredictor(prediction_len=prediction_len,data_dir=k,pattern = i)
        #prediction_model.train(num_epochs=100, lr=1e-3, data_dir= k)

        for type in ['test']:
            dirpath = os.path.join(k, type)
            for file in os.listdir(dirpath):
                if re.match(i, file):
                    #filepath_to_load = os.path.join(dirpath, file)
                    name, _ = os.path.splitext(file)
                    filepath_to_save_predictions = os.path.join(dirpath, name + '_mul_predictions.npy')
                    #filepath_to_save_targets = os.path.join(dirpath, name + '_targets.npy')
                # data = np.load(filepath_to_load)
                # time_steps, n_pedestrians, _ = data.shape

                    target,prediction=prediction_model.testtraj2(j)
                    # idx, prediction_len, number of agents, 2
                    prediction=prediction
                    #target= target
                    np.save(filepath_to_save_predictions, prediction)      # shape = (# effective time steps, prediction length, # pedestrians, 2)
                    #np.save(filepath_to_save_targets, target)
    return


if __name__ == '__main__':
    main()