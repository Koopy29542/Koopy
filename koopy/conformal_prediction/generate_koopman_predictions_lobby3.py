import os
import re
import numpy as np
from models.koopman_predictor_clu_geo import KoopmanPredictor
import matplotlib.pyplot as plt
import time

def main():
    prediction_len = 12
    history_len = 8
    
    pattern = r'^\d+\.npy$'

    # model
    prediction_model = KoopmanPredictor(prediction_len=prediction_len,data_dir='lobby3',pattern = r'^\d+\.npy$',dt = 0.1)
    #prediction_model.train(num_epochs=50, lr=1e-3, data_dir='lobby2/biwi_eth')
    #loaded_model = KoopmanPredictor.load_model('koopman_model_encoder_geo2.pkl')
    

    for type in ['test']:
        dirpath = os.path.join('./lobby3', type)
        for file in os.listdir(dirpath):
            if re.match(pattern, file):
                #filepath_to_load = os.path.join(dirpath, file)
                name, _ = os.path.splitext(file)
                filepath_to_save_predictions = os.path.join(dirpath, name + '_koopman_clu_geo_predictions.npy')
                filepath_to_save_targets = os.path.join(dirpath, name + '_targets.npy')
               # data = np.load(filepath_to_load)
               # time_steps, n_pedestrians, _ = data.shape
                ##target, prediction = prediction_model.testtraj2('lobby3/test')
                start_time = time.time()
                target, prediction = prediction_model.testtraj2('lobby3/test')
                elapsed_time = time.time() - start_time
                print(f"Elapsed time for testtraj2: {elapsed_time:.6f} seconds")

                print(prediction.shape)

                file_path_gp = "/home/snowhan1021/jungjin/conformal_prediction/lobby3/test/0_gp_predictions.npy"
                file_path_traj = "/home/snowhan1021/jungjin/conformal_prediction/lobby3/test/0_trajectron_predictions.npy"
                file_path_linear = "/home/snowhan1021/jungjin/conformal_prediction/lobby3/test/0_linear_predictions.npy"
                file_path_eigen = "/home/snowhan1021/jungjin/conformal_prediction/lobby3/test/0_eigen_predictions.npy"

                trajectron_prediction = np.load(file_path_traj)
                gp_prediction = np.load(file_path_gp)
                linear_prediction = np.load(file_path_linear)
                eigen_prediction = np.load(file_path_eigen)

                error_eigen = np.linalg.norm(eigen_prediction - target, axis=-1)
                ade_eigen = np.nanmean(error_eigen)
                fde_eigen = np.nanmean(error_eigen[:, -1, :])
                print(f"Average Distance Error (ADE)_eigen: {ade_eigen:.4f}")
                print(f"Final Distance Error (FDE)_eigen: {fde_eigen:.4f}")


                error_gp = np.linalg.norm(gp_prediction - target, axis=-1)
                ade_gp = np.nanmean(error_gp)
                fde_gp = np.nanmean(error_gp[:, -1, :])
                print(f"Average Distance Error (ADE)_gp: {ade_gp:.4f}")
                print(f"Final Distance Error (FDE)_gp: {fde_gp:.4f}")

                error_linear = np.linalg.norm(linear_prediction - target, axis=-1)
                ade_linear = np.nanmean(error_linear)
                fde_linear = np.nanmean(error_linear[:, -1, :])
                print(f"Average Distance Error (ADE)_linear: {ade_linear:.4f}")
                print(f"Final Distance Error (FDE)_linear: {fde_linear:.4f}")

                error_tmp = np.linalg.norm(trajectron_prediction - target, axis=-1)
                ade_tmp = np.nanmean(error_tmp)
                fde_tmp = np.nanmean(error_tmp[:, -1, :])
                print(f"Average Distance Error (ADE)_traj: {ade_tmp:.4f}")
                print(f"Final Distance Error (FDE)_traj: {fde_tmp:.4f}")

                errors = np.linalg.norm(prediction - target, axis=-1)
                ade = np.nanmean(errors)
                fde = np.nanmean(errors[:, -1, :])
                print(f"Average Distance Error (ADE): {ade:.4f}")
                print(f"Final Distance Error (FDE): {fde:.4f}")

                np.save(filepath_to_save_predictions, prediction)      # shape = (# effective time steps, prediction length, # pedestrians, 2)
                np.save(filepath_to_save_targets, target)
    return


if __name__ == '__main__':
    main()