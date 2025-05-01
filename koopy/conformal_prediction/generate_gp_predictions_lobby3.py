import os
import re
import numpy as np
from models.gp_predictor import GaussianProcessPredictor


def main():
    prediction_len = 12
    history_len = 8
    
    pattern = r'^\d+\.npy$'

    # model
    prediction_model = GaussianProcessPredictor(prediction_len=prediction_len, history_len= 8, dt = 0.1)
    # load dataset

    # build a calibration dataset
    idx_begin, idx_end = 40, 160
    

    for type in ['test']:
        dirpath = os.path.join('./lobby3', type)
        for file in os.listdir(dirpath):
            if re.match(pattern, file):
                #filepath_to_load = os.path.join(dirpath, file)
                name, _ = os.path.splitext(file)
                filepath_to_save_predictions = os.path.join(dirpath, name + '_gp_predictions.npy')
                filepath_to_save_targets = os.path.join(dirpath, name + '_targets.npy')
               # data = np.load(filepath_to_load)
               # time_steps, n_pedestrians, _ = data.shape

                target,prediction=prediction_model.testtraj2('lobby3/test')           
                errors = np.linalg.norm(prediction - target, axis=-1)  # (T, prediction_len, # agents)
                ade = np.nanmean(errors)  # 전체 프레임, 에이전트에 대해 평균 오차 계산
                fde = np.nanmean(errors[:, -1, :])  # 마지막 예측 프레임에 대한 오차 평균
                print(f"✅ Average Distance Error (ADE): {ade:.4f}")  # 수정: ADE 출력
                print(f"✅ Final Distance Error (FDE): {fde:.4f}")      # 수정: FDE 출력

                print(prediction.shape)
                np.save(filepath_to_save_predictions, prediction)      # shape = (# effective time steps, prediction length, # pedestrians, 2)
                np.save(filepath_to_save_targets, target)
    return


if __name__ == '__main__':
    main()