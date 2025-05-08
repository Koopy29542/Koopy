import os
import re
import numpy as np
from models.koopman_clu_geo import KoopmanPredictor


def main():
    prediction_len = 12
    history_len = 8


    pattern = [r'biwi_eth.npy$',r'students001.npy$',r'students003.npy$',r'biwi_hotel.npy$',r'^.*\d{2}\.npy$',r'^.*\d{2}\.npy$']
    directories=  ['../lobby2/biwi_eth','../lobby2/univ','../lobby2/univ','../lobby2/biwi_hotel','../lobby2/crowds_zara01','../lobby2/crowds_zara02']
    test_dirpaths = ['../lobby2/biwi_eth/test','../lobby2/univ/test','../lobby2/univ/test','../lobby2/biwi_hotel/test','../lobby2/crowds_zara01/test','../lobby2/crowds_zara02/test']
    for i,k,j in zip(pattern,directories,test_dirpaths):
    # model
        for type in ['test']:
            dirpath = os.path.join(k, type)
            for file in os.listdir(dirpath):
                if re.match(i, file):
                    #filepath_to_load = os.path.join(dirpath, file)
                    name, _ = os.path.splitext(file)
                    filepath_to_save_targets = os.path.join(dirpath, name + '_targets.npy')
                    #target,prediction=prediction_model.testtraj(j)
                    # idx, prediction_len, number of agents, 2
                    test_dir=j
                    goal = []
                    prediction_len = 12
                    history_len = 8
                    diff = 8 - history_len
                    for fname in os.listdir(test_dir):
                        if re.match(i,fname):
                            filepath = os.path.join(test_dir, fname)
                            data = np.load(filepath)
                            T, num_agents, _ = data.shape
                            for agent_id in range(num_agents):
                                all_ade = []
                                agent_xy = data[:, agent_id, :]
                                valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                                if len(valid_idx) < 20:
                                    # 관측 길이가 부족한 경우
                                    for _ in range(len(agent_xy)):
                                        gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                                        all_ade.append(gt_future)
                                else:
                                    # enough frames
                                    for start_idx in range(valid_idx[0] + history_len - 1):
                                        gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                                        all_ade.append(gt_future)
                                    for start_idx in range(len(valid_idx) - history_len):
                                        history_idx = valid_idx[start_idx : start_idx + history_len]
                                        future_idx = valid_idx[start_idx + history_len : start_idx + history_len + prediction_len]

                                        if len(future_idx) < prediction_len:
                                            pad_size = prediction_len - len(future_idx)
                                            pad = [[np.nan, np.nan] for _ in range(pad_size)]
                                            gt_future = np.concatenate([agent_xy[future_idx], pad])
                                            all_ade.append(gt_future)
                                        else:
                                            gt_future = agent_xy[future_idx]
                                            all_ade.append(gt_future)

                                    # 나머지 구간 NaN 패딩
                                    for _ in range(valid_idx[-1], len(agent_xy)):
                                        gt_future = [[np.nan, np.nan] for _ in range(prediction_len)]
                                        all_ade.append(gt_future)

                                if len(goal):
                                    all_ade = np.reshape(all_ade, (len(agent_xy), prediction_len, 1, 2))
                                    goal = np.concatenate([goal, all_ade], axis=2)
                                else:
                                    goal = np.reshape(all_ade, (len(agent_xy), prediction_len, 1, 2))
                    print(filepath_to_save_targets)
                    print(goal.shape)
                    target= goal  # shape = (# effective time steps, prediction length, # pedestrians, 2)
                    np.save(filepath_to_save_targets, target)
    return


if __name__ == '__main__':
    main()