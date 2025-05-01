import os
import re
import numpy as np
from models.linear_predictor import LinearPredictor


def mathismathing(prediction_model,prediction_len = 12,history_len = 8,pattern = r'biwi_eth.npy$',directory='./lobby2/biwi_eth',namer='_linear_predictions.npy'):
    for type in [ 'test']:
            dirpath = os.path.join(directory, type)
            for file in os.listdir(dirpath):
                if re.match(pattern, file):
                    filepath_to_load = os.path.join(dirpath, file)
                    name, _ = os.path.splitext(file)
                    filepath_to_save_predictions = os.path.join(dirpath, name + namer)
                    data = np.load(filepath_to_load)
                    time_steps, n_pedestrians, _ = data.shape

                    predictions = []
                    T, num_agents, _ = data.shape
                    for agent_id in range(num_agents):
                        agent_xy = data[:, agent_id, :]
                        valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]  # NaN 제거
                        agent_xy=agent_xy.reshape(agent_xy.shape[0],1,agent_xy.shape[1])
               
                        all_fde=[]
                        #if valid_idx[0]+8 <idx_begin:
                        #    continue
                        #else:
                        if len(valid_idx) < 20 or not np.all(np.diff(valid_idx) == 1):
                            for start_idx in range(len(agent_xy)):
                                gt_future=np.array([[np.nan for k in range(2)] for i in range(prediction_len)])
                                gt_future=gt_future.reshape(12,1,2)
                                all_fde.append(gt_future)
                        else:
                            for start_idx in range(valid_idx[0]+history_len-1):
                                gt_future=np.array([[np.nan for k in range(2)] for i in range(prediction_len)])
                                gt_future=gt_future.reshape(12,1,2)
                                all_fde.append(gt_future)
                            #basically make the system such that it outputs enough for it to have a history len and a prediction length
                            for start_idx in range(len(valid_idx) -history_len):
                                history_idx = valid_idx[start_idx:start_idx + history_len]  
                                future_idx = valid_idx[start_idx+history_len:start_idx+history_len+prediction_len]  
                                output_t =prediction_model({pedestrian:agent_xy[history_idx,pedestrian] for pedestrian in range(1)})
                                if agent_id==18 and "zara02" in directory:
                                    print({pedestrian:agent_xy[history_idx,pedestrian] for pedestrian in range(1)})
                                output_t=np.array(list(output_t.values())).reshape(12,1,2)
                                all_fde.append(output_t)
                            for start_idx in range(valid_idx[-1],len(agent_xy)):
                                gt_future=np.array([[np.nan for k in range(2)] for i in range(prediction_len)])
                                gt_future=gt_future.reshape(12,1,2)
                                all_fde.append(gt_future)
                        if len(predictions):
                            all_fde=np.reshape(all_fde,(len(agent_xy),prediction_len,1,2))
                            predictions= np.concatenate([predictions,all_fde],axis=2)
                        else:
                            predictions=np.reshape(all_fde,(len(agent_xy),prediction_len,1,2))
                    print(filepath_to_save_predictions)
                    print(predictions.shape)
                    if "zara02" in directory:
                        arr3d = predictions[112] 
                        for j in range(arr3d.shape[1]):
                            block = arr3d[:, j, :]
                            if not np.isnan(block).any():
                                print(f"Index {j} (clean):\n{block}\n")
                    if "zara02" in directory:
                        arr3d = predictions[113] 
                        for j in range(arr3d.shape[1]):
                            block = arr3d[:, j, :]
                            if not np.isnan(block).any():
                                print(f"Index {j} (clean):\n{block}\n")
                    if "zara02" in directory:
                        arr3d = predictions[114] 
                        for j in range(arr3d.shape[1]):
                            block = arr3d[:, j, :]
                            if not np.isnan(block).any():
                                print(f"Index {j} (clean):\n{block}\n")
                    np.save(filepath_to_save_predictions,  predictions)     # shape = (# effective time steps, prediction length, # pedestrians, 2)
    return
