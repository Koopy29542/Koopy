import os
import re
import numpy as np
from ..models.linear_predictor import LinearPredictor

def main():
    prediction_len = 12
    history_len = 8
    pattern = r'^\d+\.npy$'
    
    # 모델
    prediction_model = LinearPredictor(
        prediction_len=prediction_len,
        history_len=history_len,
        smoothing_factor=0.7,
        dt=0.1
    )
    
    test_dir = './lobby3/test'
    
    for file in os.listdir(test_dir):
        if re.match(pattern, file):
            filepath_to_load = os.path.join(test_dir, file)
            name, _ = os.path.splitext(file)
            
            filepath_to_save_predictions = os.path.join(test_dir, name + '_linear_predictions.npy')
            filepath_to_save_targets = os.path.join(test_dir, name + '_targets.npy')
            
            data = np.load(filepath_to_load)  # (T, num_agents, 2)
            T, num_agents, _ = data.shape
            
            # 에이전트별 결과를 쌓을 리스트
            all_agents_predictions = []
            all_agents_targets = []
            
            for agent_id in range(num_agents):
                agent_xy = data[:, agent_id, :]             # (T, 2)
                valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]

                # testtraj2와 동일: (T, 12, 2)를 담을 리스트(길이 T)
                # 마지막에 np.array로 변환 후 (T,12,2)가 됨
                # 그리고 expand_dims(axis=2) -> (T,12,1,2)
                all_preds = []
                all_tgts  = []

                # 1) 관측 길이 부족 or 연속된 구간이 아님 -> T 전부 NaN
                if len(valid_idx) < 20 or not np.all(np.diff(valid_idx) == 1):
                    for _ in range(T):
                        nan_future = np.full((prediction_len, 2), np.nan)
                        all_preds.append(nan_future)
                        all_tgts.append(nan_future)
                
                else:
                    # (A) 앞부분: testtraj2와 동일
                    #  -> range(valid_idx[0] + history_len - 1) 개수만큼 NaN
                    for _ in range(valid_idx[0] + history_len - 1):
                        nan_future = np.full((prediction_len, 2), np.nan)
                        all_preds.append(nan_future)
                        all_tgts.append(nan_future)
                    
                    # (B) 메인 슬라이딩 구간
                    #     testtraj2: range(len(valid_idx) - history_len)
                    for start_idx in range(len(valid_idx) - history_len):
                        # history 구간
                        history_idx = valid_idx[start_idx : start_idx + history_len]
                        # future 구간
                        future_idx = valid_idx[start_idx + history_len : 
                                               start_idx + history_len + prediction_len]
                        
                        history = agent_xy[history_idx]  # shape (8,2)
                        
                        # 실제 future (부족하면 NaN 패딩)
                        if len(future_idx) < prediction_len:
                            # 남은 future가 예측길이보다 짧으면 NaN으로 채움
                            actual_part = agent_xy[future_idx]  # (미만프레임,2)
                            pad_size = prediction_len - actual_part.shape[0]
                            gt_future = np.concatenate([
                                actual_part,
                                np.full((pad_size, 2), np.nan)
                            ], axis=0)
                        else:
                            gt_future = agent_xy[future_idx]
                        
                        # 모델 예측
                        pred_dict = prediction_model({agent_id: history})  # {agent_id: (12,2)}
                        pred_future = pred_dict[agent_id][:prediction_len]
                        
                        # 혹시 예측 길이가 다를 경우 체크
                        if pred_future.shape[0] != prediction_len:
                            print(f"[Warning] Agent {agent_id} 예측 길이 불일치: {pred_future.shape[0]}개")
                            # 그냥 NaN 패딩 처리 (또는 continue)
                            pred_future = np.full((prediction_len, 2), np.nan)
                        
                        # testtraj2와 동일하게 list에 append
                        all_preds.append(pred_future)
                        all_tgts.append(gt_future)
                    
                    # (C) 뒷부분:
                    #     range(valid_idx[-1], T) 구간은 전부 NaN
                    for _ in range(valid_idx[-1], T):
                        nan_future = np.full((prediction_len, 2), np.nan)
                        all_preds.append(nan_future)
                        all_tgts.append(nan_future)
                
                # (T,) 길이의 list -> (T,12,2) np.array
                all_preds_np = np.array(all_preds)   # (T,12,2)
                all_tgts_np  = np.array(all_tgts)    # (T,12,2)

                # (T,12,2) -> (T,12,1,2)
                all_preds_np = np.expand_dims(all_preds_np, axis=2)
                all_tgts_np  = np.expand_dims(all_tgts_np, axis=2)
                
                all_agents_predictions.append(all_preds_np)
                all_agents_targets.append(all_tgts_np)
            
            # 에이전트 차원(axis=2)으로 concatenate -> (T,12,num_agents,2)
            predictions = np.concatenate(all_agents_predictions, axis=2)
            targets = np.concatenate(all_agents_targets, axis=2)
            
            # ADE/FDE 계산
            errors = np.linalg.norm(predictions - targets, axis=-1)  # (T,12,num_agents)
            ade = np.nanmean(errors)
            fde = np.nanmean(errors[:, -1, :])
            
            print("File:", file)
            print("✅ ADE: {:.4f}, FDE: {:.4f}".format(ade, fde))
            print("Predictions shape:", predictions.shape,
                  "Targets shape:", targets.shape)
            
            # 저장
            np.save(filepath_to_save_predictions, predictions)  # (T,12,num_agents,2)
            np.save(filepath_to_save_targets, targets)
            print("Saved predictions to:", filepath_to_save_predictions)
            print("Saved targets to:", filepath_to_save_targets)
            print("====================================\n")

if __name__ == '__main__':
    main()
