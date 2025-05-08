import os
import re
import math
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt

###########################################
# 1. 기본 장애물 정보 (lobby/train)
###########################################
default_obstacles_meter = {
    'pillar_left': [(0.12, -0.62), (-0.16, -0.62), (-0.38, -0.88), (-0.3, -1.38), (0.16, -1.44), (0.42, -1.16), (0.36, -0.76)],
    'pillar_right': [(5.62, -0.96), (5.48, -1.36), (5.62, -1.64), (6.12, -1.80), (6.36, -1.42), (6.26, -1.04)],
    'wall_north': [(0.68, 3.98), (0.08, 1.34), (11.80, 0.58), (11.76, 2.52)],
    'wall_east': [(8.24, -1.24), (7.66, -10.02), (11.58, -10.30), (11.70, -1.54)],
    'wall_south': [(-2.70, -8.38), (-2.78, -9.44), (7.60, -9.92), (7.72, -8.86)],
    'wall_west': [(-4.12, 0.02), (-4.42, -8.64), (-2.66, -8.80), (-2.24, -0.22)],
    'wall_northwest': [(-3.34, 1.46), (-1.88, 1.30), (-1.62, 4.48), (-3.08, 4.38)],
    'fanuc': [(2.44, -1.86), (2.26, -4.38), (3.20, -4.44), (3.42, -1.96)],
    'entrance_wall_left': [(0.38, -6.36), (0.26, -8.82), (1.68, -8.92), (1.84, -6.48)],
    'entrance_wall_right': [(3.66, -6.44), (3.42, -8.82), (4.86, -9.00), (5.04, -6.54)]
}
default_obstacle_polygons = [Polygon(coords) for coords in default_obstacles_meter.values()]

###########################################
# 2. lobby2 파일별 장애물 정보
###########################################

obstacles_meter_eth = {
    'obstacle1': [(14.41, -6.17), (14.41, -1.6), (-0.5, -1.0), (-0.7, -6.17), (14.41, -6.17)],
    'obstacle2': [(-1.6, 12.8), (-1.8, 16.4), (14.41, 16.4), (14.41, 15.8), (-1.6, 12.8)]
}

obstacles_meter_hotel = {
    'obstacle1': [(5.4, -10.31), (4.4, -10.31), (4.1, 4.31), (5.4, 4.31), (5.4, -10.31)],
    'obstacle2': [(-2.5, -10.31), (-0.8, -10.31), (-0.8, 4.31), (-2.5, 4.31), (-2.5, -10.31)]
}
obstacles_meter_zara01 = {
    'obstacle1': [(-0.02104651, 8.0), (-0.02104651, 12.3864436), (9.5, 12.3864436), (9.5, 8.0), (-0.02104651, 8.0)],
    'obstacle2': [(10.0, 11.2), (10.0, 13.3864436), (12.0, 13.3864436), (12.0, 11.2), (10.0, 11.2)],
    'obstacle3': [(10.0, 6.5), (11.9, 6.5), (11.9, 11.3), (10.0, 11.3), (10.0, 6.5)],
    'obstacle4': [(-0.02104651, 0.76134018), (15.13244069, 0.76134018), (15.13244069, 2.9), (-0.02104651, 2.9), (-0.02104651, 0.76134018)]
}
obstacles_meter_zara02 = {
    'obstacle1': [(-0.35779069, 9.0), (-0.35779069, 14.94274416), (9.5, 14.94274416), (9.5, 9.0), (-0.35779069, 9.0)],
    'obstacle2': [(10.0, 11.2), (10.0, 14.94274416), (12.0, 14.94274416), (12.0, 11.2), (10.0, 11.2)],
    'obstacle3': [(10.0, 6.5), (11.9, 6.5), (11.9, 11.3), (10.0, 11.3), (10.0, 6.5)],
    'obstacle4': [(-0.35779069, 0.72625721), (15.55842276, 0.72625721), (15.55842276, 2.9), (-0.35779069, 2.9), (-0.35779069, 0.72625721)]
}
obstacles_meter_uni = {
    'obstacle1': [(3.8, 13.85420137), (6.0, 11.5), (10.0, 13.85420137), (3.8, 13.85420137)],
    'obstacle2': [(-0.17468604, 9.8), (-0.17468604, 11.5), (4.0, 11.5), (3.5, 9.8), (-0.17468604, 9.8)]
}

###########################################
# 2. lobby3 장애물 정보
###########################################
obstacles_meter_lobby3 = {
    'left_top': [(-2.28, 1.19), (-10, 1.19), (-10, 5), (-2.28, 5)],
    'left_bottom': [(-2.59, -8.95), (-10, -8.95), (-10, -15), (-2.59, -15)],
    'right_top': [(-0.23, 1.17), (10, 1.17), (10, 5), (-0.23, 5)],
    'right_bottom': [(8.13, -8.95), (10, -8.95), (10, -15), (8.13, -15)],
    'bottom': [(-2.59, -8.95), (-2.59, -15), (8.13, -15), (8.13, -8.95)],  # 연결된 하단 벽
    'left': [(-10, -0.6), (-2.59, -0.6), (-2.59, -8.95), (-10, -8.95)],  # left_bottom과 연결
    'right': [(8.13, -8.95), (8.13, -0.6), (10, -0.6), (10, -8.95)],
    'elevator': [(-2.30, 3.92), (-0.26, 3.92), (-0.26, 5), (-2.30, 5)],
    'entrance': [(0.63, -8.86), (0.63, -6.54), (5.03, -6.54), (5.03, -8.86)],
    'middle_obstacle': [(2.27, -2.00), (3.26, -2.00), (3.29, -4.29), (2.21, -4.34)],
    'pillar_left': [(-0.26, -0.55), (-0.54, -0.73), (-0.67, -1.11), (-0.52, -1.44), (-0.23, -1.59),
                    (0.06, -1.37), (0.20, -1.07), (0.03, -0.75)],
    'pillar_right': [(5.80, -0.68), (5.56, -0.77), (5.38, -1.09), (5.48, -1.41),
                     (5.79, -1.55), (6.06, -1.39), (6.18, -1.16), (6.04, -0.86)]
}

###########################################
# 3. 12차원 binary 벡터 계산 함수 (polygons 인자 추가)
###########################################
def get_12dim_obstacle_vector(x0, y0, radius=3.0, polygons=None):
    """
    주어진 (x0, y0)에서 30도 간격(총 12방향) 선분(길이 1m)을 생성하고,
    각 선분이 장애물과 교차하면 1, 아니면 0을 반환하여 12차원 binary 벡터를 리턴합니다.
    polygons: 사용할 장애물 다각형 리스트. 없으면 default_obstacle_polygons 사용.
    """
    if polygons is None:
        polygons = default_obstacle_polygons
    angles = range(0, 360, 30)  # 0, 30, ..., 330 (총 12방향)
    vector = []
    for deg in angles:
        rad = math.radians(deg - 90)  # '위쪽'을 0도로 맞추기 위해
        x_end = x0 + radius * math.cos(rad)
        y_end = y0 + radius * math.sin(rad)
        line = LineString([(x0, y0), (x_end, y_end)])
        intersect = any(poly.intersects(line) for poly in polygons)
        vector.append(1 if intersect else 0)
    return vector

##########################################################
# KoopmanPredictor: 클러스터별로 Koopman matrix 학습
##########################################################
class KoopmanPredictor:
    def __init__(self,
                 prediction_len=12,    # 미래 예측 길이
                 data_dir='lobby2/biwi_eth',
                 min_samples=20,
                 dt=0.1,
                 pattern=r'^.*\d{2}\.npy$',
                 n_clusters=10):
        self._prediction_len = prediction_len
        self._min_samples = min_samples
        self._dt = dt
        self.pattern = pattern
        self.n_clusters = n_clusters
        
        # self.obstacle_polygons를 설정하지 않으면 기본(default)를 사용합니다.

        # 1. 학습 데이터 로드 (history 8프레임 → 16차원 state)
        all_past, all_future = self._load_training_data(data_dir)
        self.all_past,_ = self._load_training_data(data_dir)

        # 2. 각 training sample에 대해 **8프레임 중 마지막 프레임** (최신 프레임)의 (x,y) 좌표로
        #    12차원 binary obstacle 벡터를 계산합니다.
        samples = all_past.T  # shape: (N, 16) – 각 행이 하나의 sample
        features = []
        for s in samples:
            features.append(self._compute_binary_features(s))
        features = np.array(features)  # (N, 12)

        # 3. KMeans를 이용해 12차원 binary vector들로 클러스터링 수행
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(features)

        # 4. 각 클러스터별로 local Koopman operator 학습
        self.local_Ks = []  # 클러스터 label에 따른 Koopman matrix 저장
        for c in range(n_clusters):
            idx = np.where(cluster_labels == c)[0]
            if len(idx) < self._min_samples:
                K_local = np.eye(self._observable_dim())
            else:
                local_past = all_past[:, idx]     # (16, n_local)
                local_future = all_future[:, idx] # (16, n_local)
                psi_past_list = []
                psi_future_list = []
                n_local = local_past.shape[1]
                for k in range(n_local):
                    psi_past_list.append(self._compute_observables(local_past[:, k]))
                    psi_future_list.append(self._compute_observables(local_future[:, k]))
                psi_past = np.column_stack(psi_past_list)   # (d_obs, n_local)
                psi_future = np.column_stack(psi_future_list) # (d_obs, n_local)
                try:
                    pseudo_inv = np.linalg.pinv(psi_past)
                    K_local = psi_future @ pseudo_inv
                except np.linalg.LinAlgError:
                    K_local = np.eye(self._observable_dim())

            self.local_Ks.append(K_local)

    def _compute_binary_features(self, state):
        """
        state: 16차원 vector (8프레임의 x,y; 최신 프레임이 state[0:2])
        → **최신 프레임의 (x,y) 좌표**에 대해 12차원 binary obstacle 벡터를 계산하여 리턴합니다.
        """
        x = state[0]
        y = state[1]
        # 만약 self.obstacle_polygons가 설정되어 있으면 사용하고, 아니면 기본(default)를 사용합니다.
        polygons = self.obstacle_polygons if hasattr(self, "obstacle_polygons") else default_obstacle_polygons
        return np.array(get_12dim_obstacle_vector(x, y, radius=3.0, polygons=polygons))  # (12,)

    def __call__(self, tracking_result):
        prediction_result = {}
        required_history = 8  # history 길이 8 사용
        for object_id, history in tracking_result.items():
            if not isinstance(history, np.ndarray):
                history = np.array(history)
            if len(history) < required_history:
                xy0 = history[-1] if len(history) > 0 else np.array([0., 0.])
                p = np.tile(xy0[None, ...], (self._prediction_len, 1))
                prediction_result[object_id] = p
                continue
            past_frames = history[-required_history:]
            state = past_frames[::-1].flatten()  # 최신 프레임이 state[0:2]에 위치
            ps = []
            for _ in range(self._prediction_len):
                state = self._forward(state)  # 다음 state 예측
                ps.append(state[:2])          # 예측된 위치 (x, y)
            prediction_result[object_id] = np.array(ps)
        return prediction_result

    def _load_training_data(self, data_dir):
        """
        data_dir/train 내의 각 npy 파일을 불러와서,
        각 agent의 연속 8 프레임을 past, 다음 8 프레임을 future로 사용합니다.
        - lobby2인 경우, 파일명을 기준으로 다른 obstacles_meter를 사용합니다.
        - lobby3인 경우, 별도의 obstacle_polygons (obstacles_meter_lobby3)를 사용합니다.
        """
        train_dir = os.path.join(data_dir, 'train')
        pattern = re.compile(r'^.*\.npy$')
        all_past = []
        all_future = []
        required_history = 8  # history 길이

        # lobby2일 경우 valid_files 설정
        valid_files = None
        if "lobby2" in data_dir:
            valid_files = {"crowds_eth_train.npy","crowds_hotel_train.npy","crowds_zara01_train.npy", "crowds_zara02_train.npy", "uni_examples_train.npy"}
        elif "lobby3" in data_dir:
            self.obstacle_polygons = [Polygon(coords) for coords in obstacles_meter_lobby3.values()]
        for fname in os.listdir(train_dir):
            if pattern.match(fname):
                if valid_files is not None and fname not in valid_files:
                    continue
                filepath = os.path.join(train_dir, fname)
                if valid_files is not None:

                    if fname == "crowds_eth_train.npy":
                        obstacles_meter = obstacles_meter_eth
                    elif fname == "crowds_hotel_train.npy":
                        obstacles_meter = obstacles_meter_hotel                    
                    elif fname == "crowds_zara01_train.npy":
                        obstacles_meter = obstacles_meter_zara01
                    elif fname == "crowds_zara02_train.npy":
                        obstacles_meter = obstacles_meter_zara02
                    elif fname == "uni_examples_train.npy":
                        obstacles_meter = obstacles_meter_uni
                    else:
                        continue
                    # 해당 파일의 장애물 다각형 목록을 self.obstacle_polygons로 저장 (추후 사용)
                    self.obstacle_polygons = [Polygon(coords) for coords in obstacles_meter.values()]
                data = np.load(filepath)  # (T, num_agents, 2)
                T, num_agents, _ = data.shape
                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                    if len(valid_idx) < (required_history + 1):
                        continue
                    for i in range(len(valid_idx) - required_history):
                        past_indices = valid_idx[i:i+required_history]
                        future_indices = valid_idx[i+1:i+required_history+1]
                        past_window = agent_xy[past_indices]    # (8,2)
                        future_window = agent_xy[future_indices]  # (8,2)
                        # 최신 프레임이 맨 앞에 오도록 뒤집음 (state[0:2]가 최신 위치)
                        past_state = past_window[::-1].flatten()
                        future_state = future_window[::-1].flatten()
                        all_past.append(past_state)
                        all_future.append(future_state)
        if len(all_past) == 0:
            raise ValueError(f"No valid training data found in {train_dir}.")
        all_past = np.array(all_past).T  # (16, N)
        all_future = np.array(all_future).T  # (16, N)
        return all_past, all_future

    # observable 차원 (여기서는 16 사용)
    def _observable_dim(self):
        # state (16) + speed_mean (1) + omega_mean (1)
        return 16+16
    

    def _compute_observables(self, state):
        state = state.flatten()  # (16,)
        dt = self._dt  # Time step

        # 1차 미분 (속도)
        dx = (state[::2][1:] - state[::2][:-1]) / dt  # (7,)
        dy = (state[1::2][1:] - state[1::2][:-1]) / dt  # (7,)

        # 선속도 (magnitude of velocity)
        speed = np.sqrt(dx**2 + dy**2)  # (7,)
        speed_mean = np.mean(speed)

        # 각속도 (변화량)
        theta = np.arctan2(dy, dx)  # (7,)
        omega = (theta[1:] - theta[:-1]) / dt  # (6,)
        omega_mean = np.mean(omega)

        observables = np.concatenate([state, state**2])
        return observables


    def _get_K(self, state):
        """
        예측 시, 입력 state에 대해 **최신 프레임**의 (x,y)로 계산한 12차원 binary feature를 사용해
        클러스터 label을 얻고, 해당 클러스터의 Koopman matrix를 반환합니다.
        """
        binary_feat = self._compute_binary_features(state)
        feature = binary_feat.reshape(1, -1)
        cluster_label = self.kmeans.predict(feature)[0]
        K = self.local_Ks[cluster_label]
        if K.shape[0] != self._observable_dim():
            K = np.eye(self._observable_dim())
            self.local_Ks[cluster_label] = K
        return K

    def _forward(self, state):
        K = self._get_K(state)
        o = self._compute_observables(state)
        o_next = K @ o
        next_state = self._to_state(o_next)
        return next_state

    def _to_state(self, observables):
        # observables에서 원래 16차원 state 복원
        return observables[:16]

    # 이하의 testtraj, testtraj2, evaluate_test 함수들은 기존 코드를 그대로 유지합니다.
    def testtraj(self, test_dir):
        pattern = re.compile(self.pattern)
        goal = []
        future = []
        prediction_len = 12
        history_len = 8
        diff = 8 - history_len
        for fname in os.listdir(test_dir):
            goal = []
            future = []
            if pattern.match(fname):
                filepath = os.path.join(test_dir, fname)
                data = np.load(filepath)
                name, _ = os.path.splitext(filepath)
                T, num_agents, _ = data.shape
                filepath_to_save_predictions =  name + '_koopman_8_geo_predictions.npy'
                for agent_id in range(num_agents):
                    all_ade = []
                    all_fde = []
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                    if len(valid_idx) < 20:  # 최소 예측 가능 길이 확인 (2개 관측 + 12개 예측)
                        for start_idx in range(len(agent_xy)):
                            gt_future=[[np.nan,np.nan]for i in range(prediction_len)]
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                    else:
                        for start_idx in range(valid_idx[0]+history_len-1):
                            gt_future=[[np.nan,np.nan] for i in range(prediction_len)]
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                        # (t, t+1)로 예측하고 t+2 ~ t+13과 비교
                        for start_idx in range(len(valid_idx)-history_len):
                            history_idx = valid_idx[start_idx:start_idx + history_len]   # 관측 프레임
                            future_idx = valid_idx[start_idx + history_len:start_idx + history_len+prediction_len]  # 예측 대상 프레임

                            if len(future_idx) < prediction_len:
                                history = agent_xy[history_idx]
                                pred_dict = self({agent_id: history})  # Koopman 예측
                                pred_future = pred_dict[agent_id][:prediction_len]  # 처음 12 프레임 사용
                                all_fde.append(pred_future)
                                pad_size = prediction_len - len(future_idx)
                                pad=[[np.nan,np.nan]for i in range(pad_size)]
                                gt_future = np.concatenate([agent_xy[future_idx],pad])
                                all_ade.append(gt_future)
                            else:
                                history = agent_xy[history_idx]  # 초기 2개 프레임
                                gt_future = agent_xy[future_idx]  # 실제 미래 궤적
                                pred_dict = self({agent_id: history})  # Koopman 예측
                                pred_future = pred_dict[agent_id][:prediction_len]  # 처음 12 프레임 사용
                                all_ade.append(gt_future)
                                all_fde.append(pred_future)
                                # 예측 길이 검증
                                if pred_future.shape[0] != prediction_len:
                                    print(f"[Warning] Agent {agent_id} 예측 길이 불일치: {pred_future.shape[0]}개")
                                    continue
                        for start_idx in range(valid_idx[-1],len(agent_xy)):
                            gt_future=[[np.nan,np.nan]for i in range(prediction_len)]
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                    if len(goal):
                        all_fde=np.reshape(all_fde,(len(agent_xy),prediction_len,1,2))
                        all_ade=np.reshape(all_ade,(len(agent_xy),prediction_len,1,2))
                        future= np.concatenate([future,all_fde],axis=2)
                        goal= np.concatenate([goal,all_ade],axis=2)
                    else:
                        future= np.reshape(all_fde,(len(agent_xy),prediction_len,1,2))
                        goal=np.reshape(all_ade,(len(agent_xy),prediction_len,1,2))
                print(len(agent_xy))
                print(goal.shape)
                print(future.shape)
                np.save(filepath_to_save_predictions, future)   
        return goal, future

    def testtraj2(self, test_dir):
        pattern = re.compile(self.pattern)
        goal = None
        future = None
        prediction_len = 12
        history_len = 8
        diff = 8 - history_len
        
        for fname in os.listdir(test_dir):
            if pattern.match(fname):
                filepath = os.path.join(test_dir, fname)
                data = np.load(filepath)
                name, _ = os.path.splitext(filepath)
                T, num_agents, _ = data.shape
                
                all_ade_list = []
                all_fde_list = []

                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]

                    all_ade = []
                    all_fde = []

                    if len(valid_idx) < 20 or not np.all(np.diff(valid_idx) == 1):
                        # 관측 길이가 부족한 경우 NaN 패딩
                        for start_idx in range(len(agent_xy)):
                            gt_future = np.full((prediction_len, 2), np.nan)
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)
                    else:
                        for start_idx in range(valid_idx[0] + history_len - 1):
                            gt_future = np.full((prediction_len, 2), np.nan)
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)

                        for start_idx in range(len(valid_idx) - history_len):
                            history_idx = valid_idx[start_idx : start_idx + history_len]
                            future_idx = valid_idx[start_idx + history_len : start_idx + history_len + prediction_len]

                            history = agent_xy[history_idx]

                            if len(future_idx) < prediction_len:
                                pred_dict = self({agent_id: history})
                                pred_future = pred_dict[agent_id][:prediction_len]
                                pad_size = prediction_len - len(future_idx)
                                pad = np.full((pad_size, 2), np.nan)
                                gt_future = np.vstack([agent_xy[future_idx], pad])
                            else:
                                gt_future = agent_xy[future_idx]
                                pred_dict = self({agent_id: history})
                                pred_future = pred_dict[agent_id][:prediction_len]

                                if pred_future.shape[0] != prediction_len:
                                    print(f"[Warning] Agent {agent_id} 예측 길이 불일치: {pred_future.shape[0]}개")
                                    continue

                            all_ade.append(gt_future)
                            all_fde.append(pred_future)

                        for start_idx in range(valid_idx[-1], len(agent_xy)):
                            gt_future = np.full((prediction_len, 2), np.nan)
                            all_ade.append(gt_future)
                            all_fde.append(gt_future)

                    all_ade_list.append(np.array(all_ade))  
                    all_fde_list.append(np.array(all_fde))  

                all_ade_np = np.stack(all_ade_list, axis=2)
                all_fde_np = np.stack(all_fde_list, axis=2)

                if goal is None:
                    goal = all_ade_np
                    future = all_fde_np
                else:
                    goal = np.concatenate([goal, all_ade_np], axis=2)
                    future = np.concatenate([future, all_fde_np], axis=2)

                print(len(agent_xy))
                print(goal.shape)
                print(future.shape)

        return goal, future

    @staticmethod
    def pad_to_length(arr, target_length):
        """arr의 axis=0 길이를 target_length로 padding (부족하면 np.nan으로 채움)"""
        current_length = arr.shape[0]
        if current_length < target_length:
            pad_width = ((0, target_length - current_length), (0, 0), (0, 0), (0, 0)) # frame_id에 대해서만 패딩추가
            return np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
        return arr

    def evaluate_test(self, test_dir):
        pattern = re.compile(r'^.*\.npy$')

        all_ade = []
        all_fde = []

        history_len = 8  # Modified: history 길이 8
        prediction_len = 12
        required_length = history_len + prediction_len  # 20 프레임 필요
        for fname in os.listdir(test_dir):
            if pattern.match(fname):
                filepath = os.path.join(test_dir, fname)
                data = np.load(filepath)  # shape (T, num_agents, 2)
                T, num_agents, _ = data.shape

                for agent_id in range(num_agents):
                    agent_xy = data[:, agent_id, :]
                    valid_idx = np.where(~np.isnan(agent_xy).any(axis=1))[0]
                    if len(valid_idx) < required_length:
                        continue

                    # (t ~ t+7)로 예측하고 t+8 ~ t+19과 비교
                    for start_idx in range(len(valid_idx) - required_length + 1):
                        history_idx = valid_idx[start_idx:start_idx + history_len]
                        future_idx = valid_idx[start_idx + history_len:start_idx + required_length]
                        if len(history_idx) < history_len or len(future_idx) < prediction_len:
                            continue

                        history = agent_xy[history_idx]
                        gt_future = agent_xy[future_idx]
                        pred_dict = self({agent_id: history})
                        pred_future = pred_dict[agent_id][:prediction_len]
                        if pred_future.shape[0] != prediction_len:
                            print(f"[Warning] Agent {agent_id} 예측 길이 불일치: {pred_future.shape[0]}개")
                            continue
                        errors = np.linalg.norm(pred_future - gt_future, axis=1)
                        all_ade.append(errors.mean())
                        all_fde.append(errors[-1])
        if len(all_ade) > 0:
            ade = np.mean(all_ade)
            fde = np.mean(all_fde)
            print(f"✅ Average Distance Error (ADE): {ade:.4f}")
            print(f"✅ Final Distance Error (FDE): {fde:.4f}")
        else:
            print("❌ No valid test data found.")

