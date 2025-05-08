import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import math
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt

def get_12dim_obstacle_vector(x0, y0, radius=3.0, polygons=None):
    if polygons is None:
        raise ValueError("polygons must be provided")
    angles = range(0, 360, 30)
    vector = []
    for deg in angles:
        rad = math.radians(deg - 90)
        x_end = x0 + radius * math.cos(rad)
        y_end = y0 + radius * math.sin(rad)
        line = LineString([(x0, y0), (x_end, y_end)])
        vector.append(1 if any(poly.intersects(line) for poly in polygons) else 0)
    return vector

class GeoEncoder(nn.Module):
    def __init__(self, input_dim=12, encoded_dim=4):
        super(GeoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, encoded_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class KoopmanPredictor:
    def __init__(self, prediction_len=12, model_file='koopy_3.pkl', eig_top_k=10):
        self._prediction_len = prediction_len
        self.encoder = GeoEncoder()
        self.K = None
        self.K_trunc = None
        self.obstacle_polygons = None

        if os.path.exists(model_file):
            print(f"[INFO] Loading Koopman model from {model_file}")
            self._load_model(model_file)
            self.K_trunc = self._truncate_K(self.K, eig_top_k)
            print(f"[INFO] Truncated K using top {eig_top_k} eigenvalues.")
        else:
            raise FileNotFoundError(f"Model file {model_file} not found.")

    def _load_model(self, filename):
        with open(filename, 'rb') as f:
            model_dict = pickle.load(f)
        self.K = model_dict['K']
        self.encoder.load_state_dict(model_dict['encoder_state'])
        self.obstacle_polygons = model_dict['obstacle_polygons']
        print(f"[INFO] Model loaded from {filename}.")

    # Truncate Koopman matrix using top-k eigenvalues
    def _truncate_K(self, K, top_k):
        eigenvals, V_full = np.linalg.eig(K)
        V_full_inv = np.linalg.inv(V_full)
        
        idx = np.argsort(np.abs(eigenvals))[::-1][:top_k]
        Λ = np.diag(eigenvals[idx])
        V = V_full[:, idx]
        W = V_full_inv[idx, :] 

        return V @ Λ @ W


    # Predict future trajectory
    def __call__(self, tracking_result):
        """
        K^s 를 한 번씩 곱해 위치만 뽑아내는 단순 rollout 방식
        """
        prediction_result = {}
        required_history = 8
        dim = self.K.shape[0]  # 20

        for object_id, history in tracking_result.items():
            hist = np.asarray(history)
            # 히스토리 부족 시 마지막 위치 복사
            if hist.shape[0] < required_history:
                xy0 = hist[-1] if hist.size else np.array([0., 0.])
                prediction_result[object_id] = np.tile(xy0, (self._prediction_len, 1))
                continue

            #(16D)
            past = hist[-required_history:][::-1].flatten()
            #geo 4D
            last_xy = hist[-1]
            geo_vec = get_12dim_obstacle_vector(
                last_xy[0], last_xy[1],
                radius=3.0,
                polygons=self.obstacle_polygons
            )
            geo_feat = self.encoder(
                torch.tensor(geo_vec, dtype=torch.float32).unsqueeze(0)
            ).detach().cpu().numpy().squeeze()

            # 3) 초기 상태 벡터 (20D)
            z0 = np.concatenate([past, geo_feat])  # shape (20,)

            # 4) K^s rollout
            preds = []
            K_power = np.eye(dim)
            for s in range(1, self._prediction_len + 1):
                K_power = K_power @ self.K_trunc        # K^s
                z_s = K_power @ z0               # shape (20,)
                preds.append(z_s[:2])            # x,y
            prediction_result[object_id] = np.vstack(preds)

        return prediction_result

    # One-step prediction forward pass
    def _forward(self, state):
        o_next = self.K_trunc @ state
        next_xy = o_next[:2]
        geo_vec = get_12dim_obstacle_vector(next_xy[0], next_xy[1], polygons=self.obstacle_polygons)
        geo_feat = self.encoder(torch.tensor(geo_vec, dtype=torch.float32).unsqueeze(0)).detach().cpu().numpy().squeeze()
        return np.concatenate([o_next[:16], geo_feat])

# Visualize input history and predicted future trajectory
def visualize_history_and_prediction(history, prediction, polygons):
    fig, ax = plt.subplots(figsize=(6,6))
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, facecolor='gray', edgecolor='black')
    ax.plot(history[:,0], history[:,1], 'bo-', label='History')
    ax.plot(prediction[:,0], prediction[:,1], 'r*', markersize=12, label='Predicted')
    ax.set_xlim(-3, 0)
    ax.set_ylim(-5, 0)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    ax.set_title('History and Predicted Points')
    plt.show()

# Load predictor and generate trajectories for various truncation levels
predictor = KoopmanPredictor(prediction_len=12, model_file='koopy_3.pkl', eig_top_k=20)

history_positions = np.array([
    [1.0, 1],
    [1.1, 1],
    [1.2, 1],
    [1.3, 1],
    [1.4, 1],
    [1.5, 1],
    [1.6, 1],
    [1.7, 1]
])

ks = range(1, 21)
all_trajs = []
for k in ks:
    predictor.K_trunc = predictor._truncate_K(predictor.K, k)
    pred = predictor({'agent_0': history_positions})['agent_0']
    all_trajs.append(pred)

# Plot predicted trajectories for each truncation level
import matplotlib.cm as cm
import matplotlib.colors as mcolors

cmap = plt.colormaps.get_cmap('coolwarm_r')
norm = mcolors.Normalize(vmin=1, vmax=20)
fig, ax = plt.subplots(figsize=(7, 6))

# 장애물
for poly in predictor.obstacle_polygons:
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=1.2)

# 과거 궤적
ax.plot(history_positions[:, 0], history_positions[:, 1], 'ko-', label='History',
        markersize=6, linewidth=3.0)


# 예측 궤적
for k, traj in zip(ks, all_trajs):
    color = cmap(norm(k))
    ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=4.0, color=color)
    ax.plot(traj[-1, 0], traj[-1, 1], 'o', color=color, markersize=6)

    if k == 1:
        ax.text(traj[-1, 0] + 0.05, traj[-1, 1], '1', fontsize=20, color='black')
    elif k == 2:
        ax.text(traj[-1, 0] + 0.05, traj[-1, 1], '2', fontsize=20, color='black')
    elif k == 3:
        ax.text(traj[-1, 0] + 0.05, traj[-1, 1], '3', fontsize=20, color='black')
    elif k == 4:
        ax.text(traj[-1, 0] + 0.05, traj[-1, 1], '4', fontsize=20, color='black')
    elif k == 5:
        ax.text(traj[-1, 0] + 0.05, traj[-1, 1], '20', fontsize=20, color='black')


ax.set_xlim(0, 5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal', 'box')
# ax.axis('off')
ax.set_xticks([])
ax.set_yticks([])
# ax.set_xlabel('X [m]', fontsize=14)
# ax.set_ylabel('Y [m]', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.grid(False)

# 컬러바
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, ticks=[1, 5, 10, 15, 20])
cbar.set_label("Mode count (n)", fontsize=22)
cbar.ax.tick_params(labelsize=11)

# 범례, 제목
ax.legend(fontsize=20)
# ax.set_title('Eigenmode-Truncated Koopman Prediction', fontsize=16)

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()


