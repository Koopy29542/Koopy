import numpy as np
import pandas as pd

def check_in_range(x, y, x_edges, y_edges, ix, iy):
    # (x, y) in (ix, iy) bin checking
    return (
        (x >= x_edges[ix]) and (x <= x_edges[ix+1]) and
        (y >= y_edges[iy]) and (y <= y_edges[iy+1])
    )

def psi(state):
    # Lifting function: [x; y; theta]
    x = state[0, :]
    y = state[1, :]
    th = state[2, :]

    return np.vstack([
        x, y,
        np.cos(th), np.sin(th),
        x**2, y**2,
        np.cos(th)**2, np.sin(th)**2,
        x*y,
        x*np.cos(th), x*np.sin(th),
        y*np.cos(th), y*np.sin(th),
        np.cos(th)*np.sin(th),
    ])

def restoration(lifted_state):
    # Inverse mapping
    x = lifted_state[0, :]
    y = lifted_state[1, :]
    cos_theta = lifted_state[2, :]
    sin_theta = lifted_state[3, :]

    theta = np.arctan2(sin_theta, cos_theta)
    return np.vstack([x, y, theta])


data = pd.read_csv('merged_pose_400.csv', header=0)

# 1 column =agent_id, 2~4 column=x,y,theta 라고 가정
agent_ids = data.iloc[:, 1].values
xyz = data.iloc[:, 2:5].values

# agent grouping
unique_agents = np.unique(agent_ids)
grouped_data = []
for agent_id in unique_agents:
    grouped_data.append(xyz[agent_ids == agent_id, :])

# mesh, domain parameter
min_x, max_x = -2, 12
min_y, max_y = -8, 2
mesh_size = 1

x_edges = np.arange(min_x, max_x + mesh_size, mesh_size)
y_edges = np.arange(min_y, max_y + mesh_size, mesh_size)

num_x = len(x_edges) - 1
num_y = len(y_edges) - 1

Koopman = [[np.nan for _ in range(num_y)] for _ in range(num_x)]

# Koopman matrix calculation
for ix in range(num_x):
    for iy in range(num_y):
        past = []
        future = []

        for traj in grouped_data:
            if traj.shape[0] < 2:
                continue

            for j in range(traj.shape[0] - 1):
                x_curr, y_curr, th_curr = traj[j, :]
                x_next, y_next, th_next = traj[j+1, :]

                if (check_in_range(x_curr, y_curr, x_edges, y_edges, ix, iy) and 
                    check_in_range(x_next, y_next, x_edges, y_edges, ix, iy)):
                    past.append([x_curr, y_curr, th_curr])
                    future.append([x_next, y_next, th_next])

        if len(past) == 0:
            Koopman[ix][iy] = np.nan
        else:
            past = np.array(past).T
            future = np.array(future).T

            past_lifted = psi(past)
            future_lifted = psi(future)

            K = np.dot(future_lifted, np.linalg.pinv(past_lifted))
            Koopman[ix][iy] = K


print(Koopman[0][0])