import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from cp.adaptive_cp import AdaptiveConformalPredictionModule
from utils.score_functions import stepwise_displacement_error
from utils.visualization_utils import draw_map3, visualize_cp_result2, visualize_tracking_result, visualize_prediction_result, visualize_controller_info
from env import Environment
from control.grid_solver import GridMPC
from PIL import Image
from scipy import ndimage
import cv2
import multiprocessing

# 장애물 좌표 (예시)
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
    'obstacle3': [(10.0, 7.5), (11.9, 7.5), (11.9, 11.3), (10.0, 11.3), (10.0, 7.5)],
    'obstacle4': [(-0.35779069, 0.72625721), (15.55842276, 0.72625721), (15.55842276, 2.9), (-0.35779069, 2.9), (-0.35779069, 0.72625721)]
}
obstacles_meter_uni = {
    'obstacle1': [(3.8, 13.85420137), (6.0, 11.5), (10.0, 13.85420137), (3.8, 13.85420137)],
    'obstacle2': [(-0.17468604, 9.8), (-0.17468604, 11.5), (4.0, 11.5), (3.5, 9.8), (-0.17468604, 9.8)]
}

def draw_obstacles(ax, obstacles, color='blue', alpha=0.5):
    """Matplotlib Axes 위에 장애물 다각형을 그립니다."""
    for obs_key, coords in obstacles.items():
        poly = Polygon(coords, closed=True, color=color, alpha=alpha)
        ax.add_patch(poly)

def main(model, cp, n_pedestrians, test_dirpath, goal_x, goal_y, init_x, init_y, map_size, bg_img_path=None):
    """
    :param model: 예측 모델 (예: 'linear', 'gp' 등)
    :param cp: conformal prediction 유형 ('split' 또는 'adaptive')
    :param n_pedestrians: 전체 보행자 수 (유효성 확인용)
    :param test_dirpath: 테스트 파일이 있는 디렉토리 경로
    :param goal_x: 목표지점 x 좌표
    :param goal_y: 목표지점 y 좌표
    :param init_x: 초기 로봇 x 좌표
    :param init_y: 초기 로봇 y 좌표
    :param map_size: 지도 시각화 크기 (예: [60, 40, -60, -40])
    :return:
    """
    target_miscoverage_level = 0.1
    calibration_set_size = 30.0

    if cp == 'split':
        step_size = 0.0
    elif cp == 'adaptive':
        step_size = 0.05
    else:
        raise ValueError('invalid type of conformal prediction; choose from split & adaptive')

    stat_dir = './CP_stats'
    os.makedirs(stat_dir, exist_ok=True)

    dt = 0.4
    prediction_len = 12

    robot_radius = 0.5
    pedestrian_radius = 1
    collision_distance = robot_radius + pedestrian_radius

    total_control_cost = 0.0
    infeasible_count = 0
    collision_count = 0
    total_steps = 0

    # 로봇 이미지 로드 (실제 파일 경로에 real_robot.png가 있어야 함)
    robot_img = Image.open(os.path.join(os.path.dirname(__file__), "real_robot.png"))
    t_begin = 10
    t_step = t_begin
    t_end = 170

    max_valid_pedestrians = 2

    # 테스트 경로 이름에 따라 해당 장애물 정보를 선택합니다.
    if 'biwi_eth' in test_dirpath:
        current_obstacles = obstacles_meter_eth
    elif 'biwi_hotel' in test_dirpath:
        current_obstacles = obstacles_meter_hotel
    elif 'crowds_zara01' in test_dirpath:
        current_obstacles = obstacles_meter_zara01
    elif 'crowds_zara02' in test_dirpath:
        current_obstacles = obstacles_meter_zara02
    elif 'univ' in test_dirpath:
        current_obstacles = obstacles_meter_uni
    else:
        current_obstacles = {}

    for file in os.listdir(test_dirpath):
        if file.endswith('.txt'):
            name, _ = os.path.splitext(file)
            print('running scenario {} from {}...'.format(name, test_dirpath))

            test_set_y_model = np.load(os.path.join(test_dirpath, name + '_{}_predictions.npy'.format(model)))
            test_set_y = np.load(os.path.join(test_dirpath, name + '_targets.npy'.format(model)))

            T = test_set_y.shape[0]
            if T <= t_begin:
                print(f"Data is too short for scenario {name}, skipping.")
                continue
            t_end = min(t_end, T - 1)

            max_interval_lengths = 0.5 * dt * np.arange(1, prediction_len + 1)
            offline_calibration_set = {i: [] for i in range(prediction_len)}

            cp_module = AdaptiveConformalPredictionModule(
                target_miscoverage_level=target_miscoverage_level,
                step_size=step_size,
                n_scores=prediction_len,
                max_interval_lengths=max_interval_lengths,
                sample_size=20,
                offline_calibration_set=offline_calibration_set
            )

            controller = GridMPC(n_steps=prediction_len, dt=dt)

            init_robot_pose = np.array([init_x, init_y, np.pi / 2.0])
            goal = np.array([goal_x, goal_y])
            velocity = np.array([0.0, 0.0])

            environment = Environment(
                filepath=os.path.join(test_dirpath, name + '.npy'),
                dt=dt,
                init_robot_pose=init_robot_pose,
                n_pedestrians=max_valid_pedestrians,
                t_begin=t_begin,
                t_end=t_end
            )

            buffer_intervals = []
            buffer_robot_poses = []
            buffer_controller_info = []
            buffer_tracking_res = []
            buffer_prediction_res = []
            buffer_prediction_true_res = []
            done = False
            robot_pose = environment.reset()
            detected_pedestrians = []
            valid_pedestrians = []

            # 초기 유효 보행자 확인
            for i in range(n_pedestrians):
                if len(valid_pedestrians) >= max_valid_pedestrians:
                    break
                has_nan_now = np.isnan(test_set_y[t_step, :, i]).any()
                if not has_nan_now:
                    valid_pedestrians.append(i)
                    detected_pedestrians.append(i)
                if len(valid_pedestrians) >= max_valid_pedestrians:
                    break

            tracking_res = environment._get_obs(valid_pedestrians)
            position_x, position_y, orientation_z = robot_pose
            linear_x, angular_z = velocity
            x_vals = test_set_y[..., 0]
            y_vals = test_set_y[..., 1]
            finite_x = x_vals[np.isfinite(x_vals)]
            finite_y = y_vals[np.isfinite(y_vals)]
            x_min = finite_x.min()
            x_max = finite_x.max()

            # y‐axis:
            y_min = finite_y.min()
            y_max = finite_y.max()
            while t_step < t_end and not done:
                prediction_res = {
                    i: test_set_y_model[t_step, :, i] for i in valid_pedestrians
                }
                prediction_true_res = {
                    i: test_set_y[t_step, :, i] for i in valid_pedestrians
                }
                confidence_intervals = cp_module.update(tracking_res, prediction_res)

                velocity, info = controller(
                    pos_x=position_x,
                    pos_y=position_y,
                    orientation_z=orientation_z,
                    linear_x=linear_x,
                    angular_z=angular_z,
                    boxes=[],  # 장애물 정보를 직접 전달할 수 있으나, 본 예제에서는 시각화에만 사용
                    predictions=prediction_res,
                    confidence_intervals=confidence_intervals,
                    goal=goal
                )

                if not info['feasible']:
                    infeasible_count += 1
                    velocity = np.array([0.0, 0.0])

                total_control_cost += 0.01 * np.linalg.norm(velocity) ** 2
                total_steps += 1

                robot_pose, done = environment.step(velocity)
                position_x, position_y, orientation_z = robot_pose
                t_step += 1

                if isinstance(tracking_res, dict):
                    pedestrian_positions = list(tracking_res.values())
                else:
                    pedestrian_positions = tracking_res
                for ped_pos in pedestrian_positions:
                    if np.linalg.norm(np.array(robot_pose[:2]) - np.array(ped_pos[:2])) < collision_distance:
                        collision_count += 1

                valid_pedestrians = []
                for i in range(n_pedestrians):
                    if len(valid_pedestrians) >= max_valid_pedestrians:
                        break
                    if t_step >= T:
                        break
                    all_nan_now = np.isnan(test_set_y[t_step, :, i]).all()
                    if i in detected_pedestrians and all_nan_now:
                        detected_pedestrians.remove(i)
                        continue
                    if i in detected_pedestrians:
                        valid_pedestrians.append(i)
                        continue
                    has_nan_now = np.isnan(test_set_y[t_step, :, i]).any()
                    had_non_nan_past = np.isnan(test_set_y[:t_step, :, i]).all()
                    if not has_nan_now and had_non_nan_past:
                        valid_pedestrians.append(i)
                        detected_pedestrians.append(i)
                    if len(valid_pedestrians) >= max_valid_pedestrians:
                        break

                tracking_res = environment._get_obs(valid_pedestrians)
                buffer_robot_poses.append(robot_pose)
                buffer_intervals.append(confidence_intervals)
                buffer_controller_info.append(info)
                buffer_prediction_res.append(prediction_res)
                buffer_prediction_true_res.append(prediction_true_res)
                buffer_tracking_res.append(tracking_res)

            # 영상 파일 생성
            video_root_dir = './videoimage'
            video_dir = os.path.join(video_root_dir, name, f'{model}_{cp}')
            os.makedirs(video_dir, exist_ok=True)

            # ← CHANGED: set up VideoWriter instead of saving individual JPEGs
            fps = 1.0 / dt
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(video_dir, f'{name}_{model}_{cp}.mp4')
            output_size = (800, 800)
            writer = cv2.VideoWriter(video_path, fourcc, fps, output_size)
            print('visualizing & writing video → {}'.format(video_path))

            for t, y_model in enumerate(test_set_y_model[t_begin:120+t_begin]):
                intervals = buffer_intervals[t]
                info = buffer_controller_info[t]
                robot_pose = buffer_robot_poses[t]
                track_res = buffer_tracking_res[t]
                pred_res = buffer_prediction_res[t]
                pred_true_res = buffer_prediction_true_res[t]

                plt.clf()
                plt.cla()
                fig, ax, bg_img, extent = draw_map3(*map_size, bg_img_path=bg_img_path)
                ax.axis('off')
                ax.margins(0)
                draw_obstacles(ax, current_obstacles)

                selected_steps = [1, 6, 11]
                visualize_tracking_result(track_res, ax)
                visualize_controller_info(info, ax)
                visualize_prediction_result(pred_res, ax, color='blue', linestyle='dashed')
                visualize_prediction_result(pred_true_res, ax, color='black', linestyle='solid')
                visualize_cp_result2(intervals, pred_res, selected_steps,	ax, bg_img, extent)

                position_x, position_y, orientation_z = robot_pose

                img_rotated = OffsetImage(
                    Image.fromarray(ndimage.rotate(robot_img, orientation_z / np.pi * 180. - 90.)),
                    zoom=0.05
                )
                ax.add_artist(AnnotationBbox(
                    img_rotated, (position_x, position_y), frameon=False, zorder=200))
                dx = 0.8 * np.cos(orientation_z)
                dy = 0.8 * np.sin(orientation_z)
                ax.arrow(position_x, position_y, dx, dy, head_width=0.05, head_length=0.1,
                         fc='black', ec='black', zorder=120)
                ax.scatter([goal_x], [goal_y], color='tab:red', marker='s', s=80)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                fig.canvas.draw()
                image = np.array(fig.canvas.renderer.buffer_rgba())
                bbox = ax.get_window_extent()
                x0, y0, x1, y1 = map(int, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
                h = image.shape[0]
                crop = image[h - y1: h -	y0, x0: x1, :]
                frame = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
                resized = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)

                writer.write(resized)
                plt.close(fig)

            writer.release()
            print(f"Saved video → {video_path}")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='linear')
    parser.add_argument('--cp', default='adaptive')
    models = ['linear', 'gp', 'eigen', 'koopy', 'trajectron']
    images = [
        "../ethucyimages/eth.png",
        "../ethucyimages/students_003.jpg",
        "../ethucyimages/students_003.jpg",
        "../ethucyimages/hotel.png",
        "../ethucyimages/crowds_zara01.jpg",
        "../ethucyimages/crowds_zara02.jpg"
    ]
    test_dirpaths =[
        '../lobby2/biwi_eth/test',
        '../lobby2/univ/test/001',
        '../lobby2/univ/test/003',
        '../lobby2/biwi_hotel/test',
        '../lobby2/crowds_zara01/test',
        '../lobby2/crowds_zara02/test'
    ]
    map_sizes = [
        [18.42,  17.21,  -8.69,    -6.17],
        [15.4369843957, 13.8542013734,  -0.174686040989,  -0.222192273533],
        [15.4369843957, 13.8542013734,  -0.174686040989,  -0.222192273533],
        [6.35,   4.31,   -3.25,   -10.31],
        [15.13244069, 13.3864436,  -0.02104651,  0.76134018],
        [15.558422764, 14.9427441591,  -0.357790686363,  0.726257209729],
    ]

    n_pedestrian = [367, 415, 434, 420, 148, 204]
    goal_x = [15, 0, 0, 0, 1, 14]
    goal_y = [4, 6, 5, 5, 3, 12]
    init_x = [-1, 13, 15, 1.5, 14, 1]
    init_y = [0, 11, 10, -18, 12, 2]

    num_processes = len(models)
    args = parser.parse_args()

    for i, j, k, l, m, n, o,p in (list(zip(n_pedestrian, test_dirpaths, goal_x, goal_y, init_x, init_y, map_sizes,images))):
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(
                main,
                list(zip(
                    models,
                    [args.cp] * num_processes,
                    [i] * num_processes,
                    [j] * num_processes,
                    [k] * num_processes,
                    [l] * num_processes,
                    [m] * num_processes,
                    [n] * num_processes,
                    [o] * num_processes,
                    [p]* num_processes
                ))
            )
