import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from cp.adaptive_cp import AdaptiveConformalPredictionModule
from utils.score_functions import stepwise_displacement_error
from utils.visualization_utils import draw_map2, visualize_cp_result, visualize_tracking_result, visualize_prediction_result, visualize_controller_info
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
    'obstacle3': [(10.0, 6.5), (11.9, 6.5), (11.9, 11.3), (10.0, 11.3), (10.0, 6.5)],
    'obstacle4': [(-0.35779069, 0.72625721), (15.55842276, 0.72625721), (15.55842276, 2.9), (-0.35779069, 2.9), (-0.35779069, 0.72625721)]
}
obstacles_meter_uni = {
    'obstacle1': [(3.8, 13.85420137), (6.0, 11.5), (10.0, 13.85420137), (3.8, 13.85420137)],
    'obstacle2': [(-0.17468604, 9.8), (-0.17468604, 11.5), (4.0, 11.5), (3.5, 9.8), (-0.17468604, 9.8)]
}

def draw_obstacles(ax, obstacles, color='gray', alpha=0.5):
    """Matplotlib Axes 위에 장애물 다각형을 그립니다."""
    for obs_key, coords in obstacles.items():
        poly = Polygon(coords, closed=True, color=color, alpha=alpha)
        ax.add_patch(poly)

def main(model, cp, n_pedestrians, test_dirpath, goal_x, goal_y, init_x, init_y, map_size):
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

    robot_radius = 0.4
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

            while t_step < t_end and not done:
                # 예측 결과 생성
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

                # 유효 보행자 업데이트
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

            # # video start
            # video_root_dir = './videos'
            # video_dir = os.path.join(video_root_dir, name, '{}_{}'.format(model, cp))
            # os.makedirs(video_dir, exist_ok=True)
            # image_list = []
            # print('visualizing the result at {}'.format(video_dir))
            # for t, y_model in enumerate(test_set_y_model[t_begin:120+t_begin]):
            #     intervals = buffer_intervals[t]
            #     info = buffer_controller_info[t]
            #     robot_pose = buffer_robot_poses[t]
            #     track_res = buffer_tracking_res[t]
            #     pred_res = buffer_prediction_res[t]
            #     pred_true_res = buffer_prediction_true_res[t]

            #     plt.clf()
            #     plt.cla()
            #     fig, ax = draw_map2(*map_size)
                
            #     # 장애물 그리기
            #     draw_obstacles(ax, current_obstacles)

            #     selected_steps = [1, 6, 11]
            #     visualize_tracking_result(track_res, ax)
            #     visualize_controller_info(info, ax)
            #     visualize_prediction_result(pred_res, ax, color='blue', linestyle='dashed')
            #     visualize_prediction_result(pred_true_res, ax, color='black', linestyle='solid')
            #     visualize_cp_result(intervals, pred_res, selected_steps, ax)

            #     position_x, position_y, orientation_z = robot_pose

            #     img_rotated = OffsetImage(
            #         Image.fromarray(ndimage.rotate(robot_img, orientation_z / np.pi * 180. - 90.)),
            #         zoom=0.05
            #     )
            #     ax.add_artist(AnnotationBbox(
            #         img_rotated, (position_x, position_y), frameon=False, zorder=200))
            #     dx = 0.8 * np.cos(orientation_z)
            #     dy = 0.8 * np.sin(orientation_z)
            #     ax.arrow(position_x, position_y, dx, dy, head_width=0.05, head_length=0.1,
            #              fc='black', ec='black', zorder=120)
            #     ax.scatter([goal_x], [goal_y], color='tab:red', marker='s', s=80, label='goal')
            #     ax.set_title('test scenario {} ({} + {})'.format(name, model, cp))
            #     fig.canvas.draw()
            #     image = np.array(fig.canvas.renderer.buffer_rgba())
            #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            #     image_list.append(image)
            #     plt.close(fig)
            # video_path = os.path.join(video_dir, 'test_scenario_{}_{}_{}_{}.mp4'.format(name, n_pedestrians, model, cp))
            # if image_list:
            #     height, width, _ = image_list[0].shape
            #     fps = 2.5
            #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #     video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            #     for img in image_list:
            #         video_writer.write(img)
            #     video_writer.release()

            #     # video end


    # === 시뮬레이션 종료 후 결과 요약 ===
    total_calls = len(buffer_controller_info)
    infeasible_calls = sum(1 for info in buffer_controller_info if not info.get('feasible', False))
    infeasible_rate = infeasible_calls / total_calls if total_calls > 0 else 0.0

    # cost 구성 요소 및 도착 시간 모으기
    inter_costs, ctrl_costs, term_costs, total_costs = [], [], [], []
    goal_arrivals = []

    for info in buffer_controller_info:
        if info.get('feasible', False):
            c = info['cost_components']
            inter_costs.append(c['intermediate'])
            ctrl_costs.append(c['control'])
            term_costs.append(c['terminal'])
            total_costs.append(c['total'])

            gtime = info.get('goal_arrival_time', -1)
            if gtime >= 0:
                goal_arrivals.append(gtime)

    # 안전한 평균 함수
    def safe_mean(x): return np.mean(x) if x else None
    def safe_std(x): return np.std(x) if x else None

    avg_inter = safe_mean(inter_costs)
    avg_ctrl = safe_mean(ctrl_costs)
    avg_term = safe_mean(term_costs)
    avg_total = safe_mean(total_costs)
    std_total = safe_std(total_costs)
    avg_goal_arrival = safe_mean(goal_arrivals)


    return name, model, avg_inter, avg_ctrl, avg_term, avg_total, std_total, infeasible_rate, avg_goal_arrival



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='linear')
    parser.add_argument('--cp', default='adaptive')
    models = ['linear', 'koopy', 'trajectron', 'eigen','gp']
    test_dirpaths =[
        '../lobby2/biwi_eth/test',
        '../lobby2/univ/test/001',
        '../lobby2/univ/test/003',
        '../lobby2/biwi_hotel/test',
        '../lobby2/crowds_zara01/test',
        '../lobby2/crowds_zara02/test'
    ]
    map_sizes = [ # x max, y max, x min, y min
        [25, 20, -5, -10],
        [15, 15, -1, -1],
        [15, 15, -1, -1],
        [8, 8, -8, -20],
        [15, 15, 0, 0],
        [15, 15, 0, 0]
    ]
    n_pedestrian = [367, 415, 434, 420, 148, 204]
    goal_x = [15, 0, 0, 0, 1, 13]
    goal_y = [10, 6, 5, 5, 3, 12]
    init_x = [0, 12, 12, 2, 14, 3]
    init_y = [0, 10, 9, -18, 12, 1]
    num_processes = len(models)
    args = parser.parse_args()
    # multiprocessing을 이용하여 여러 모델을 병렬 실행
    for i, j, k, l, m, n, o in reversed(list(zip(n_pedestrian, test_dirpaths, goal_x, goal_y, init_x, init_y, map_sizes))):
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(main, list(zip(
                models,
                [args.cp] * num_processes,
                [i] * num_processes,
                [j] * num_processes,
                [k] * num_processes,
                [l] * num_processes,
                [m] * num_processes,
                [n] * num_processes,
                [o] * num_processes
            )))
            print("\n=== Whole Scenario Results Summary ===")
            print("{:<12} {:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>15} {:>12}".format(
                "Scenario", "Model", "AvgInter", "AvgCtrl", "AvgTerm", "AvgTotal", "StdTotal", "Infeasible Rate", "GoalTime"))

            for res in results:
                name, model, ai, ac, at, atot, stdtot, infeas, gtime = res
                f = lambda x: f"{x:.2f}" if x is not None else "N/A"
                print("{:<12} {:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>15} {:>12}".format(
                    name, model, f(ai), f(ac), f(at), f(atot), f(stdtot), f"{infeas:.2%}", f(gtime)
                ))
