import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from utils.score_functions import stepwise_displacement_error
from utils.visualization_utils import draw_map2, visualize_tracking_result, visualize_prediction_result, visualize_controller_info
from env import Environment
# from control.sampling_based_mpc import SamplingBasedMPC  # 사용하지 않음
from control.eacp_mpc_new import EgocentricACPMPC  # grid_solver.py 대신 사용
from PIL import Image
from scipy import ndimage
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import multiprocessing

# 장애물이 없을 때 필터링 문제 회피를 위한 dummy box (아주 먼 위치)
class DummyBox:
    def __init__(self):
        self.pos = np.array([1000.0, 1000.0])
        self.w = 0.0
        self.h = 0.0
        self.rad = 0.0

dummy_box = DummyBox()

def main(model, cp, n_pedestrians, test_dirpath, goal_x, goal_y, init_x, init_y, map_size):
    # 
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
    # dt와 예측 길이 설정
    dt = 0.4
    prediction_len = 12

    stat_dir = './CP_stats'
    os.makedirs(stat_dir, exist_ok=True)

    robot_img = Image.open(os.path.join(os.path.dirname(__file__), "real_robot.png"))



    max_valid_pedestrians = 2

    # 테스트 데이터를 담고 있는 파일들 순회
    for file in os.listdir(test_dirpath):
        if file.endswith('.txt'):
            name, _ = os.path.splitext(file)


            if name in ['crowds_zara01', 'crowds_zara02', 'students001', 'students003']:
                t_begin, t_end = 100, 260
            elif name in ['biwi_hotel']:
                t_begin, t_end = 40, 200
            elif name in ['biwi_eth']:
                t_begin, t_end = 10, 170
            
            t_step = t_begin
                

            print('running scenario {} from {} (t:{}~{})...'.format(name, test_dirpath, t_begin, t_end))

            test_set_y_model = np.load(os.path.join(test_dirpath, name + '_{}_predictions.npy'.format(model)))
            test_set_y = np.load(os.path.join(test_dirpath, name + '_targets.npy'.format(model)))

            # cp_module 관련 코드는 제거합니다.
            # controller 인스턴스 생성: EgocentricACPMPC 사용
            controller = EgocentricACPMPC(
                    n_steps=prediction_len,
                    dt=dt,
                    min_linear_x=-0.5,
                    max_linear_x=0.5,
                    min_angular_z=-0.7,
                    max_angular_z=0.7,
                    n_skip=4,
                    robot_rad=0.4,
                    calibration_set_size=12,
                    step_size=0.05,
                    miscoverage_level=0.1,
                    mode='deterministic'
                    )

            init_robot_pose = np.array([init_x, init_y, np.pi / 2.])
            goal = np.array([goal_x, goal_y])
            velocity = np.array([0., 0.])

            environment = Environment(
                filepath=os.path.join(test_dirpath, name + '.npy'),
                dt=dt,
                init_robot_pose=init_robot_pose,
                n_pedestrians=max_valid_pedestrians,
                t_begin=t_begin,
                t_end=t_end
            )

            buffer_robot_poses = []
            buffer_controller_info = []
            buffer_prediction_res = []
            buffer_prediction_true_res = []
            buffer_tracking_res = []
            done = False

            # 초기 관측: 환경 reset
            robot_pose = environment.reset()
            position_x, position_y, orientation_z = robot_pose

            # pedestrian detection 초기화
            detected_pedestrians = []
            valid_pedestrians = []
            for i in range(n_pedestrians):
                if len(valid_pedestrians) >= max_valid_pedestrians:
                    break
                has_nan_now = np.isnan(test_set_y[t_step, :, i]).any()
                if not has_nan_now :
                    valid_pedestrians.append(i)
                    detected_pedestrians.append(i)
                if len(valid_pedestrians) >= max_valid_pedestrians:
                    break

            # 부트스트래핑: internal queue에 최소 (n_steps + 1) 관측값이 쌓일 때까지 기다림
            bootstrapping = False
            boot_counter = 0





            while t_step < t_end and not done:
                # 현재 valid pedestrian에 대한 예측 및 관측 결과 생성

                raw_pred = {i: test_set_y_model[t_step, :, i] for i in valid_pedestrians}
                prediction_res = {
                    i: pred for i, pred in raw_pred.items()
                    if not np.isnan(pred).any()
                }

                prediction_true_res = {i: test_set_y[t_step, :, i] for i in valid_pedestrians}
                
                tracking_res = environment._get_obs(valid_pedestrians)
            
                err = controller.update_observations_new(tracking_res)
 
                controller.update_predictions(prediction_res)

                # 부트스트래핑 단계: 내부 큐(tracking 데이터)가 충분히 쌓일 때까지 controller 호출 전 관측만 진행
                if bootstrapping:
                    boot_counter += 1
                    t_step += 1
                    robot_pose, done = environment.step(np.array([0.0, 0.0]))  # 정지 상태로 진행
                    position_x, position_y, orientation_z = robot_pose
                    if boot_counter < controller._n_steps + 1:
                        continue
                    else:
                        bootstrapping = False
                # controller 호출  
                # boxes가 빈 리스트로 넘어가면 내부 계산 시 shape 오류가 발생하므로 dummy_box 전달
                velocity, info = controller(
                    pos_x=position_x,
                    pos_y=position_y,
                    orientation_z=orientation_z,
                    boxes=[],
                    predictions=prediction_res,
                    goal=goal
                )

                if not info['feasible']:
                    velocity = np.array([0., 0.])
                    # print('linear_x={} / angular_z={} (infeasible)'.format(*velocity))
                # else:
                    # print('linear_x={} / angular_z={} (feasible)'.format(*velocity))

                # 환경에서 한 스텝 진행
                robot_pose, done = environment.step(velocity)
                position_x, position_y, orientation_z = robot_pose
                t_step += 1

                # pedestrian 유효성 검증
                valid_pedestrians = []
                for i in range(n_pedestrians):
                    if len(valid_pedestrians) >= max_valid_pedestrians:
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
                buffer_controller_info.append(info)
                buffer_prediction_res.append(prediction_res)
                buffer_prediction_true_res.append(prediction_true_res)
                buffer_tracking_res.append(tracking_res)

            total_calls = len(buffer_controller_info)
            infeasible_calls = sum(1 for info in buffer_controller_info if not info.get('feasible', False))
            infeasible_rate = infeasible_calls / total_calls if total_calls > 0 else 0.0

            feasible_costs = [info['cost'] for info in buffer_controller_info if info.get('feasible', False)]
            if feasible_costs:
                avg_cost = np.mean(feasible_costs)
                std_cost = np.std(feasible_costs)
            else:
                avg_cost, std_cost = None, None

  
           
            video_root_dir = './videos'
            video_dir = os.path.join(video_root_dir, name, '{}_{}'.format(model, cp))
            os.makedirs(video_dir, exist_ok=True)
            image_list = []
            np.save(os.path.join(video_dir, 'robot_path.npy'), np.array(buffer_robot_poses))
            print('visualizing the result at {}'.format(video_dir))
            for t, y_model in enumerate(test_set_y_model[t_begin:120+t_begin]):
                info = buffer_controller_info[t]
                robot_pose = buffer_robot_poses[t]
                track_res = buffer_tracking_res[t]
                pred_res = buffer_prediction_res[t]
                pred_true_res = buffer_prediction_true_res[t]

              
                plt.clf(), plt.cla()
                fig, ax = draw_map2(*map_size)

                
                if 'eth' in test_dirpath.lower():
                    obstacle_set = obstacles_meter_eth
                elif 'hotel' in test_dirpath.lower():
                    obstacle_set = obstacles_meter_hotel
                elif 'zara01' in test_dirpath.lower():
                    obstacle_set = obstacles_meter_zara01
                elif 'zara02' in test_dirpath.lower():
                    obstacle_set = obstacles_meter_zara02
                elif 'uni' in test_dirpath.lower():
                    obstacle_set = obstacles_meter_uni
                else:
                    obstacle_set = {}

               
                for obstacle_coords in obstacle_set.values():
                    obs_patch = Polygon(obstacle_coords, closed=True,
                                        facecolor='gray', edgecolor='black',
                                        alpha=0.3, zorder=50)
                    ax.add_patch(obs_patch)


               
                selected_steps = [1, 6, 11]
                visualize_tracking_result(track_res, ax)
                visualize_controller_info(info, ax)
                visualize_prediction_result(pred_res, ax, color='blue', linestyle='dashed')
                visualize_prediction_result(pred_true_res, ax, color='black', linestyle='solid')

                position_x, position_y, orientation_z = robot_pose

                img_rotated = OffsetImage(
                    Image.fromarray(ndimage.rotate(robot_img, orientation_z / np.pi * 180. - 90.)),
                    zoom=0.05
                )
                ax.add_artist(AnnotationBbox(img_rotated, (position_x, position_y), frameon=False, zorder=200))
                dx = 0.8 * np.cos(orientation_z)
                dy = 0.8 * np.sin(orientation_z)
                ax.arrow(position_x, position_y, dx, dy, head_width=0.05, head_length=0.1,
                         fc='black', ec='black', zorder=120)
                ax.scatter([goal_x], [goal_y], color='tab:red', marker='s', s=80, label='goal')
                ax.set_title('test scenario {} ({} + {})'.format(name, model, cp))
                fig.canvas.draw()
                image = np.array(fig.canvas.renderer.buffer_rgba())
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                image_list.append(image)
                plt.close(fig)

            video_path = os.path.join(video_dir, 'test_scenario_{}_{}_{}_{}.mp4'.format(name, n_pedestrians, model, cp))
            if image_list:
                height, width, _ = image_list[0].shape
                fps = 2.5  # 초당 프레임 수
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                for img in image_list:
                    video_writer.write(img)
                video_writer.release()



    total_calls = len(buffer_controller_info)
    infeasible_calls = sum(1 for info in buffer_controller_info if not info.get('feasible', False))
    infeasible_rate = infeasible_calls / total_calls if total_calls > 0 else 0.0


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
    # models = ['linear', 'koopman_enc','koopman_clu_vel','koopman_clu_geo', 'trajectron', 'eigen','gp']
    models = ['koopman_enc', 'koopman_clu_geo', 'linear', 'eigen', 'trajectron', 'gp']
    test_dirpaths = [
        'lobby2/biwi_eth/test',
        'lobby2/univ/test/001',
        'lobby2/univ/test/003',
        'lobby2/biwi_hotel/test',
        'lobby2/crowds_zara01/test',
        'lobby2/crowds_zara02/test'
    ]
    map_sizes = [ # x max, y max, x min, y min
        [25, 20, -5, -10],
        [15, 15, -1, -1],
        [15, 15, -1, -1],
        [8, 8, -8, -20],
        [15, 15, 0, 0],
        [15, 15, 0, 0]
    ]

    '''
    n_pedestrian = [367, 415, 434, 420, 148, 204]
    goal_x = [15, 0, 0, 0, 1, 14]
    goal_y = [4, 6, 5, 5, 3, 12]
    init_x = [-1, 13, 15, 1.5, 14, 1]
    init_y = [0, 11, 10, -18, 12, 2]
    '''

    

    n_pedestrian = [367, 415, 434, 420, 148, 204]
    goal_x = [15, 0, 0, 0, 1, 13]
    goal_y = [10, 6, 5, 5, 3, 12]
    init_x = [0, 12, 12, 2, 14, 3]
    init_y = [0, 10, 9, -18, 12, 1]

    num_processes = len(models)
    args = parser.parse_args()

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

            '''
            print("{:<12} {:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>15} {:>12}".format(
                "Scenario", "Model", "AvgInter", "AvgCtrl", "AvgTerm", "AvgTotal", "StdTotal", "Infeasible Rate", "GoalTime"))

            for res in results:
                name, model, ai, ac, at, atot, stdtot, infeas, gtime = res
                f = lambda x: f"{x:.2f}" if x is not None else "N/A"
                print("{:<12} {:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>15} {:>12}".format(
                    name, model, f(ai), f(ac), f(at), f(atot), f(stdtot), f"{infeas:.2%}", f(gtime)
                ))
            '''
            print("{:<12} {:<10}  {:>10} {:>15} {:>12}".format(
                "Scenario", "Model", "AvgTotal", "Infeasible Rate", "GoalTime"))

            for res in results:
                name, model, ai, ac, at, atot, stdtot, infeas, gtime = res
                f = lambda x: f"{x:.2f}" if x is not None else "N/A"
                print("{:<12} {:<10} {:>10} {:>15} {:>12}".format(
                    name, model, f(atot), f"{infeas:.2%}", f(gtime)
                ))
