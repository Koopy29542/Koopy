import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
# from adaptive_cp import AdaptiveConformalPredictionModule
from cp.adaptive_cp import AdaptiveConformalPredictionModule
from utils.score_functions import stepwise_displacement_error
from utils.visualization_utils import draw_map2, visualize_cp_result, visualize_tracking_result, visualize_prediction_result, visualize_controller_info
from env import Environment
# from control.sampling_based_mpc import SamplingBasedMPC
from control.grid_solver import GridMPC
from PIL import Image
from scipy import ndimage
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import multiprocessing

def main(model, cp, n_pedestrians, test_dirpath, goal_x, goal_y, init_x, init_y, map_size):
    """
    :param model: prediction model
    :param cp: type of conformal prediction
    :param n_pedestrians: total number of pedestrians (to check for valid ones)
    :param test_dirpath: directory containing the test files
    :param goal_x: goal x coordinate
    :param goal_y: goal y coordinate
    :param init_x: initial x coordinate for the robot
    :param init_y: initial y coordinate for the robot
    :param map_size: size of the map for visualization (e.g., [15,15,0,0])
    :return:
    """

    target_miscoverage_level = 0.1
    calibration_set_size = 30.

    if cp == 'split':
        # split conformal prediction (but with 'shifted' calibration sets)
        step_size = 0.
    elif cp == 'adaptive':
        # adaptive conformal prediction
        step_size = 0.05
    else:
        raise ValueError('invalid type of conformal prediction; have to choose from split & adaptive')

    stat_dir = '../CP_stats'
    os.makedirs(stat_dir, exist_ok=True)

    dt = 0.1
    prediction_len = 12

    # ★★★ Added: Cost calculation variables (robot & pedestrian radii, collision threshold, and cost accumulators)
    robot_radius = 0.5
    pedestrian_radius = 1
    collision_distance = robot_radius + pedestrian_radius

    total_control_cost = 0.0
    infeasible_count = 0
    collision_count = 0
    total_steps = 0

    robot_img = Image.open(os.path.join(os.path.dirname(__file__), "real_robot.png"))
    t_begin = 48
    t_step = t_begin
    t_end = 108
    max_valid_pedestrians = 4
    for file in os.listdir(test_dirpath):
        if file.endswith('.txt'):

            name, _ = os.path.splitext(file)
            print('running scenario {} from {}...'.format(name, test_dirpath))

            test_set_y_model = np.load(os.path.join(test_dirpath, name + '_{}_predictions.npy'.format(model)))
            test_set_y = np.load(os.path.join(test_dirpath, name + '_targets.npy'.format(model)))

            # Since the offline calibration set is significantly different from the online data distribution, we will ignore it now.
            max_interval_lengths = 0.5 * dt * np.arange(1, prediction_len + 1)
            offline_calibration_set = {i: [] for i in range(prediction_len)}

            cp_module = AdaptiveConformalPredictionModule(target_miscoverage_level=0.1,
                                                          step_size=0.05,
                                                          n_scores=prediction_len,
                                                          max_interval_lengths=max_interval_lengths,
                                                          sample_size=20,
                                                          offline_calibration_set=offline_calibration_set
                                                          )

            controller = GridMPC(n_steps=prediction_len, dt=dt)

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
            for i in range(n_pedestrians):
                # Check if all values are NaN at the current timestep
                if len(valid_pedestrians) >= max_valid_pedestrians:
                    break
                has_nan_now = np.isnan(test_set_y[t_step, :, i]).any()
                if not has_nan_now:
                    valid_pedestrians.append(i)
                    detected_pedestrians.append(i)  # Mark pedestrian as detected

                # Stop if we reach the max limit
                if len(valid_pedestrians) >= max_valid_pedestrians:
                    break

            tracking_res = environment._get_obs(valid_pedestrians)
            position_x, position_y, orientation_z = robot_pose
            linear_x, angular_z = velocity
            
            while not done:
                # prediction set
                # pedestrian set to valid ones for now
                prediction_res = {
                    i: test_set_y_model[t_step, :, i] for i in valid_pedestrians
                }

                prediction_true_res = {
                    i: test_set_y[t_step, :, i] for i in valid_pedestrians
                }
                # Update the conformal prediction module
                confidence_intervals = cp_module.update(tracking_res, prediction_res)

                velocity, info = controller(pos_x=position_x,
                                            pos_y=position_y,
                                            orientation_z=orientation_z,
                                            linear_x=linear_x,
                                            angular_z=angular_z,
                                            boxes=[],       # TODO
                                            predictions=prediction_res,
                                            confidence_intervals=confidence_intervals,
                                            goal=goal
                                            )

                if not info['feasible']:
                    # ★★★ Added: Count infeasible steps and set velocity to zero if not feasible
                    infeasible_count += 1
                    velocity = np.array([0., 0.])
                #     print('linear_x={} / angular_z={} (infeasible)'.format(*velocity))
                # else:
                #     print('linear_x={} / angular_z={} (feasible)'.format(*velocity))

                # ★★★ Added: Accumulate control cost (L2 norm of control inputs) and count steps
                total_control_cost += 0.01 * np.linalg.norm(velocity)**2
                total_steps += 1

                robot_pose, done = environment.step(velocity)
                position_x, position_y, orientation_z = robot_pose
                t_step += 1

                # ★★★ Added: Collision check between the robot and pedestrians
                if isinstance(tracking_res, dict):
                    pedestrian_positions = list(tracking_res.values())
                else:
                    pedestrian_positions = tracking_res
                for ped_pos in pedestrian_positions:
                    if np.linalg.norm(np.array(robot_pose[:2]) - np.array(ped_pos[:2])) < collision_distance:
                        collision_count += 1

                valid_pedestrians = []
                for i in range(n_pedestrians):
                    # Check if all values are NaN at the current timestep
                    if len(valid_pedestrians) >= max_valid_pedestrians:
                        break
                    all_nan_now = np.isnan(test_set_y[t_step, :, i]).all()

                    # If pedestrian was previously detected but is now completely NaN, remove from the list
                    if i in detected_pedestrians and all_nan_now:
                        detected_pedestrians.remove(i)
                        continue  # Skip adding this pedestrian

                    # If pedestrian was detected before, keep them in the list
                    if i in detected_pedestrians:
                        valid_pedestrians.append(i)
                        continue  # Skip further checks
                    
                    # Check if the pedestrian has any NaN values at the current timestep
                    has_nan_now = np.isnan(test_set_y[t_step, :, i]).any()

                    # Check if the pedestrian has had any non-NaN values in past timesteps (0:t_step)
                    had_non_nan_past = np.isnan(test_set_y[:t_step, :, i]).all()

                    # If it's a new pedestrian, include only if they have no NaNs now and had only NaNs before
                    if not has_nan_now and had_non_nan_past:
                        valid_pedestrians.append(i)
                        detected_pedestrians.append(i)  # Mark pedestrian as detected

                    # Stop if we reach the max limit
                    if len(valid_pedestrians) >= max_valid_pedestrians:
                        break
                tracking_res = environment._get_obs(valid_pedestrians)
                buffer_robot_poses.append(robot_pose)
                buffer_intervals.append(confidence_intervals)
                buffer_controller_info.append(info)
                buffer_prediction_res.append(prediction_res)
                buffer_prediction_true_res.append(prediction_true_res)
                buffer_tracking_res.append(tracking_res)

            # ★★★ Added: Final cost computations after simulation ends
            terminal_cost = np.linalg.norm(np.array(robot_pose[:2]) - goal[:2])
            total_cost = terminal_cost + total_control_cost
            infeasible_rate = infeasible_count / total_steps if total_steps > 0 else 0

            print(f"Running model: {model}")
            print(f"Cost metrics for scenario {name}:")
            print(f"  Terminal cost: {terminal_cost}")
            print(f"  Control cost: {total_control_cost}")
            print(f"  Total cost: {total_cost}")
            print(f"  Infeasible rate: {infeasible_rate}")
            print(f"  Collision count: {collision_count}")
            print(f"  Total steps: {total_steps}")

            video_root_dir = '../videos'
            video_dir = os.path.join(video_root_dir, name, '{}_{}'.format(model, cp))
            os.makedirs(video_dir, exist_ok=True)
            image_list = []
            print('visualizing the result at {}'.format(video_dir))
            for t, y_model in enumerate(test_set_y_model[t_begin:120+t_begin]):
                intervals = buffer_intervals[t]
                info = buffer_controller_info[t]
                robot_pose = buffer_robot_poses[t]
                track_res = buffer_tracking_res[t]
                pred_res = buffer_prediction_res[t]
                pred_true_res = buffer_prediction_true_res[t]

                _, n_pedestrians, _ = y_model.shape
                plt.clf(), plt.cla()
                fig, ax = draw_map2(*map_size)

                selected_steps = [1, 6, 11]

                visualize_tracking_result(track_res, ax)
                visualize_controller_info(info, ax)
                visualize_prediction_result(pred_res, ax, color='blue', linestyle='dashed')
                visualize_prediction_result(pred_true_res, ax, color='black', linestyle='solid')
                visualize_cp_result(intervals, pred_res, selected_steps, ax)

                position_x, position_y, orientation_z = robot_pose

                img_rotated = OffsetImage(Image.fromarray(ndimage.rotate(
                    robot_img, orientation_z / np.pi * 180. - 90.)), zoom=0.05)

                ax.add_artist(AnnotationBbox(
                    img_rotated, (position_x, position_y), frameon=False, zorder=200))
                # ax.scatter([position_x], [position_y], color='tab:blue', marker='o', s=80, label='robot')
                dx = 0.8 * np.cos(orientation_z)
                dy = 0.8 * np.sin(orientation_z)
                ax.arrow(position_x, position_y, dx, dy, head_width=0.05, head_length=0.1, fc='black', ec='black',
                         zorder=120)

                ax.scatter([goal_x], [goal_y], color='tab:red', marker='s', s=80, label='goal')

                ax.set_title('test scenario {} ({} + {})'.format(name, model, cp))
                # fig.savefig(os.path.join(video_dir, '{:03d}.png'.format(t)))
                fig.canvas.draw()
                image = np.array(fig.canvas.renderer.buffer_rgba())  # Get image array from figure
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR for OpenCV
                image_list.append(image)
                plt.close(fig)
            video_path = os.path.join(video_dir, 'test_scenario_{}_{}_{}_{}.mp4'.format(name, n_pedestrians, model, cp))
            if image_list:
                height, width, _ = image_list[0].shape
                fps = 2.5  # Set FPS (frames per second)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                for img in image_list:
                    video_writer.write(img)
                video_writer.release()
            '''
            np.save(os.path.join(stat_dir, 'interval_{}_{}_{}.npy'.format(model, cp, name)), interval_lengths)
            np.save(os.path.join(stat_dir, 'score_{}_{}_{}.npy'.format(model, cp, name)), scores)
            np.save(os.path.join(stat_dir, 'coverage_{}_{}_{}.npy'.format(model, cp, name)), cumul_coverage)
            np.save(os.path.join(stat_dir, 'alpha_{}_{}_{}.npy'.format(model, cp, name)), effective_alphas)
            '''

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='linear')
    parser.add_argument('--cp', default='adaptive')
    models = ['koopman']
    test_dirpaths = ['../lobby3/test']
    map_sizes=[[15,15,0,0]]
    n_pedestrian=[204]
    goal_x=[14]
    goal_y=[12]
    init_x=[1]
    init_y=[2]
    num_processes=len(models)
    args = parser.parse_args()
    for i,j,k,l,m,n,o in reversed(list(zip(n_pedestrian,test_dirpaths,goal_x,goal_y,init_x,init_y,map_sizes))):
        main(models[0],args.cp,i,j,k,l,m,n,o)