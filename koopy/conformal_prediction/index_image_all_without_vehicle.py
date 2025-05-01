import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from cp.adaptive_cp import AdaptiveConformalPredictionModule
from score_functions import stepwise_displacement_error
from visualization_utils import draw_map3, visualize_cp_result2, visualize_tracking_result, visualize_prediction_result, visualize_controller_info
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
def main(
    test_dirpath: str,
    models: list[str],
    cp: str,
    n_pedestrians: int,
    map_size: list[float],
    bg_img_path: str | None = None,
    output_root: str = './paperimage_compare'
):
    """
    Compare multiple models' 12-step forecasts against ground truth for each timestep.

    :param test_dirpath:    path to one scenario’s folder of .npy files
    :param models:         list of model names (e.g. ['linear','gp',…])
    :param cp:             conformal-prediction tag (e.g. 'adaptive')
    :param n_pedestrians:  number of agents in this scenario
    :param map_size:       [x_max, y_max, x_min, y_min]
    :param bg_img_path:    optional background-image file
    :param output_root:    root folder to write per-scenario images
    """
    # --- pick the right obstacle set ---
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

    scenario = os.path.basename(os.path.dirname(test_dirpath.rstrip(os.sep)))
    print(f"Comparing models for scenario '{scenario}'…")

    # --- load predictions + ground truth ---
    all_preds = {
        m: np.load(os.path.join(test_dirpath, f"{scenario}_{m}_predictions.npy"))
        for m in models
    }
    y_true = np.load(os.path.join(test_dirpath, f"{scenario}_targets.npy"))

    T = y_true.shape[0]
    t_begin = 10
    t_end = min(t_begin + 120, T - 1)

    # --- prepare output directory ---
    out_dir = os.path.join(output_root, scenario, cp)
    os.makedirs(out_dir, exist_ok=True)

    # --- loop over time steps ---
    for t in range(t_begin, t_end):
        fig, ax, bg_img, extent = draw_map3(*map_size, bg_img_path=bg_img_path)
        ax.axis('off')
        ax.margins(0)
        draw_obstacles(ax, current_obstacles)

        # 1) plot each model’s dashed forecast
        for style, m in zip(plt.rcParams['axes.prop_cycle'], models):
            color = style['color']
            preds = {
                i: all_preds[m][t, :, i]
                for i in range(n_pedestrians)
                if not np.isnan(y_true[t, :, i]).any()
            }
            visualize_prediction_result(
                preds,
                ax,
                color=color,
                linestyle='dashed',
                label=m
            )

        # 2) plot the solid ground-truth
        true_preds = {
            i: y_true[t, :, i]
            for i in range(n_pedestrians)
            if not np.isnan(y_true[t, :, i]).any()
        }
        visualize_prediction_result(
            true_preds,
            ax,
            color='black',
            linestyle='solid',
            label='ground truth'
        )

        ax.legend(loc='upper left')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()

        # 3) crop to just the axes and save
        img = np.array(fig.canvas.renderer.buffer_rgba())
        bbox = ax.get_window_extent().astype(int)
        h = img.shape[0]
        crop = img[h - bbox.y1 : h - bbox.y0, bbox.x0 : bbox.x1]
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
        out_path = os.path.join(out_dir, f"{t - t_begin:03d}.jpg")
        cv2.imwrite(out_path, cv2.resize(bgr, (800, 800)))

        plt.close(fig)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cp',     default='adaptive')
    parser.add_argument('--models', nargs='+',
                        default=['linear','gp','eigen',
                                 'koopman_clu_geo','koopman_clu_vel',
                                 'koopman_enc','trajectron'])
    args = parser.parse_args()

    test_dirs = [
        'lobby2/biwi_eth/test',
        'lobby2/univ/test/001',
        'lobby2/univ/test/003',
        'lobby2/biwi_hotel/test',
        'lobby2/crowds_zara01/test',
        'lobby2/crowds_zara02/test'
    ]
    map_sizes = [
        [18.42, 17.21, -8.69, -6.17],
        [15.4369, 13.8542, -0.1747, -0.2222],
        [15.4369, 13.8542, -0.1747, -0.2222],
        [6.35,   4.31,   -3.25,  -10.31],
        [15.1324, 13.3864, -0.0210,  0.7613],
        [15.5584, 14.9427, -0.3578,  0.7263]
    ]
    n_peds = [367, 415, 434, 420, 148, 204]
    bg_imgs = [
        "ethucyimages/eth.png",
        "ethucyimages/students_003.jpg",
        "ethucyimages/students_003.jpg",
        "ethucyimages/hotel.png",
        "ethucyimages/crowds_zara01.jpg",
        "ethucyimages/crowds_zara02.jpg"
    ]

    for td, n, ms, bg in zip(test_dirs, n_peds, map_sizes, bg_imgs):
        main(
            test_dirpath=td,
            models=args.models,
            cp=args.cp,
            n_pedestrians=n,
            map_size=ms,
            bg_img_path=bg
        )