import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from utils.adaptive_cp import AdaptiveConformalPredictionModule
from utils.score_functions import stepwise_displacement_error
from utils.visualization_utils import draw_map3
import cv2
import multiprocessing
video_dir = '../images/compare/test'
def datasaver(test_dirpath, n_pedestrians, map_size, bg_img_path):
    models = ['linear', 'gp', 'eigen', 'mul', 'trajectron']
    colors = ['red', 'violet', 'green', 'brown', 'blue']
    min_frame_gap = 30

    for file in os.listdir(test_dirpath):
        if file.endswith('.txt'):
            name, _ = os.path.splitext(file)
            scenario_dir = os.path.join(video_dir, name)
            os.makedirs(scenario_dir, exist_ok=True)

            test_set_y = np.load(os.path.join(test_dirpath, name + '_targets.npy'))
            test_set_y_model_dict = {
                model: np.load(os.path.join(test_dirpath, f"{name}_{model}_predictions.npy"))
                for model in models
            }

            mul_win_cases = []  # List of (t, error)

            for t, y in enumerate(test_set_y[:200]):
                valid_js = []
                for j in range(n_pedestrians):
                    if all(
                        not np.isnan(test_set_y_model_dict[model][t, :12, j, :2]).any()
                        for model in models
                    ):
                        valid_js.append(j)
                valid_js = valid_js[:4]
                if not valid_js:
                    continue

                model_errors = {}
                for model in models:
                    pred = test_set_y_model_dict[model][t][:12, valid_js, :2]
                    truth = y[:12, valid_js, :2]
                    model_errors[model] = np.linalg.norm(pred - truth, axis=2).mean()

                mul_error = model_errors['mul']
                if mul_error == min(model_errors.values()):
                    mul_win_cases.append((t, mul_error))

            # Sort all valid mul-win frames by increasing error
            mul_win_cases.sort(key=lambda x: x[1])

            if not mul_win_cases:
                print(f"{name}: No valid frames where 'mul' outperforms others.")
                return

            best_t, best_err = mul_win_cases[0]
            second_best_t, second_best_err = None, None

            # Find second-best that is ≥ 30 frames apart
            for t_candidate, err_candidate in mul_win_cases[1:]:
                if abs(t_candidate - best_t) >= min_frame_gap:
                    second_best_t = t_candidate
                    second_best_err = err_candidate
                    break

            print(f"{name}: Best mul frame t={best_t} with error={best_err:.4f}")
            if second_best_t is not None:
                print(f"{name}: Second-best mul frame t={second_best_t} with error={second_best_err:.4f}")
            else:
                print(f"{name}: No second-best frame found ≥ {min_frame_gap} frames apart from best.")

    return

def main():

    n_pedestrian=[367,415,434,420,148,204]
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
        [18.42, 17.21, -8.69, -6.17],
        [15.4369, 13.8542, -0.1747, -0.2222],
        [15.4369, 13.8542, -0.1747, -0.2222],
        [6.35,   4.31,   -3.25,  -10.31],
        [15.1324, 13.3864, -0.0210,  0.7613],
        [15.5584, 14.9427, -0.3578,  0.7263]
    ]

    os.makedirs(video_dir, exist_ok=True)
    
    num_processes=len(test_dirpaths)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(datasaver, list(zip(test_dirpaths,n_pedestrian,map_sizes,images))) 
    # visualization

    return


if __name__ == '__main__':
    main()