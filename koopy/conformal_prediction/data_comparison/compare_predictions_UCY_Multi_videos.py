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

video_dir = '../videos/compare/test'

def datasaver(test_dirpath, n_pedestrians, map_size, bg_img_path):
    models = ['linear', 'gp', 'eigen','mul' , 'trajectron']
    colors = [ 'brown','violet','green', 'red','blue']

    for file in os.listdir(test_dirpath):
        if not file.endswith('.txt'):
            continue

        name, _ = os.path.splitext(file)
        scenario_dir = os.path.join(video_dir, name)
        os.makedirs(scenario_dir, exist_ok=True)

        # --- setup video writer ---
        video_path = os.path.join(scenario_dir, f"{name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        frame_size = (800, 800)
        writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        # load ground truth and all model predictions
        test_set_y = np.load(os.path.join(test_dirpath, name + '_targets.npy'))
        test_set_y_model_dict = {
            model: np.load(os.path.join(test_dirpath, f"{name}_{model}_predictions.npy"))
            for model in models
        }

        for t, y in enumerate(test_set_y[:200]):
            # redraw figure
            plt.clf(); plt.cla()
            fig, ax, _, _ = draw_map3(*map_size, bg_img_path=bg_img_path)

            # find valid pedestrian indices
            valid_js = []
            for j in range(n_pedestrians):
                if all(not np.isnan(test_set_y_model_dict[m][t][:12, j, :2]).any() for m in models):
                    valid_js.append(j)
            valid_js = valid_js[:4]

            # plot ground truth
            for rank, j in enumerate(valid_js):
                ax.scatter(y[0, j, 0], y[0, j, 1], color='k', s=50, zorder=300)
                ax.plot(y[:, j, 0], y[:, j, 1], color='k', zorder=300,
                        label='True' if rank == 0 else None)

            # plot model predictions
            for i, model in enumerate(models):
                color = colors[i]
                y_model = test_set_y_model_dict[model][t]
                for j in valid_js:
                    ax.plot(y_model[:, j, 0],
                            y_model[:, j, 1],
                            color=color,
                            zorder=300)

            # render to image
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            raw_bbox = ax.get_window_extent()
            x0, y0, x1, y1 = map(int, raw_bbox.extents)
            h = img.shape[0]
            crop = img[h - y1 : h - y0, x0 : x1]
            bgr = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
            frame = cv2.resize(bgr, frame_size)

            # write to video
            writer.write(frame)

            plt.close(fig)

        writer.release()
        print(f"Saved video for scenario {name} → {video_path}")

    return

def main():
    n_pedestrian = [367,415,434,420,148,204]
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
    num_processes = len(test_dirpaths)
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(datasaver, zip(test_dirpaths, n_pedestrian, map_sizes, images))

if __name__ == '__main__':
    main()
