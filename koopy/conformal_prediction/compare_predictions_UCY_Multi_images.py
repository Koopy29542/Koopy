import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from adaptive_cp import AdaptiveConformalPredictionModule
from score_functions import stepwise_displacement_error
from visualization_utils import draw_map3
import cv2
import multiprocessing
video_dir = './images/compare/test'
def datasaver(test_dirpath,n_pedestrians,map_size,bg_img_path):

    #models = ['linear', 'trajectron', 'eigen', 'koopman']
    models = ['linear', 'gp', 'eigen', 'koopman_clu_geo','koopman_clu_vel','koopman_enc', 'trajectron']
    colors = [ 'red','violet','green', 'brown','blue','grey','pink']
    #labels = ['Linear', 'Trajectron++', 'EigenTrajectory + STGCNN', 'Koopman']
    #labels = ['Linear', 'GP','EigenTrajectory + STGCNN','Koopman+ 8 history','Trajectron++','Trajectron++ 20']



    for file in os.listdir(test_dirpath):
        if file.endswith('.txt'):
            name, _ = os.path.splitext(file)
            scenario_dir = os.path.join(video_dir, name)
            os.makedirs(scenario_dir, exist_ok=True)

            test_set_y = np.load(os.path.join(test_dirpath, name + '_targets.npy'))
            # test_set_y = test_set_y

            test_set_y_model_dict = {}

            for model in models:
                test_set_y_model_dict[model] = np.load(os.path.join(test_dirpath, name + '_{}_predictions.npy'.format(model)))
            for t, y in enumerate(test_set_y[:200]):

                plt.clf(), plt.cla()
                fig, ax,_,_ = draw_map3(*map_size,bg_img_path=bg_img_path)
                valid_js = []
                for j in range(n_pedestrians):
                    # check ground truth

                    # check *all* models
                    ok = True
                    for model in models:
                        y_model = test_set_y_model_dict[model][t]
                        if np.isnan(y_model[:12, j, :2]).any():
                            ok = False
                            break

                    if ok:
                        valid_js.append(j)

                # 2) optionally cap at 4 pedestrians
                valid_js = valid_js[:4]

                # 3) plot ground truth only for those valid_js
                for rank, j in enumerate(valid_js):
                    ax.scatter(y[0, j, 0], y[0, j, 1], color='k', s=50, zorder=300)
                    if j == valid_js[0]:
                        ax.plot(y[:, j, 0], y[:, j, 1], color='k', zorder=300, label='True')
                    else:
                        ax.plot(y[:, j, 0], y[:, j, 1], color='k', zorder=300)

                # 4) plot each model’s predictions only for those same valid_js
                for i, model in enumerate(models):
                    color = colors[i]
                    y_model = test_set_y_model_dict[model][t]
                    for j in valid_js:
                        ax.plot(y_model[:, j, 0],
                                y_model[:, j, 1],
                                color=color,
                                zorder=300)
                fig.canvas.draw()
                img = np.array(fig.canvas.renderer.buffer_rgba())  # Get image array from figure
                raw_bbox   = ax.get_window_extent()              # TransformedBbox
                x0, y0, x1, y1 = raw_bbox.extents                # four floats
                x0_i, y0_i, x1_i, y1_i = map(int, (x0, y0, x1, y1))

                # 3) because image array origin is top‐left, invert the y‐coords:
                h = img.shape[0]
                crop = img[ h - y1_i : h - y0_i,  x0_i : x1_i ] 
                bgr = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
                out_path = os.path.join(scenario_dir, f"{t}_{name}_straight.jpg")
                cv2.imwrite(out_path, cv2.resize(bgr, (800, 800)))

                plt.close(fig)
    print(name)
    return
def main():

    n_pedestrian=[367,415,434,420,148,204]
    images = [
    "ethucyimages/eth.png",
    "ethucyimages/students_003.jpg",
    "ethucyimages/students_003.jpg",
    "ethucyimages/hotel.png",
    "ethucyimages/crowds_zara01.jpg",
    "ethucyimages/crowds_zara02.jpg"
]
    test_dirpaths =[
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

    os.makedirs(video_dir, exist_ok=True)
    
    num_processes=len(test_dirpaths)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(datasaver, list(zip(test_dirpaths,n_pedestrian,map_sizes,images))) 
    # visualization

    return


if __name__ == '__main__':
    main()