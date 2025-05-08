import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

from cp.adaptive_cp import AdaptiveConformalPredictionModule
from utils.score_functions import stepwise_displacement_error
from utils.visualization_utils import draw_map4
import cv2

def main():
    t_start = 2000  # 시작 프레임
    t_end = 3000    # 종료 프레임
    
    n_pedestrians = 205 
    models = ['linear','koopy','trajectron']
    colors = ['brown','red','blue']
    labels = ['Linear','Koopy','Trajectron++']

    test_dirpath = '../lobby3/test'
    video_dir = '../videos/lobby3/test'
    os.makedirs(video_dir, exist_ok=True)

    obstacles_meter_lobby3 = {
        'left_top': [(-2.28, 1.19), (-10, 1.19), (-10, 5), (-2.28, 5)],
        'left_bottom': [(-2.59, -8.95), (-10, -8.95), (-10, -15), (-2.59, -15)],
        'right_top': [(-0.23, 1.17), (10, 1.17), (10, 5), (-0.23, 5)],
        'right_bottom': [(8.13, -8.95), (10, -8.95), (10, -15), (8.13, -15)],
        'bottom': [(-2.59, -8.95), (-2.59, -15), (8.13, -15), (8.13, -8.95)],
        'left': [(-10, -0.6), (-2.59, -0.6), (-2.59, -8.95), (-10, -8.95)],
        'right': [(8.13, -8.95), (8.13, -0.6), (10, -0.6), (10, -8.95)],
        'elevator': [(-2.30, 3.92), (-0.26, 3.92), (-0.26, 5), (-2.30, 5)],
        'entrance': [(0.63, -8.86), (0.63, -6.54), (5.03, -6.54), (5.03, -8.86)],
        'middle_obstacle': [(2.27, -2.00), (3.26, -2.00), (3.29, -4.29), (2.21, -4.34)],
        'pillar_left': [(-0.26, -0.55), (-0.54, -0.73), (-0.67, -1.11), (-0.52, -1.44), (-0.23, -1.59),
                        (0.06, -1.37), (0.20, -1.07), (0.03, -0.75)],
        'pillar_right': [(5.80, -0.68), (5.56, -0.77), (5.38, -1.09), (5.48, -1.41),
                         (5.79, -1.55), (6.06, -1.39), (6.18, -1.16), (6.04, -0.86)]
    }

    for file in os.listdir(test_dirpath):
        if file.endswith('.txt'):
            name, _ = os.path.splitext(file)
            scenario_dir = os.path.join(video_dir, name)
            os.makedirs(scenario_dir, exist_ok=True)

            test_set_y = np.load(os.path.join(test_dirpath, name + '_targets.npy'))
            test_set_y_model_dict = {}

            for model in models:
                test_set_y_model_dict[model] = np.load(os.path.join(test_dirpath, name + '_{}_predictions.npy'.format(model)))
            
            image_list = []
            k = range(n_pedestrians)
            q = range(1, n_pedestrians)
            
            for t in range(t_start, min(t_end, len(test_set_y))):
                y = test_set_y[t]
                plt.clf(), plt.cla()
                fig, ax = draw_map4(xlim=8.6, ylim=1.5, belx=-2.8, bely=-9.3, bg_img_path="./lobby3.png")
                ax.set_xlim([-10, 10])
                ax.set_ylim([-15, 5])
                
                #for obs in obstacles_meter_lobby3.values():
                #    polygon = Polygon(obs, closed=True, edgecolor='black', facecolor='gray', alpha=0.5, zorder=200)
                #    ax.add_patch(polygon)
                
                ax.scatter(y[0, k, 0], y[0, k, 1], color='k', s=50, zorder=300)
                ax.plot(y[:, 0, 0], y[:, 0, 1], color='k', zorder=300, label='True')
                ax.plot(y[:, q, 0], y[:, q, 1], color='k', zorder=300)
                
                for i, model in enumerate(models):
                    color = colors[i]
                    label = labels[i]
                    y_model = test_set_y_model_dict[model][t]
                    ax.plot(y_model[:, 0, 0], y_model[:, 0, 1], color=color, label=label, linestyle='-', zorder=300)
                    ax.plot(y_model[:, q, 0], y_model[:, q, 1], color=color, linestyle='-', zorder=300)
                #ax.legend(loc='upper right')

                ax.set_title('test scenario {}'.format(name))
                fig.canvas.draw()
                image = np.array(fig.canvas.renderer.buffer_rgba())
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                image = cv2.resize(image, (800, 1000))
                image_list.append(image)
                plt.close(fig)
            
            video_path = os.path.join(video_dir, 'test_scenario_{}.mp4'.format(name))
            if image_list:
                height, width, _ = image_list[0].shape
                fps = 10
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                for img in image_list:
                    video_writer.write(img)
                video_writer.release()
                cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
