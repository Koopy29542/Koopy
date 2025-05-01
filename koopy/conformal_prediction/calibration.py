import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Rectangle, Circle



def prediction_score_function(prediction, output):
    return np.max([np.sum((prediction[p_id] - output[p_id]) ** 2, axis=-1) ** .5 for p_id in prediction.keys()], axis=0)


def evaluate_conformity_score(model, score_function, input, output):
    return score_function(model(input), output)


def calibrate(model, score_function, inputs, outputs):
    scores = []
    for i, o in zip(inputs, outputs):
        s = evaluate_conformity_score(model, score_function, i, o)
        scores.append(s)
    return scores


def get_predictive_interval(scores, miscoverage_rate):

    q = np.quantile(scores, q=1-miscoverage_rate, axis=0)
    return q


def validatate_CP(model, score_function, input, interval_size, output):

    score = evaluate_conformity_score(model, score_function, input, output)
    return score < interval_size


def main():

    prediction_len = 20
    history_len = 23

    # model
    prediction_model = LinearPredictor(prediction_len=prediction_len,
                                       history_len=history_len-3,
                                       smoothing_factor=0.7,
                                       dt=0.1
                                       )
    # load dataset
    idx_begin = 40
    idx_end = 160

    # build a calibration dataset
    calibration_set_size = 50

    dataset = os.listdir('dataset')

    dataset_names = np.random.choice(dataset, replace=True, size=calibration_set_size)
    indices = np.random.randint(low=idx_begin, high=idx_end, size=calibration_set_size)

    inputs = []
    outputs = []

    for i, dirname in enumerate(dataset_names):
        dirpath = os.path.join('dataset', dirname)
        pose = np.load(os.path.join(dirpath, 'pose.npy'))
        # velocity = np.load(os.path.join(dirpath, 'velocity.npy'))

        time_steps, n_pedestrians, _ = pose.shape
        idx = indices[i]
        input = {p_id: pose[idx-history_len+1:idx+1, p_id, :2] for p_id in range(n_pedestrians)}
        output = {p_id: pose[idx+1:idx+prediction_len+1, p_id, :2] for p_id in range(n_pedestrians)}

        inputs.append(input)
        outputs.append(output)

    scores = calibrate(prediction_model, prediction_score_function, inputs, outputs)
    interval = get_predictive_interval(scores, miscoverage_rate=0.1)

    print('predictive interval:', interval)

    # evaluation
    # eval_data = np.random.choice(dataset)
    eval_data = '47'
    eval_dirpath = os.path.join('dataset', eval_data)
    eval_pose = np.load(os.path.join(eval_dirpath, 'pose.npy'))

    validation_res = []
    validation_per_time = []
    true_seq = []
    pred_seq = []


    for t in range(idx_begin, idx_end):
        true_seq.append(eval_pose[t+10, :, :2])
        plt.clf(), plt.cla()
        fig, ax = test_visualization()
        eval_input = {p_id: eval_pose[t-history_len+1:t+1, p_id, :2] for p_id in range(n_pedestrians)}
        # print(eval_input)
        pred = prediction_model(eval_input)

        for p_id in range(n_pedestrians):
            ax.plot(eval_pose[:t, p_id, 0], eval_pose[:t, p_id, 1], color='black', zorder=500)
            pred_pos = pred[p_id]
            # print(pred_pos)
            ax.plot(pred_pos[:, 0], pred_pos[:, 1], color='black', linestyle='dashed', zorder=300)

            for tau in range(prediction_len):
                center = pred_pos[tau, :2]
                circ = Circle(center, radius=interval[tau], color='tab:gray', alpha=0.2, zorder=200)
                ax.add_patch(circ)

        eval_output = {p_id: eval_pose[t+1:t+prediction_len+1, p_id, :2] for p_id in range(n_pedestrians)}
        is_accurate = validatate_CP(prediction_model, prediction_score_function, eval_input, interval, eval_output)

        pred_to_save = np.array([pred[p_id][9, :] for p_id in range(n_pedestrians)])
        pred_seq.append(pred_to_save)

        validation_res.append(np.all(is_accurate))
        validation_per_time.append(is_accurate)
        print('time {}: in predictive region={}'.format(t, is_accurate))
        total_accuracy = np.sum(validation_res) / (t+1 - idx_begin) * 100
        accuracy_per_time = np.sum(validation_per_time, axis=0) / (t+1 - idx_begin) * 100

        for i, tau in enumerate([0, 4, 9, 14, 19]):
            ax.text(14.1, 4.0 - .8 * i, r'$t+{}$: {:.2f}%'.format(tau+1, accuracy_per_time[tau]))

        ax.set_title('(t={}) accuracy: {:.2f}%'.format(t, total_accuracy))
        fig.tight_layout()
        fig.savefig('CP{:03d}.png'.format(t))
        plt.close()

    pred_seq = np.array(pred_seq)
    true_seq = np.array(true_seq)
    plt.clf(), plt.cla()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    times = np.arange(idx_begin, idx_end)
    for dim, varname in enumerate([r'$x(t+10)$ vs $x(t+10|t)$', r'$y(t+10)$ vs $y(t+10|t)$']):

        ax[dim].set_xlim(idx_begin, idx_end-1)
        ax[dim].grid(True)
        colors = ['tab:red', 'tab:blue', 'tab:brown', 'tab:purple']
        for p_id, color in zip(range(n_pedestrians), colors):

            ax[dim].plot(times, true_seq[:, p_id, dim], label='pedestrian {}'.format(p_id), color=color)
            ax[dim].plot(times, pred_seq[:, p_id, dim], linestyle='dashed', color=color)
        ax[dim].set_xlabel(r'$t$')
        ax[dim].legend()
        ax[dim].set_title(varname)
    # fig.tight_layout()
    fig.savefig('error.png')

    return

if __name__ == '__main__':
    main()