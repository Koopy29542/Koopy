import os
import numpy as np


def compute_interval_lengths(miscoverage_levels, calibration_set_y, n_scores, score_functions, max_interval_lengths):
    """
    :param miscoverage_level of shape (m,): target miscoverage levels; lie between [0, 1]
    :param calibration_set: a list of pairs (f(x), y)'s where (x, y): input-output data & f: model
    :param score_functions: a function from Y x Y to R^m where m represents the number of different score functions
    Given (f(x), y), this outputs (s_1, ..., s_i, ..., s_m) where s_i = s_i(f(x), y): i-th score function

    :return: length R of the predictive interval for m score functions; of shape (m,)

    Specifically, the prediction set for each score s_i is given as C_{alpha, i}(x|D) = {y: s_i(f(x), y) <= R[i]}

    """
    scores = np.array([score_functions(y_model, y) for y_model, y in calibration_set_y])    # shape: (N, m)

    less_than_zero = miscoverage_levels <= 0.       # alpha >= 0
    more_than_one = miscoverage_levels >= 1.        # alpha <= 1
    valid = np.logical_not(less_than_zero + more_than_one)      # 0 < alpha < 1
    score_indices, = np.where(valid)
    interval_len = np.zeros((n_scores,))

    for idx in score_indices:
        interval_len[idx] = np.quantile(scores[:, idx], q=1. - miscoverage_levels[idx], axis=0)   # along data dimension
    interval_len[less_than_zero] = max_interval_lengths[less_than_zero]
    interval_len[more_than_one] = 0.       # just for clarity...
    return interval_len


PREDICTION_LEN = 20
HISTORY_LEN = 20



def evaluate_prediction_set(test_set_y, score_functions, interval_len):
    """
    :param test_set: a list of pairs (f(x), y)'s where (x, y) test data apart from those used for the calibration
    :param interval_len: result of the conformal prediction; independent of the choice between CP/ACP
    :return: numpy boolean array of shape (N, m); True if the prediction interval contains
    """
    eval_scores = np.array([score_functions(y_model, y) for y_model, y in test_set_y])

    # numpy array of shape (N, m), where N: test data size & m: # of score functions
    correct = eval_scores < interval_len

    # np.all(correct, axis=-1)              # shape: (N,)
    # The above checks if all prediction sets (corresponding to different score functions) contain true y (data-wise)
    return correct


def sample_calibration_set_y(model, size):

    # directory containing validation scenarios
    val_dirpath = './val'
    val_scenarios = os.listdir(val_dirpath)

    calibration_set_y = []

    sampled_scenarios = np.random.choice(val_scenarios, size=size)

    for scenario in sampled_scenarios:
        # TODO: read true/predicted trajectories from the specified file path & choose its fragment
        y_path = os.path.join(val_dirpath, scenario, model)     # ground truth
        model_y_path = os.path.join(val_dirpath, scenario, model)

        time_begin, time_end = 20, 180
        time_idx = np.random.randint(time_begin, time_end)

        # shape (prediction length, # pedestrians, 2)
        y_model, y = None, None
        calibration_set_y.append((y_model, y))

    return calibration_set_y


def  load_test_set_y(model, ood):
    if ood:
        test_dirpath = './test'
    else:
        test_dirpath = './test_ood'

    test_scenarios = os.listdir(test_dirpath)

    test_set_y = []

    for scenario in test_scenarios:
        y_path = os.path.join(test_dirpath, scenario, model)
        model_y_path = os.path.join(test_dirpath, scenario, model)

        # TODO: read true trajectories from the specified file path & choose its fragment

        time_begin, time_end = 20, 180
        time_idx = np.random.randint(time_begin, time_end)

        y_model, y = None, None
        test_set_y.append((y_model, y))

    return test_set_y

