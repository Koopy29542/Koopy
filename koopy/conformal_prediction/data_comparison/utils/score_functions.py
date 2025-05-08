import numpy as np


def stepwise_displacement_error(y1, y2):
    """
    :param y1, y2: numpy array of shape (prediction length, # pedestrians, 2)

    :return: numpy array of shape (prediction length,) each entry representing the maximum displacement error at time t
    """
    return np.max(np.sum((y1 - y2) ** 2, axis=-1) ** .5, axis=-1)