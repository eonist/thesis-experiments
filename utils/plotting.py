import matplotlib.pyplot as plt
import numpy as np

from config import Path
from utils.prints import Print
from utils.utils import create_path_if_not_existing


def plot_training_history(training_history, loss_function="", show=True, save=False):
    fig = plt.figure(figsize=(16, 6))
    loss_ax = fig.add_subplot(121)

    Print.data(list(training_history.history.keys()))

    loss_ax.plot(training_history.history['loss'], 'r', linewidth=3.0)
    loss_ax.plot(training_history.history['val_loss'], 'b', linewidth=3.0)
    loss_ax.legend(['Training loss', 'Validation Loss'], fontsize=18)
    loss_ax.set_xlabel('Epochs ', fontsize=16)
    loss_ax.set_ylabel('Loss', fontsize=16)
    loss_ax.set_title('Loss Curves: {}'.format(loss_function), fontsize=16)

    acc_ax = fig.add_subplot(122)
    acc_ax.plot(training_history.history['acc'], 'r', linewidth=3.0)
    acc_ax.plot(training_history.history['val_acc'], 'b', linewidth=3.0)
    acc_ax.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    acc_ax.set_xlabel('Epochs ', fontsize=16)
    acc_ax.set_ylabel('Accuracy', fontsize=16)
    acc_ax.set_title('Accuracy Curves', fontsize=16)

    plt.tight_layout()

    if save:
        create_path_if_not_existing(Path.plots)
        fp = "/".join([Path.plots, save])
        plt.savefig(fp, format="png", dpi=400)

    if show:
        plt.show()


def plot_boolean_arrays(arrays):
    l = len(arrays[0])
    n = len(arrays)
    row_height = int(np.max([l / 10, 1]))

    m = np.zeros([n * row_height, l])

    for i, array in enumerate(arrays):
        m[i * row_height:(i + 1) * row_height, :] = np.tile(array, (row_height, 1))

    plt.matshow(m)
    plt.show()


def plot_matrix(m, upscale_lowest_dim=True):
    if upscale_lowest_dim:
        min_ratio = 5
        h, w = np.shape(m)

        Print.pandas(m)

        row_height = 1 if h >= w / min_ratio else int(w / (min_ratio * h))
        col_width = 1 if w >= h / min_ratio else int(h / (min_ratio * w))

        Print.data(row_height)
        Print.data(col_width)

        res = np.zeros([h * row_height, w * col_width])

        Print.data(np.shape(res))

        if row_height > col_width:
            for i, row in enumerate(m):
                res[i * row_height: (i + 1) * row_height, :] = np.tile(row, (row_height, 1))
        elif col_width > row_height:
            for i, col in enumerate(m.T):
                res[:, i * col_width: (i + 1) * col_width] = np.tile(col, (col_width, 1)).T
        else:
            res = m
    else:
        res = m

    cmap = plt.cm.Blues
    plt.imshow(res, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    arrays = [[True, True, False], [True, False, False], [False, False, False]]

    plot_boolean_arrays(arrays)
