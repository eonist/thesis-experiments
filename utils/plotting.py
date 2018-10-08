import matplotlib.pyplot as plt

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
