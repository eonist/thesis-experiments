import itertools

import matplotlib.pyplot as plt
import numpy as np

from config import Path


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          raw=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    if not raw:
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    else:
        plt.axis('off')

    plt.tight_layout()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


c_matrix = np.array([
    [331, 8, 43, 26, 36],
    [20, 276, 127, 5, 7],
    [47, 156, 150, 17, 24],
    [33, 8, 31, 229, 82],
    [25, 4, 33, 190, 201]
])

class_names = ["none", "left arm", "right arm", "left foot", "right foot"]
# class_names = ["none", "event"]
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(c_matrix, classes=class_names, raw=False)

# plt.show()
plt.savefig(Path.plots + "/csp4_cmat.png", bbox_inches='tight')
