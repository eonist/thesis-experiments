from mne.decoding import CSP
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from transformers.mne_filter import MneFilter


class OptionList:
    def __init__(self, val):
        self.val = val


class PipelineElement:
    def __init__(self, initializer, param_options):
        self.initializer = initializer
        self.param_options = param_options


svm = PipelineElement(svm.SVC, {"svc__kernel": ["linear", "rbf", "sigmoid"]})
lda = PipelineElement(LinearDiscriminantAnalysis,
                      {"lineardiscriminantanalysis__solver": ["svd", "lsqr", "eigen"]})

csp = PipelineElement(CSP, {"csp__log": [True, False]})

filter = PipelineElement(MneFilter, {"mnefilter__l_freq": [7, 10, 15]})

pipelines = [
    [filter, csp, svm],
    [filter, csp, lda]
]
