import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, ShuffleSplit

from models.session import Session
from transformers.csp import CSP
from transformers.filter import Filter


def process_dataset(X, y):
    filter = Filter(l_freq=7, h_freq=30)

    X = filter.transform(X)

    csp = CSP()

    print(np.shape(X))
    X = csp.fit_transform(X, y)

    return np.mean(X, axis=2)


def classify(X, y):
    lda = LinearDiscriminantAnalysis()

    cv = ShuffleSplit(n_splits=3, test_size=0.2)

    scores = cross_val_score(lda, X, y, cv=cv, n_jobs=1)
    print(scores)


def main():
    ds = Session.full_dataset()
    print(ds.distribution())
    # ds = Session.combined_dataset([40])
    y = ds.binary_y("none_rest")
    X = process_dataset(ds.X, y)
    print(np.shape(X))

    classify(X, y)


if __name__ == '__main__':
    main()
