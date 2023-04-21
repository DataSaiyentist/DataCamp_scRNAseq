import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier


def _preprocess_X(X_sparse, y=None, rm_gene=None):
    # cast a dense array
    X = X_sparse.toarray()

    # compute call rates and remove genes and cells
    if y is not None:
        call_rate = X.mean(axis=1)
        X = X[call_rate > 0.01, :]
        y = y[call_rate > 0.01]
        call_rate = X.mean(axis=0)
        X = X[:, call_rate > 0.1]

        return X / X.sum(axis=1)[:, np.newaxis], y, np.where(call_rate > 0.1)[0]  # noqa

    X = X[:, rm_gene]

    # normalize each row
    return X / X.sum(axis=1)[:, np.newaxis]


class Classifier(object):
    def __init__(self):
        # use scikit-learn's pipeline
        self.pipe = make_pipeline(
            StandardScaler(),
            VarianceThreshold(threshold=0.1),
            SelectKBest(f_classif, k=1000),
            GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_features="sqrt"
            ),
        )

        self.rm_gene = np.array([])

    def fit(self, X_sparse, y):
        X, y, idx_gene = _preprocess_X(X_sparse, y)
        # save removed genes index
        setattr(self, "rm_gene", idx_gene)
        self.pipe.fit(X, y)

        pass

    def predict_proba(self, X_sparse):
        X = _preprocess_X(X_sparse, rm_gene=self.rm_gene)
        return self.pipe.predict_proba(X)
