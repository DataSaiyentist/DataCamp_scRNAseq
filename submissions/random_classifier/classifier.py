import numpy as np


class Classifier(object):
    def __init__(self):
        pass

    def fit(self, X_sparse, y):
        classes = [
            "T_cells_CD8+",
            "T_cells_CD4+",
            "Cancer_cells",
            "NK_cells",
        ]
        self.n_classes = len(classes)
        pass

    def predict_proba(self, X_sparse):
        # random soft proba
        n, _ = X_sparse.shape

        proba = np.random.rand(n, self.n_classes)
        proba /= proba.sum(axis=1)[:, np.newaxis]

        # random hard proba
        # idx = np.random.choice(range(self.n_classes), len(videos))
        # proba = np.zeros((len(videos), self.n_classes))
        # proba[range(len(videos)), idx] = 1
        return proba
