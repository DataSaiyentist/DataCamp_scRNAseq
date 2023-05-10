import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# @Malik-Omar-Tristan
class Classifier(object):
    def __init__(
        self,
        gene_call_rate_threshold=0.02,
        cell_call_rate_threshold=0.01,
        n_genes=500,
    ):
        scaler = StandardScaler(with_mean=True, with_std=True)
        # pca = PCA(n_components=100)
        svm_model = SVC(
            C=5.2, kernel="rbf", decision_function_shape="ovo", probability=True  # noqa
        )

        self.scaler = scaler
        # self.pca = pca
        self.svm_model = svm_model
        self.gene_call_rate_threshold = gene_call_rate_threshold
        self.cell_call_rate_threshold = cell_call_rate_threshold
        self.n_genes = n_genes

    def preprocess_fit(self, X, y):
        # remove genes with a low call rate
        calls = X != 0
        self.high_call_rate_genes = (
            calls.sum(axis=0) / X.shape[0]
        ) > self.gene_call_rate_threshold
        X = X[:, self.high_call_rate_genes]

        # remove cells with a low call rate
        calls = X != 0
        self.high_call_rate_cells = (
            calls.sum(axis=1) / X.shape[1]
        ) > self.cell_call_rate_threshold
        X = X[self.high_call_rate_cells, :]
        y = y[self.high_call_rate_cells]

        # log-normalization
        X += 1  # we add a pseudo-count to avoid 0 values for log-transformation  # noqa
        library_size = X.sum(axis=1)
        self.library_size_mean = library_size.mean()
        size_factor = library_size / self.library_size_mean
        X = X / size_factor[:, None]
        X = np.log2(X)

        # selection of the Highly-Variable Genes
        variance = np.var(X, axis=0)
        self.highly_variable_genes = np.argsort(-variance)[: self.n_genes]
        X = X[:, self.highly_variable_genes]

        return X, y

    def preprocess_predict(self, X):
        # remove genes with a low call rate
        X = X[:, self.high_call_rate_genes]

        # log-normalization
        X += 1  # we add a pseudo-count to avoid 0 values for log-transformation  # noqa
        library_size = X.sum(axis=1)
        size_factor = library_size / self.library_size_mean
        X = X / size_factor[:, None]
        X = np.log2(X)

        # selection of the Highly-Variable Genes
        X = X[:, self.highly_variable_genes]

        return X

    def fit(self, X_sparse, y):
        X = X_sparse.toarray()
        X, y = self.preprocess_fit(X, y)
        self.scaler.fit(X)
        # self.pca.fit(X)  # fit the PCA before the scaling seems to be better
        X = self.scaler.transform(X)
        # self.pca.fit(X)
        # X = self.pca.transform(X)
        self.svm_model.fit(X, y)
        pass

    def predict_proba(self, X_sparse):
        X = X_sparse.toarray()
        X = self.preprocess_predict(X)
        X = self.scaler.transform(X)
        # X = self.pca.transform(X)
        return self.svm_model.predict_proba(X)
