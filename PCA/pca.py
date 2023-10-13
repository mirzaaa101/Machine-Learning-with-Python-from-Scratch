import numpy as np

class PrincipalComponentsAnalysis:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self, X):
        # compute central mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute covarience
        cov = np.cov(X.T)

        # compute eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        # sort eigenvectors based on eigenvalues
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]


        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)