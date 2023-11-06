import numpy as np

class LinearRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialize weights and bias as zero
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            # predict using y_pred = wx + b
            y_pred = np.dot(self.weights, X.T) + self.bias

            # compute error
            error = y_pred -y

            # gradient to tune weights and bias
            dw = (1/n_samples ) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            self.weights = self.weights - dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        y_pred = np.dot(self.weights, X.T) + self.bias
        return y_pred
