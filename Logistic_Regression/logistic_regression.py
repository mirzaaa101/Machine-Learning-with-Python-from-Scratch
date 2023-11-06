import numpy as np

def sigmoid(x):
    value = 1/(1+np.exp(-x))
    return value


class LogisticRegression():
    def __init__(self, lr=0.1, max_iters=1000):
        self.lr = lr
        self.max_iters = max_iters
        self.bias = None
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias = 0
        self.weights = np.zeros(n_features)

        for _ in range(self.max_iters):
            linear_pred = self.bias + np.dot(X, self.weights)
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples)*np.dot(X.T, (predictions-y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        linear_pred = self.bias + np.dot(X, self.weights)
        y_pred = sigmoid(linear_pred)
        class_pred = []

        for pred in y_pred:
            if pred <= 0.5:
                class_pred.append(0)
            else:
                class_pred.append(1)

        return class_pred
