import numpy as np
from collections import Counter

def euclidean_dist(x1, x2):
    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_new):
        predictions = [self._predict(x) for x in X_new]
        return predictions

    def _predict(self, x):
        # calculate distance
        dist = [euclidean_dist(x, X_train) for X_train in self.X_train]

        # take k-nearest instances
        k_indices = np.argsort(dist)[:self.k]
        k_nearest_class = [self.y_train[i] for i in k_indices]

        # return mejority voting
        knn = Counter(k_nearest_class).most_common()
        return knn[0][0]