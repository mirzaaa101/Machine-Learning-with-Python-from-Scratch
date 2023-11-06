import numpy as np

class GaussianNaiveBayesClassifier:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._varience = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for _class in self._classes:
            X_class = X[_class==y]
            self._mean[_class,:] = X_class.mean(axis=0)
            self._varience[_class,:] = X_class.var(axis=0)
            self._priors[_class] = X_class.shape[0]/float(n_samples)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions


    def _predict(self,x):
        posteriors = []

        for index,_class in enumerate(self._classes):
            prior = np.log(self._priors[index])
            posterior  = np.sum(np.log(self._probability_density(x,index)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _probability_density(self, x,index):
        mean = self._mean[index]
        varience = self._varience[index]
        numerator = np.exp(-((x - mean) ** 2) / (2 * varience))
        denominator = np.sqrt(2 * np.pi * varience)
        return numerator / denominator