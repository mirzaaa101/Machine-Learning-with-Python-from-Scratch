import numpy as np
from k_means import KMeansClustering

np.random.seed(42)
from sklearn.datasets import make_blobs

X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeansClustering(k=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()