import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x, y):
    distance = np.sqrt(np.sum((x-y)**2))
    return distance

class KMeansClustering:
    def __init__(self, k=5, max_iters=100, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []


    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        k_random_samples_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[index] for index in k_random_samples_indices]

        for _ in range(self.max_iters):
            # assign samples to closest centroids
            self.clusters = self._create_cluster(self.centroids)
            if self.plot_steps ==True:
                self._plot()

            # compute new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check stopping criteria
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps ==True:
                self._plot()


        return self._get_cluster_labels(self.clusters)


    def _create_cluster(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for index, sample in enumerate(self.X):
            centroid_index = self._closest_centroid(sample, centroids)
            clusters[centroid_index].append(index)

        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index


    def _get_centroids(self, clusters):
        centroid = np.zeros((self.k, self.n_features))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroid[cluster_index] = cluster_mean

        return centroid

    def _is_converged(self, centroids_old,centroid_new):
        distance = [euclidean_distance(centroids_old[i], centroid_new[i]) for i in range(self.k)]
        return sum(distance)==0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index

        return labels

    def _plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()