import numpy as np

class OutlierRemovalClustering:
    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def filter(self, X, labels, centroids):
        mask = np.ones(len(X), dtype=bool)

        for k in range(len(centroids)):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                continue

            d = np.linalg.norm(cluster_points - centroids[k], axis=1)
            d_max = d.max()

            if d_max > 1e-9:
                outlyingness = d / d_max
                idx = np.where(labels == k)[0]
                mask[idx[outlyingness > self.threshold]] = False

        return mask
