import numpy as np

class CBCC:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def _gravitational_force(self, C):
        force = 0.0
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                d = np.linalg.norm(C[i] - C[j])
                if d > 1e-9:
                    force += 1 / (d ** 2)
        return force

    def initialize(self, X):
        norms = np.linalg.norm(X, axis=1)
        indices = np.argsort(norms)[-self.n_clusters:]
        best = X[indices].copy()

        for i in range(len(X)):
            for k in range(self.n_clusters):
                temp = best.copy()
                temp[k] = X[i]
                if self._gravitational_force(temp) < self._gravitational_force(best):
                    best = temp.copy()

        return best
