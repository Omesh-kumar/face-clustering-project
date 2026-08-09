import numpy as np

class RCPSO:
    def __init__(self, n_clusters=3, iterations=30, particles=20):
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.particles = particles
        self.centroids_ = None
        self.labels_ = None

    def assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fitness(self, centroids, X):
        labels = self.assign_clusters(X, centroids)
        total = 0.0
        for k in range(self.n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                total += np.linalg.norm(pts - centroids[k], axis=1).sum()
        return total

    def fit(self, X, init_centroids=None):
        """
        init_centroids:
        - None → random initialization
        - ndarray → CBCC-initialized centroids
        """
        n_features = X.shape[1]

        # 🔹 Initialize particles
        if init_centroids is None:
            particles = np.random.randn(
                self.particles, self.n_clusters, n_features
            )
        else:
            particles = np.tile(init_centroids, (self.particles, 1, 1))
            particles += np.random.randn(*particles.shape) * 0.05

        best_particle = particles[0]
        best_score = self.fitness(best_particle, X)

        # 🔹 PSO loop
        for _ in range(self.iterations):
            for p in particles:
                score = self.fitness(p, X)
                if score < best_score:
                    best_score = score
                    best_particle = p.copy()

        self.centroids_ = best_particle
        self.labels_ = self.assign_clusters(X, best_particle)

        return self
