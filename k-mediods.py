import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11]
])

k = 2
np.random.seed(0)

medoid_indices = np.random.choice(len(X), k, replace=False)
medoids = X[medoid_indices]

for _ in range(10):
    distances = np.linalg.norm(X[:, None] - medoids[None, :], axis=2)
    labels = np.argmin(distances, axis=1)

    new_medoids = []
    for i in range(k):
        cluster_points = X[labels == i]
        cost = np.sum(np.linalg.norm(cluster_points[:, None] - cluster_points[None, :], axis=2), axis=1)
        new_medoids.append(cluster_points[np.argmin(cost)])
    new_medoids = np.array(new_medoids)

    if np.allclose(medoids, new_medoids):
        break
    medoids = new_medoids

print("Medoids:\n", medoids)
print("Labels:", labels)

colors = ['r', 'b']
for i, c in enumerate(X):
    plt.scatter(c[0], c[1], color=colors[labels[i]])
plt.scatter(medoids[:, 0], medoids[:, 1], marker='X', s=200, c='green')
plt.title("K-Medoids Clustering")
plt.show()
