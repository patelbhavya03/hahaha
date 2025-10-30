from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11]
])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centers = kmeans.cluster_centers_
labels = kmeans.labels_

print("Cluster Centers:\n", centers)
print("Labels:", labels)

colors = ['r', 'b']
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], color=colors[labels[i]])
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='green')
plt.title("K-Means Clustering")
plt.show()
