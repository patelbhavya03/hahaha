import numpy as np
links = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0]
], dtype=float)

num_pages = links.shape[0]
hubs = np.ones(num_pages)
authorities = np.ones(num_pages)

for _ in range(10):  # 10 iterations
    authorities = np.dot(links.T, hubs)
    hubs = np.dot(links, authorities)
    
    # Normalize
    authorities /= np.linalg.norm(authorities)
    hubs /= np.linalg.norm(hubs)

print("Authority Scores:", np.round(authorities, 3))
print("Hub Scores:", np.round(hubs, 3))
