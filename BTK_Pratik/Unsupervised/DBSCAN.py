from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

# factor -> içteki halka dıştakinin 0.5 katı büyüklüğündedi
X, _ = make_circles(n_samples= 1000, factor= 0.5, noise= 0.05, random_state= 42)

plt.figure()
plt.scatter(X[:,0],X[:,1])

# eps -> iki noktanın komşu kabul edilebilmesi için max mesafe
# küme olması için gereken min örnek sayısı
dbscan = DBSCAN(eps= 0.1, min_samples= 5)
cluster_labels = dbscan.fit_predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c = cluster_labels, cmap= "viridis")
plt.title("DBSCAN sonuçları")

plt.show()