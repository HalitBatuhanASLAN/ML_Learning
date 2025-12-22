from sklearn.datasets import make_blobs # bu bize veri seti oluşturmada yardımcı olur
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

# alttaki 2 değer return ediyor, sağdaki kullanamyacağıız için öyle bir şey kullandık
# center kaç tane küme merkezi olacağı(kaç küme olacağı)
# cluster_std nokraların standart sapması, yani merkeden ne kadar uzakta dağıldıkları büyükse gürültü fazla
X, _ = make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42)

plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.title("Örnek veri")

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

labels = kmeans.labels_

plt.figure()
plt.scatter(X[:,0], X[:,1], c = labels, cmap="viridis")

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c="red",marker="X")
plt.title("K-Means")
plt.show()




















