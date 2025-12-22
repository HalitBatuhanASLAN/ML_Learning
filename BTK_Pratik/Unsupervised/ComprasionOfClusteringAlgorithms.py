from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples= n_samples,factor=0.5,noise=0.05)
noisy_moons = datasets.make_moons(n_samples= n_samples, noise= 0.05)
blobs = datasets.make_blobs(n_samples= n_samples)
no_sturucture = np.random.rand(n_samples, 2 ), None # hepsi iki tane değer döndürüyor onun için ikinci değeri None yaptık

clustering_names = ["MiniBatchKMeans", "SpectralClustering", "Ward", 
                    "AgglomerativeClustering", "DBSCAN", "Birch"]

colors = np.array(["b","r","g","c","m","y"])
datasets = [noisy_circles, noisy_moons, blobs, no_sturucture]

plt.figure()
i = 1
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    spectral = cluster.SpectralClustering(n_clusters=2)
    ward = cluster.AgglomerativeClustering(n_clusters=2,linkage= "ward")
    average_linkage = cluster.AgglomerativeClustering(n_clusters= 2, linkage= "average")
    dbscan = cluster.DBSCAN(eps=0.2)
    birch = cluster.Birch(n_clusters=2)

    clustering_alforithms = [two_means, spectral, ward, average_linkage, dbscan, birch]

    for name, algo in zip(clustering_names,clustering_alforithms):
        algo.fit(X) # train

        #prediction
        if hasattr(algo, "labels_"): # bazılarında labels oluri sonucu oraya atar direkt
            y_pred = algo.labels_.astype(int) # küme etiketini, id return eder
        else:
            y_pred = algo.predict(X)

        plt.subplot(len(datasets),len(clustering_alforithms),i)
        if i_dataset == 0:
            plt.title(name)
        plt.scatter(X[:,0],X[:,1],color = colors[y_pred].tolist(), s = 10) # s nokta büyüklüğü
        i += 1

plt.show()