from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
    
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, cluster_std= 0.6, random_state= 42)

plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.title("Örnek veri")

linkage_methods = ["ward","single","average","complete"]
# ward -> küme içi varyansları küçültmeye çalışıyoruz
# sizble -> iki kümenin birbirine en yakın 2 noktası arasındaki mesafeyi ölçüt olarak alıyoruz
# verage -> iki küme arasındaki tüm noktaların arasındaki mesafelerin ortalaması
# complete -> iki küme arasındaki en uzun mesafeden


plt.figure()
for i, linkage_method in enumerate(linkage_methods,1):
    model = AgglomerativeClustering(n_clusters=4,linkage=linkage_method)
    cluster_labels = model.fit_predict(X) # kümelemeyi yaptık
    
    plt.subplot(2,4,i) # 2 satır 4 sütundan, i. indexdeki grafikteyiz
    plt.title(f"{linkage_method.capitalize()} Linkage Dendogram")
    # kümelemeyi yaparken linkage metodu ile bir mesafe ölçme tipi belilenir
    # burada sırayla ward vs linkage tipi olur
    # hiyerarşik kümeleme yöntemini gösteren bir ağaç gibi olan diyagramdır
    dendrogram(linkage(X,method = linkage_method), no_labels = True)
    plt.xlabel("Veri noktaları")
    plt.ylabel("uzaklik")

    plt.subplot(2,4,i + 4) # en altta olacağı için ve satırda 4 rablo olduğu için +4 ile atıyoruz
    plt.scatter(X[:,0],X[:,1],c = cluster_labels,cmap = "viridis")
    plt.title(f"{linkage_method.capitalize()} Linkage Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")


plt.show()