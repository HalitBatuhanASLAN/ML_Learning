from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2) # 2 adet temel bileşen(PC)
# elimizdeki n tane özelliği bizim için en öenmli 2 bilgiyi taşıyan yeni özellik üret

X_pca = pca.fit_transform(X)

plt.figure()
for i in range(len(iris.target_names)):
    # çiçeğin türüne göre noktaların çizimi, 0 pca1(x ekseni); 1 pca2(y ekseni)
    plt.scatter(X_pca[y== i, 0], X_pca[y == i,1], label = iris.target_names[i])
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA of iris dataset")
plt.legend() # label gözükmesi için

#%%

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=3) # 3 adet temel bileşen(PC)
X_pca = pca.fit_transform(X)

fig = plt.figure(figsize=(8,6))

# 111 -> 1; kaç satır; 1 kaç sütun; 1 kaçıncı grafik
# elevation -> yükseklik açısı, -150 ile alt taraftan dik bir açı ile bakıyoruz
# azimuth -> yatay dönüş açısı, dikey eksen etrafında grafiğin ne kadar döndüreüleceğini test eder
ax = fig.add_subplot(111, projection = "3d", elev=-150, azim = 110)

# s -> her bir noktanın boyutu
ax.set_title("First 3 pca componenets of iris dataset")
ax.set_xlabel("1st Eigenvector")
ax.set_ylabel("2nd Eigenvector")
ax.set_zlabel("3rd Eigenvector")
