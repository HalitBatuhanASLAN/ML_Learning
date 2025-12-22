from sklearn.datasets import make_classification, make_moons, make_circles
# yukarıdakiler ile veriseti oluşturabiliriz istediğimiz özelliklerde
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# make classification ile çizgi şeklinde olur
# n_features ile 2 özellikli(x,y ekseni) şeklinde veri oluştur
# infermative bilgi içeren future sayısı
# redundant ise bilgi içermeyen
# cluster1 diyerek birbiri içine girmeyecek şekilde ayarlam yapıyoruz, her sınıf tek öbek halinde
X, y = make_classification(n_features= 2, n_redundant=0,n_informative=2,n_clusters_per_class=1,random_state=42)
#alt taraf gürültü eklemesi yapmak için
X += 1.2 * np.random.uniform(size = X.shape)

Xy = (X,y)
# c = y ile farklı renklerde görebiliriz
plt.scatter(X[:,0],X[:,1], c= y)
plt.show()


# make moon ile bir nevi hilal şeklinde olur veriler
# noise arttıkça hilla şeklinden uzaklaşır
X, y = make_moons(noise=0.2,random_state=42)
plt.scatter(X[:,0],X[:,1], c = y)
plt.show()

# make circle ile de daire şeklinde oluştururz
# factor ile içteki ve dıştaki daire arasındai mesageler ayarlanı
X, y = make_circles(noise = 0.1,factor=0.3,random_state=42)
plt.scatter(X[:,0],X[:,1])
plt.show()

# alt tarafta yuarıdakileri tek bir çizimde görmeyi yaptık
# artık bizim sınıflandırıcı çeşitlerimizi karşılaştırabileceğimiz verisetlerimiz hazır
datasets = [
        Xy,
        make_moons(noise=0.2,random_state=42),
        make_circles(noise=0.1,factor=0.3,random_state=42)
    ]

fig = plt.figure(figsize=(6,9))
# bu enumerate ile Xy indexi ds_cnt eşitle, ds e ise Xy koy
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    
    #alttaki satır bize 3(len) satır, 1 sütundan oluşacak şekilde ds_count + 1. satırı kastediyor
    ax = plt.subplot(len(datasets),1,ds_cnt + 1)
    #plt.cm.coolwarm kullanılacak renk haritası, genelde zıt renkler olur
    ax.scatter(X[:,0],X[:,1],c = y, cmap=plt.cm.coolwarm,edgecolors="black")

plt.show()


# alt tarafta sınıflandırıcıları oluşturup karşılaştırmalarını yapıyoruz

names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB()
    ]

fig = plt.figure(figsize=(6,9))
i = 1
for ds_cnt, ds in enumerate(datasets):
    X,y = ds
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    # sınıf 0 için koyu kırmızı ve sınıf 1 türü için koyu mavi kullanacağız
    cm_bright = ListedColormap(["darkorange","darkgreen"])
    # veri boyutu kadar satır(3), ve classifier türün + 1(ham data) olacak kadar, i mevcut indexi kasteder
    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    
    # plot training data
    ax.scatter(X_train[:,0],X_train[:,1],c = y_train, cmap = cm_bright, edgecolors = "black")

    # plot test data, test verileri opak olarak çizilir, farkı görebilmek adına
    ax.scatter(X_test[:,0],X_test[:,1],c = y_test, cmap = cm_bright, edgecolors="black",alpha=0.6)
    i += 1

    for name, clf in zip(names,classifiers):
        ax = plt.subplot(len(datasets),len(classifiers)+1,i)
        #alt tarafta önce standatlaştırıyoruz sonra modeli seçiyoruz
        clf = make_pipeline(StandardScaler(),clf)
        clf.fit(X_train,y_train)
        score = clf.score(X_test, y_test) # accuracy değerini elde ettik
        # tüm alanı tarar(X eksenlerine göre hangi alan olduğunu belirleryere), arka planı 0.7 şeffaflıkla tahminine göre boyar
        # 0.5lik bir epsilon(genişleme payı) bırakır
        # arka plan boyamasının rengini rd bu ile yapar, 
        DecisionBoundaryDisplay.from_estimator(clf, X, cmap = plt.cm.RdBu,alpha = 0.7,ax = ax,eps=0.5)
        
        # plot training data
        # ax.scatter(X_train[:,0],X_train[:,1],c = y_train, cmap = cm_bright, edgecolors = "black")

        # plot test data
        ax.scatter(X_test[:,0],X_test[:,1],c = y_test, cmap = cm_bright, edgecolors="black",alpha=0.6)
        
        if ds_cnt == 0:
            ax.set_title(name)
        # sağ altına doğruluk değerini eklemek için
        ax.text(
            X[:,0].max() - 0.15,
            X[:,1].min() - 0.35,
            str(score)
            )
        i += 1

plt.show()

