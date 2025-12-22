from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

# DT modeli oluşturuğ train etme
tree_clf = DecisionTreeClassifier(criterion= "gini", max_depth= 5, random_state= 42) # criterion = entropy
tree_clf.fit(X_train, y_train)

# DT test aşaması
y_pred = tree_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("iris veri seti ile eğitilen DT model doğruluğu:",accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.figure(figsize = (15,20))
#filled true ile her yeri doldurduk, 
plot_tree(tree_clf, filled = True, feature_names= iris.feature_names, class_names= list(iris.target_names))
plt.show()

feature_importance = tree_clf.feature_importances_
feature_names = iris.feature_names
feature_importance_sort = sorted(zip(feature_importance, feature_names), reverse= True)
for importance, feature_name in feature_importance_sort:
    print(f"{feature_name}: {importance}")

#%% Feature selection yapacağız

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

import warnings
warnings.filterwarnings("ignore") #uyarı mesajlarını görmezden gelir

iris = load_iris()

n_classes = len(iris.target_names)
plot_colors = "ryb" # renklendirmedeki renk kodları red yellow blue

# farklı feature çifleri için algoritmayı çalıştırup öğrenir
for pairidx, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    X = iris.data[:, pair]
    y = iris.target
    
    clf = DecisionTreeClassifier().fit(X,y)
    
    ax = plt.subplot(2,3, pairidx + 1) # 2,3 olacak tablolar
    plt.tight_layout(h_pad= 0.5, w_pad= 0.5, pad= 2.5)
    # height->yatayda duranlar arasındaki moşbluk
    # weight->dikeyde, üst üste duranlar araındaki boşluk
    #pad dışarı ile olan boşluklar
    
    # train sonucu elde edilen sınıflandırmadaki sınırları görselleştirdik, arka plan
    # cmap = color map kullanılacak renk haritasıdır
    # ax o sıradaki tabloda çizimi ayarlar
    # response metdo ile modelin tahmininin sınırlarını kullandırtır
    DecisionBoundaryDisplay.from_estimator(clf,
                                           X,
                                           cmap = plt.cm.RdYlBu,
                                           response_method= "predict",
                                           ax = ax,
                                           xlabel=iris.feature_names[pair[0]],
                                           ylabel=iris.feature_names[pair[1]])

    # renkler ile o anki sınıflar arasında döngü oluşturur
    #idx o anki sınıfa ait veri noktalarının indexlerini bulur
    # c o anki sınıfa ait atanmış rengi verir
    # o noktanın indelerini verir
    # label ile o ana ait olan türün adını tutar
    # edgecolor ile her noktanın etraına siyat kenarlık koyarak noktaların ayırt edilmesi sağlanı
    
    for i, color in zip(range(n_classes),plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx,0], X[idx,1],c = color, label = iris.target_names[i],
                    cmap = plt.cm.RdYlBu,
                    edgecolors="black")

plt.legend() # yazdığımız labeller görselleştirmek için

#%% Regresyon problemi kısmı

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

diabets = load_diabetes()

X = diabets.data
y = diabets.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 42)

# kara ağacı regresyon modeli
tree_reg = DecisionTreeRegressor(random_state= 42).fit(X_train,y_train)

y_pred = tree_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE:",mse)

rmse = np.sqrt(mse)
print("RMSE:",rmse)


# %% Kendi oluşturduğumuz üzerinden regresyon

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

#create dataset

X = np.sort(5 * np.random.rand(80,1), axis= 0)
y = np.sin(X).ravel() # arrayi vektor olarak yağar, düzleştirir yani
y[::5] += 0.5 * (0.5 - np.random.rand(16)) # 80/5 16 tane üretmesi yeterli


regr_1 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(X,y)

regr_2 = DecisionTreeRegressor(max_depth=7)
regr_2.fit(X,y)

X_test = np.arange(0,5,0.05)[:, np.newaxis]
y_pred1 = regr_1.predict(X_test)
y_pred2 = regr_2.predict(X_test)

plt.figure()
plt.plot(X, y,c = "red",label = "Original data")
plt.scatter(X, y,c = "red",label = "Original data")
plt.plot(X_test,y_pred1, color = "blue",label= "MaxDepth:2",linewidth = 2)
plt.plot(X_test,y_pred2, color = "green",label= "MaxDepth:7",linewidth = 2)

plt.xlabel("data")
plt.ylabel("target")
plt.legend() # label gözükmesi için
