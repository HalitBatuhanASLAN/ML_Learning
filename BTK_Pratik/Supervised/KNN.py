# 1) Veri seti incelemesi

# Sklearn = Bu kütüphane ML algoritmaları için kullanılan bir kütüphanedir.
# Bu kütüphanede kullanabileceğimiz hazır veri setleri bulunmaktadır
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()

df = pd.DataFrame(data= cancer.data, columns= cancer.feature_names)
df["target"] = cancer.target

# 2) Modelin seçimi -> KNN sınıflandırıcı
# biz model olarak zaten KNN seçtik

# 3) Modelin train edilmesi

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split # veriyi bölen fonkisyon
from sklearn.preprocessing import StandardScaler

# X feature; y target
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 42)
# 42 değeri seed olarak geçer(42 olayından dolayı tercih edilir yoksa 06 da verebilirsin), kodun her çalıştığında aynı değeri vermesini sağlar
#yeniden aç kapa yapınca ya da başka bilgisayarda düşükte olsa değişebilir

# ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# önce parametleri öğrenip(standart sapma ortalam vs) fit ediyor sonra bunları transofrm ediyor
X_test = scaler.transform(X_test)

# knn oluşturma ve train etme
knn = KNeighborsClassifier(n_neighbors= 3) # Model oluşturma
knn.fit(X_train,y_train) # fit fonksitonu samples + target kullanarak modeli(knn algorsunu) eğitir

# sklearn için tüm trainker .fit ile olur
# **** X harfi features için; y harfi ise target için kullanılır

# 4) Sonuçların değerlendirilmesi

from sklearn.metrics import accuracy_score # doğruluk skoru hesaplama fonksiyonu
from sklearn.metrics import confusion_matrix # karışıklık matrixi fonksiyonu
# predic ile test edilme kısmı olur
y_pred = knn.predict(X_test) # y predictionları

accuracy = accuracy_score(y_test,y_pred)
print("Doğruluk oranı:",accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(conf_matrix)

# 5) Hiperparametrelerin ayarlanması
'''
    KNN: Htperparameter = K
        K: 1,2,3 ... N
        Accuracy = %A, %B, ... 
'''
import matplotlib.pyplot as plt
accuracy_values = []
k_values = []

#alt tarafta farklı k değerleri(hiperparametre) için doğruluk oranlarını buluyoruz

for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    k_values.append(k)
    accuracy_values.append(accuracy)

# matpltlib kütüphanesinden yararlanarak bunu grafikleştiriyoruz
plt.figure() # yeni bir çizim figürü oluştur
plt.plot(k_values,accuracy_values, marker = 'o', linestyle= "-") # çizgi grafiğini çizer
plt.title("K değrine göre doğruluk") 
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values) # x üzerinde sadece denen değerleri gösterir okumayı kolaylaştırır
plt.grid(True) # ızgara ekleyerek görüntüyü güzelleştirir



# %% Burada da bir regresyon probleminde KNN algoritmasını kullanalım

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X = np.sort(5 * np.random.rand(40, 1),axis= 0) # uniform dağılımla 0-1 arasında 40 sayı, 1 boyulu vekör,
#sonra 5 ile çarparak bunu 0-5 arası yaotık
# axis = 0 ile bunun satır boyunca(dikey) işlem olduğunu gösterdik
y = np.sin(X).ravel() # target, ravel ile tek boyutlu halde tutuyoruz sonrasındaki gürültü ekleme kısmının daha düzgün çalışması için

plt.scatter(X,y) # scatter sayesinde aradaki boşlukları görürüz

#add noise
y[::5] += 1 * (0.5 - np.random.rand(8)) # 8 rastgele sayı üret bunları 0.5 ten çıkart hem 
#pozitif hem negatif sayı var sonra 1 ile çarp(gürültünün büyüklüğü), sonra y üzerine 5er indexli olarak ekle

plt.scatter(X,y)

T = np.linspace(0,5, 500)[:, np.newaxis] # 0-5 arasında 500 sayı oluşturur ve bu vektörde 500,1 şeklinde bir vektör oluşturur
weight = "uniform"

#alttaki for döngüsü i = 0 iken weight uniform;1 iken distance oluyor
for i, weight in enumerate(["uniform","distance"]):
    knn = KNeighborsRegressor(n_neighbors= 5, weights= weight)
    y_pred = knn.fit(X, y).predict(T)
    
    plt.subplot(2,1,i+1) # 2 satır 1 sütun olarak böler 2 tablo olacağı için,hangi satırda işlem yağılacağını i+1 belirier
    plt.scatter(X, y,color = "green", label = "data") #dataları yeşil ile çizer
    plt.plot(T, y_pred, color = "orange", label = "predict")
    plt.axis("tight") #axisleri veriyre göre sıkıca ayarlar
    plt.legend() # etiketleri gösterir
    plt.title("KNN regressor weights = {}".format(weight))

plt.tight_layout() # grafiklerin birbirine girmesini engeller
plt.show()












