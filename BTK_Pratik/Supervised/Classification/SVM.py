from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# SVC support vector classification
from sklearn.metrics import classification_report

digits = load_digits()

# 2 satır, 5 sütunluk figürler oluştu
# subplot_kw ile x ye y çizgilerindeki işaretleri(sayıları) kaldırdık
fig, axes = plt.subplots(nrows= 2, ncols= 5, figsize=(10,5),
                         subplot_kw={"xticks":[], "yticks": []})
# olşan 10 grafik içinde sırayla dolaş
for i, ax in enumerate(axes.flat):
    # cmap binary ile sayısal değer(piksel değeri) ikili yani siyah beyaz olarak yazılır
    # küçük boyutlu bir resmin ekrana yansıtılırken pikseller arasında nasıl doldurulacağı belirlenir
    # nearest orjinali yapar, bilinear dersek pikseller birbirine karışırdı
    # digites.images ile de ilgiki sıradaki rakamın piksel değerini verir fonksiyona
    ax.imshow(digits.images[i],cmap="binary",interpolation = "nearest")
    # görüntünün üzerine sayının gerçek değerini yazar
    ax.set_title(digits.target[i])

plt.show()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

# linear çekirdekli bir svm oluşturur
svm_clf = SVC(kernel="linear", random_state=42)
svm_clf.fit(X_train,y_train)

y_pred = svm_clf.predict(X_test)

print(classification_report(y_test, y_pred))