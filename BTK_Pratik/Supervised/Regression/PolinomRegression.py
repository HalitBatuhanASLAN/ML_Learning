import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = 4 * np.random.rand(100,1)
# y = 2 + 3x^2
y = 2 + 3 * X**2 + np.random.rand(100,1)

poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X) # Xin karesini alır

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

plt.scatter(X, y, color="blue")

# yukarıda 0-4 arası olduğu için random verilerimiz, alttaki test veri setimiz
X_test = np.linspace(0, 4, 100).reshape(-1, 1) # -1 -> satır sayısını sen ayarla, 1 -> ama 1 sütun olacak
X_test_poly = poly_feat.fit_transform(X_test)
y_pred = poly_reg.predict(X_test_poly)

plt.plot(X_test, y_pred, color="red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polino regresyon modeli")











plt.show()









