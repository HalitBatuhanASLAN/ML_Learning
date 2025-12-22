import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# y = a0 + a1x -> linear regression
# y = a0 + a1x1 + a2x2 + ... + anxn -> multi variable linear regression
# y = a0 + a1x1 + a2x2 -> bizim örnekte bu var

X = np.random.rand(100, 2)  # x1, x2 olacağı için 2 değişkenli dedik
coef = np.array([3, 5])  # katsayıalrımız
y = np.random.rand(100) + np.dot(X, coef)  # bunların dot product çarpımı

fig = plt.figure()
# ax = fig.add_subplot(111,projection = "3d")
# ax.scatter(X[:,0],X[:,1],y)
# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
# ax.set_zlabel("y")
# plt.show()


lin_reg = LinearRegression()
lin_reg.fit(X, y)

ax = fig.add_subplot(111, projection="3d") # 3 boyutlı olarak çizecek
# burada 1 -> sayır sayısı; 1-> sütun sayısı; 1-> tablodaki 1.grafik
ax.scatter(X[:, 0], X[:, 1], y) # gerçek veri noktalarını çizer
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10)) # 0-1 arası ızgara zemin oluşturu
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T) # tek boyuta indirger ve tranpozunu alır
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), alpha=0.3) # tahmin edilen y'leri şeffaf bir şekilde çiz
plt.title("Mutli variable linear regression")

plt.show()

print("Katsayılar:", lin_reg.coef_)
print("Kesişim:", lin_reg.intercept_)


# %% Diabet örneği
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

diabets = load_diabetes()

X = diabets.data
y = diabets.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state= 42)

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# üstteki yerine kolay yol
rmse = mean_squared_error(y_test, y_pred, squared=False) # false olunca rmse oluyor direkt
# fakat bu version 1.6 ile kaldırıldı yani size uyarı mesajı vermesi doğaldır
print("rmse:",rmse)





























