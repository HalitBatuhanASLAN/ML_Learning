from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# veri oluşturma
X = np.random.rand (100,1)
y = 3 + 4 * X + np.random.rand(100,1)

lin_reg = LinearRegression()
lin_reg.fit(X,y)

plt.figure()
plt.scatter(X,y)
plt.plot(X,lin_reg.predict(X), color = "red",alpha = 0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")

# y = 3 + 4*X -> y = a0 + a1x
a1 = lin_reg.coef_[0][0]
print("a1:",a1)

a0 = lin_reg.intercept_[0]
print("a0:", a0)


for i in range(1,100):
    y_ = a0 + a1 * X
    plt.plot(X,y_,color = "green",alpha = 0.7)

# %% Diabet veri seti üzerinden örnek

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

diabets = load_diabetes()

diabets_X, diabets_y = load_diabetes(return_X_y=True) # otomatik olarak yapma

diabets_X = diabets_X[:,np.newaxis,2] # tüm satıları al, 2.sütunu al

diabets_X_train = diabets_X[:-20] # son 20si hariç al
diabets_X_test = diabets_X[-20:] # son 20yi al

diabets_y_train = diabets_y[:-20]
diabets_y_test = diabets_y[-20:]

lin_reg = LinearRegression()
lin_reg.fit(diabets_X_train, diabets_y_train)
diabets_y_pred = lin_reg.predict(diabets_X_test)

mse = mean_squared_error(diabets_y_test, diabets_y_pred)
print("mse:",mse)

# r2-> çıkan sonucun 100 ile çarpılması bizim kullandığımız parametrenin yüzde o kadarını
# seçtiğim bu değişken açıklar
r2 = r2_score(diabets_y_test, diabets_y_pred)
print("r2:",r2)

plt.scatter(diabets_X_test,diabets_y_test,color="black")
plt.plot(diabets_X_test,diabets_y_pred,color="blue")

plt.show()



