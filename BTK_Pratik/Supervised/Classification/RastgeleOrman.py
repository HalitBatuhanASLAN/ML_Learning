from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


oli = fetch_olivetti_faces()


# alt tarafta veri setini anlamaya çalıştık
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i + 1) # 2 farklı açıdan görüntü veriyor
    plt.imshow(oli.images[i+ 40], cmap = "gray") # 2 boyutlu olduğu için gray dedik renkli görüntü değil
    plt.axis("off")
plt.show()


X = oli.data
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# 100 tane decision tree olacak, 
rf_clf = RandomForestClassifier(n_estimators= 100, random_state= 42)
rf_clf.fit(X_train,y_train)

y_pred = rf_clf.predict(X_test)
accurancy = accuracy_score(y_test, y_pred)
print("Accuracny:",accurancy)

# %% REgresyon problem kısmı

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

california_housing = fetch_california_housing()

X = california_housing.data
y = california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.2, random_state= 42)

rf_regr = RandomForestRegressor(random_state= 42)
rf_regr.fit(X_train,y_train)

y_pred = rf_regr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:",rmse)




