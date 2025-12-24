import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # grafik için daha iyi, matplotlib üzerine kurulu zaten
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("insurance.csv")

# 2)veri analizi
df.head(20)
df.info()
df.describe()

plt.figure()
sns.barplot(data = df, x = "smoker", y = "charges")
plt.show()

sns.boxplot(data = df, x = "smoker", y = "charges")
plt.show()

# 3 vari hazırlanışı
# feature engineering -> encoding -> one hot encoding

df_encoded = pd.get_dummies(df,columns= ["sex","smoker","region"],drop_first=True)

# axis = 1 ile dikeyde(sütun) silme yapar
X = df_encoded.drop("charges", axis = 1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model selection

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

y_pred = lin_reg.predict(X_test)

# evulation
r2 = r2_score(y_test, y_pred)
print(f"r2 score: {r2:.4f}") # noktadan sonra kaç basamak

mse = mean_squared_error(y_test, y_pred)
print("mse:",mse)

# 6 -> deployment
# stringlit kütüphansei ile hızlıca web sitesi oluşturabiliriz
# ekranda ilgili bilgilri istiyor bunlar gelmiş gibi düşün
new_customer = pd.DataFrame([[25,30,1,0,0,1,0,0]], columns= X_train.columns)
new_customer_pred = lin_reg.predict(new_customer)
print("Tahmini sigorta policçe değeriniz :",new_customer_pred)


# alt taraf tekrardan test, çünkü ilk başarı yeterli değil
df_encoded_new = pd.get_dummies(df,columns= ["sex","smoker","region"],drop_first=True)
df_encoded_new["bmi_smoker"] = df_encoded_new["bmi"] * df_encoded_new["smoker_yes"]

df_encoded_new["is_obese"] = df_encoded_new["bmi"].apply(lambda x: 1 if x > 30 else 0)

X_new2 = df_encoded_new.drop("charges", axis = 1)
X_new1 = X_new2.drop("smoker_yes", axis = 1)
X_new = X_new1.drop("bmi", axis = 1)
y_new = df_encoded_new["charges"]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new,y_new,test_size=0.2,random_state=42)

model_new = LinearRegression()
model_new.fit(X_train_new,y_train_new)
y_pred_new = model_new.predict(X_test_new)

r2_new = r2_score(y_test_new, y_pred_new)
print(f"r2 score: {r2_new:.4f}") # noktadan sonra kaç basamak

mse_new = mean_squared_error(y_test_new, y_pred_new)
print("mse:",mse_new)









