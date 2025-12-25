import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop("customerID",axis=1)

# map ile Yes kısımlarını 1 olarak ayarlarızi bir dictionary ile hallediyoruz
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

# drop_fisrt ile kukla kolon(başka kolonun tam tersi gibi)
df_encoded = pd.get_dummies(df ,drop_first=True)

X = df_encoded.drop("Churn",axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model = DecisionTreeClassifier(max_depth=4, min_samples_split=10,)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)

plt.figure()
plot_tree(model)
plt.show()








