#Logistic Regression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # featureslar arası farkı standatize eder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report # içinde birkaç tanesini barındırır



df = pd.read_csv('diabetes.csv')

df.head()

df.describe()

df.info()

df['Outcome'].value_counts()

#Korelasyon
df.corr()
plt.figure()
sns.heatmap(df.corr(),annot=True, cmap ='coolwarm')
plt.show()


coltofix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


for c in coltofix:
    df[c]=df[c].replace(0,df[c].mean())


X = df.drop('Outcome',axis=1)
y = df['Outcome']

X_train, X_test, y_train,y_test =train_test_split(X,y,test_size=0.3, random_state=42)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# c -> hataya tolerans gibidir, c=1 hata yap canın sağ olsun, c=100 kesinlikle hata yapma
# solver -> hangi matematik algoritmasının kullanılacağı
model = LogisticRegression(C=1,solver="liblinear",random_state=42)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
# precision -> truePositive / (TF + FP)
# recall -> TP / (TP + FN)
# F1-score -> precision ile recall harmonik ortalaması
# o değerden kaç tane örneğimiz var test datasında






