from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


heart_disease = fetch_ucirepo(id=45)

df = pd.DataFrame(heart_disease.data.features)
df["target"] = heart_disease.data.targets

#drop missing
# nan = not a number
if df.isna().any().any(): # df içinde nan değeri varmı varsa
    df.dropna(inplace= True) # o değeri çıkart ve df güncelle
    print("nan")

X = df.drop(["target"],axis= 1).values # df içinden target çıkartarak X eşitle, values ile bunu dften numpy array oldu
y = df.target.values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.1, random_state= 42)

# penalty = hangi fonksiyonu kullandığımız, düzenleştirme türünü belirler
# C düzenleştirme gücüdür, küçük olması daha güçlü düzenleştirme demek
# lbfgs maliyet fonksiyonun optimezesinde kullanılan algoritmadır
log_reg = LogisticRegression(penalty= "l2",C=1, solver="lbfgs", max_iter=100)
log_reg.fit(X_train,y_train)

accuracy = log_reg.score(X_test,y_test) # bu şekilde tek yerden tahmin ve doğruluk bulunur
print("Accuracy değeri:",accuracy)

