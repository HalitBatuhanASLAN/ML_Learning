import pandas as pd

liste1 = [1,2,3,4,5]

seri = pd.Series(liste1)

harfler = pd.Series(["a","b","c"])

data = {
        "Ad":["Ali","Veli","Ahmet"],
        "Şehir":["Ankara","Adana","İstanbul"]
        }

dataframe = pd.DataFrame(data)
dataframe.head() #ilk 5 satır default
dataframe.tail(2) #son 2 satır
dataframe.sample(2) #rastgele 2 satır getir
dataframe.dtypes
dataframe.columns

#df üzerinde veri işlemleri

dataframe.loc[1]
dataframe.loc[dataframe["Ad"] == "Ahmet","Şehir"] = "Kocaeli"

dataframe.loc[5] = ["Mehmet","Sivas"]

dataframe.loc[len(dataframe)] = ["HBA","Angara"]


#silme
dataframe["Maaş"] = 500

#axis=1 sütun için
dataframe.drop("Maaş",axis=1) # sadece silinenedne osnrasını yazar ama variableda gözükmez
dataframe.drop("Maaş",axis=1,inplace=True)

#axis=0 satır için
dataframe.drop(2,axis=0,inplace=True)

#merge-> sqlde join oluyor

df_calisanlar = pd.DataFrame({
    "Calisan_ID": [101, 102, 103, 104],
    "Ad": ["Ali", "Veli", "Ayşe", "Fatma"],
    "BolumID": ["1", "1", "5", "3"]
})

df_bolumler = pd.DataFrame({
    "BolumID": ["1", "2", "3", "4"],
    "Bolum": ["IT", "HR", "Finance", "Legal"]
})

merge_inner_df = pd.merge(df_calisanlar, df_bolumler, on="BolumID",how="inner") 

merge_left_df = pd.merge(df_calisanlar, df_bolumler, on="BolumID",how="left")

merge_right_df = pd.merge(df_calisanlar, df_bolumler, on="BolumID",how="right")

#concat -> sqlde union


df1 = pd.DataFrame({"Sira":[1,2,3]})
df2 = pd.DataFrame({"Sira":[4,5,6]})

#dikeyde
pd.concat([df1,df2])

#yatayda
pd.concat([df1,df2], axis=1)

pd.concat([df1,df2], axis=1, ignore_index=True)








