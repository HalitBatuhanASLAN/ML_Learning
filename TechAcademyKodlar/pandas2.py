import pandas as pd

#header kaçıncı satırda olsuğunu verir kolon isilerinin
df = pd.read_csv("train_and_test2.csv",sep=",",header=0,usecols=["Passengerid","Age","Sex","Fare"],nrows=100)


df_excel = pd.read_excel("ham_veri.xlsx",sheet_name="Rapor")

#eksik verileri doğrulama
df_excel["Fiyat"] = df_excel["Fiyat"].fillna(0)

df_excel["Toplam tutar"] = df_excel["Fiyet"] * df_excel["Adet"]

df_filtered = df_excel[df_excel["Toplam tutar"] > 100].copy()

df_filtere.to_excel("islenmisVeri.xlsx",sheet_name="Data",index=False)