import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("tips")

plt.figure(figsize=(10,8)) # çizimin yapılacağı tablo oluşturulur

 #scatter plot örneği
plt.scatter(df["total_bill"],df["tip"],color="red",alpha=0.5)
#alphaile opaklık verdik

plt.xlabel("Hesap tutarı")
plt.ylabel("Bahşiş miktarı")
plt.title("Hesap ve bahşiş arasındaki ilişki")

plt.show()

# Histogram grafiği örneği
plt.hist(df["total_bill"],color="green")
plt.title("Hesap tutarlarının dağılımı")
plt.xlabel("Hesap tutarı")
plt.ylabel("Hesap adedi")
plt.show()

# bar grafiği örneği
ozet_data = df.groupby("day")["total_bill"].mean() # her gün ortalama hesap tutarı
plt.bar(ozet_data.index, ozet_data.values,color="orange")

plt.show()


#line grafik çizimi
plt.plot(df["total_bill"],color="yellow",linestyle="-",linewidth = 2)
plt.show()

df_sirali = df.sort_values("total_bill").reset_index()
plt.plot(df_sirali["total_bill"],color="yellow",linestyle="-",linewidth = 2)
plt.show()


#%% Seaborn
# kendisi de hesalma yapabildiği için tüm veriyi ister en başta
#scatter plot
kendi_paletim = {
    "Male": "blue",
    "Female": "Yellow"
    }

plt.figure(figsize=(10,10))
sns.scatterplot(data=df,x="total_bill", y="tip",size = 100,hue="sex",palette=kendi_paletim)
#hue ile day kolonuna göre renklendirme yapılıyor
#palette ile renk paleti belirleyebiliriz, viridis, magma ya da kendi paletimiz
plt.show()

# bar plot
sns.barplot(data=df,x="total_bill",y = "day",hue="day",estimator="sum")
plt.show()


#%% Plotly
# yukarıdakiler sadece birer görüntüdür(statiktir), fakat plotly bize tarayıcıda çalışan, 
# etkileşimler sunan bir kütüphanedir
# kendisi de hesalma yapabildiği için tüm veriyi ister en başta

import plotly.express as px # -> çizebileceğimiz grafikleri en hızlı çizebileceğimiz metodlar -> otomatik araba, kendi yapar
# import plotly.graph_objects as go -> en detaylıdır bunda da -> manuel araba -> manuel yaparsın

fig = px.scatter(df,x = "total_bill",y="tip",color="sex",size="size")
# "size" df içindeki size ve matploit kısmındaki alpha gibi ama size büyük küçük oluyor
# color ile neye göre renklendireceğimiz seçilir, seanorn için bu hue oluyordu
fig.show(renderer="browser")
# renderer parametresi ile biz grafiği göreceğimz konumu seçeriz



