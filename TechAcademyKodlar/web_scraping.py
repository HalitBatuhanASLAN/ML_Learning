import requests
import pandas as pd
from bs4 import BeautifulSoup #dönen datayı parse etmeye yarıyor
url = "https://quotes.toscrape.com/"

response_web = requests.get(url)

if response_web.status_code == 200:
    soup = BeautifulSoup(response_web.content,"html.parser")
    sayfa_basligi = soup.find("h1").text
    sayfa_linki = soup.find("h1").find("a")
    sayfa_linki["style"]
else:
    print("Hata hata kodu:",response_web.status_code)
    
    
divler = soup.findAll("div",class_ = "quote")

veri_listesi = []
for div in divler:
    soz = div.find("span",class_ = "text").text
    yazar = div.find("small",class_ = "author").text
    veri_listesi.append({"Yazar":yazar,"Soz":soz})

dataframe_sozler = pd.DataFrame(veri_listesi)

#Pagination
tum_sayfalar_verisi = []

url_sayfalar = "https://quotes.toscrape.com/page/"

for i in range(1,4):
    istek_url = f"{url_sayfalar}{i}"
    time.sleep(10)
    r = requests.get(istek_url)
    s = BeautifulSoup(r.content,"html.parser")
    sayfa_divler = s.findAll("div",class_ = "quote")
    
    for sd in sayfa_divler:
        sayfa_soz = sd.find("span",class_ = "text").text
        sayfa_soyleyen = sd.find("small", class_ = "author").text
        tum_sayfalar_verisi.append({"Yazar":sayfa_soyleyen,"Soz":sayfa_soz})

df_tumSayfalar = pd.DataFrame(tum_sayfalar_verisi)

ibb_url = "https://ibb.istanbul/gundem/duyurular/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


response_ibb = requests.get(ibb_url,headers=headers)

if response_ibb.status_code == 200:
    ibb_soup = BeautifulSoup(response_ibb.content,"html.parser")
    time.sleep(10)

#Selenium
#bununla beraber devam et yukarıdaki istek atma olayına
#bu selenium feci bir şeymiş


