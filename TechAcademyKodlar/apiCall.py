#%%Api call ile veri çekme
import pandas as pd
import requests 

api_url = "https://jsonplaceholder.typicode.com/users"


#get metodu
response = requests.get(api_url,timeout=10)
if(response.status_code == 200):
    print("Veri başarı ile okundu", response.status_code)
    response_data = response.json()
else:
    print("Veri okunamadı. Hata kodu:",response.status_code)

response_df = pd.DataFrame(response_data)
response_df.to_excel("responseDF.xlsx", index=False)

normalize_df = pd.json_normalize(response_data)
normalize_df.to_excel("normalizeDF.xlsx", index=False)


post_url = "https://jsonplaceholder.typicode.com/posts"
#post metodu

ogrenci = {
    "Ad": "Ali",
    "Yas": 30
    }

post_response = requests.post(post_url, json=ogrenci)

if post_response.status_code == 201:
    print("Veribaşarı ile gönderildi")
else :
    print("Veri gönderilemedi. Hata kodu:",post_response.status_code)
