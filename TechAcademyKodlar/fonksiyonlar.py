#%%Fonksiyonlar

def hello():
    print("Merhaba")
    
hello()

def hello_name(isim:str):
    print("Merhaba", isim)
    
hello_name("Halit")

def maas_hesapla(brut_maas,vergi_orani):
    return brut_maas*(1-vergi_orani)

print(maas_hesapla(100000,0.4))
