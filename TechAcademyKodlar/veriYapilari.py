#%%Liste

meyveler = ["Elma","Armut","Karpuz"]
print(meyveler)

#Ekleme yapma
meyveler.append("Muz")
print(meyveler)
#belirli bir indexe ekleme yapma
meyveler.insert(2,"Kavun")
print(meyveler)

#güncelleme
meyveler[2] = "Kiraz"
print(meyveler)

#silme
''' alt tarafta index girilmezse son indextekini siler '''
meyveler.pop(1)
print(meyveler)

print(meyveler[-1])

print(meyveler[0:3])

#%%Tuple

veritabani = ("localhost","admin",123456)

print(veritabani.count("admin"))


#%%Set

set1 = {1,2,3,4,5,0,5,5,5,5,5,5,4}

liste1 = [1,2,3,4,5,0,5,5,5,5,5,5,4]
set2 = set(liste1)

set3 = {4,5,6}
set3.difference(set1)
set.intersection(set2)


#%%Dictionary

kisi = {"ad":"Halit",
        "soyad":"ASLAN",
        "yas":21}

personeller = {
    "Ad":["Ahmet","Mehmet","Ali","Veli"],
    "Sehir":["İstanbul","Ankara","Adana","Ankara"]
    }

personeller["Yas"] = [10,20,30,30]
personeller.get("Yas")
personeller.get("Maas","Aradığınız kolon bulunamadı")
