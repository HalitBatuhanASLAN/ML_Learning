class BosAraba:
    pass
araba1 = BosAraba()

class Otobus:
    yolcular = []
    
angaram_otobüsü = Otobus()

angaram_otobüsü.yolcular.append("Atatürk")
angaram_otobüsü.yolcular.append("Seymenler")
print(angaram_otobüsü.yolcular)

adana_otobüsü = Otobus()
print(adana_otobüsü.yolcular) # bu direkt yukarıdaki yolcuların aynısıdır static data gibi

class Otobusler:
    def __init__(self,yolcular):
        self.yolcular = yolcular
    def yolcuSayisi(self):
        return len(self.yolcular)
        
otobus1 = Otobusler(["Ahmet","Mehmet"])
print(otobus1.yolcular)
otobus2 = Otobusler(["Ali"])
print(otobus2.yolcular)    
    
print(otobus2.yolcuSayisi())



class Araba:
    def __init__(self,marka,yil):
        self.marka = marka
        self.yil = yil
        print(self.marka,"markalı araç",self.yil,"yılında üretilmiştir")
merso = Araba("Mercedes",1997)
bmw = Araba("BMW",2025)



class GidenAraba:
    def __init__(self,marka):
        self.marka = marka
        self.motor = False
        self.hiz = 0
        self.fren = False
    
    def elFreni(self):
        self.fren = True
        
    def calistir(self):
        print("Araba çalıştı.Gür gür gür")
        self.motor = True
        self.elFreni()
        
    def hizlan(self):
        self.hiz += 100

gidenAraba1 = GidenAraba("Ford")

print(gidenAraba1.fren)
gidenAraba1.calistir()
print(gidenAraba1.motor)
print(gidenAraba1.fren)

gidenAraba1.hizlan()
print(gidenAraba1.hiz)
gidenAraba1.hizlan()
print(gidenAraba1.hiz)

#%%Encapsulation

class BankaHesabi:
    def __init__(self,iban,isim):
        self.iban = iban
        self.tutar = 0
        self.isim = isim

bankaHesabi1 = BankaHesabi("tr213123", "HBA")
print(bankaHesabi1.tutar)

class GuvenlikliBankaHesabi:
    def __init__(self,iban,isim):
        self.iban = iban
        self.__tutar = 0
        self.isim = isim
    def tutarGoruntule(self):
        print(self.__tutar)
    def havaleYap(self,miktar):
        self.__tutar = miktar
guvenlikliBankaHesabi= GuvenlikliBankaHesabi("tr123", "HBA")
guvenlikliBankaHesabi.havaleYap(100)
print(guvenlikliBankaHesabi.tutarGoruntule())

#%%İnheritance

class Hayvan:
    def __init__(self,isim):
        self.isim = isim
        
class Kurt(Hayvan):
    print("AUUUUUU AUUUUUUU")
class Kanarya(Hayvan):
    print("Şanlı Fenerbahçe")
    
    
kurt1 = Kurt("Alfa")
print(kurt1.isim)


#%%Polymorphism
class Hayvan:
    def sesCikar(self):
        pass

class Kopek(Hayvan):
    def sesCikar(self):
        print("Hav hav")

class Kedi(Hayvan):
    def sesCikar(self):
        print("Miyav miyav")
        
kopek1 = Kopek()
kopek1.sesCikar()

kedi1 = Kedi()
kedi1.sesCikar()

#%%Abstraction

from abc import ABC, abstractclassmethod

class Arac(ABC):
    #alt taraf böylece abstarcct metod oldu
    @abstractclassmethod
    def gitmek(self):
        pass
class Araba(Arac):
    def gitmek(self):
        print("Araba gidiyor")

class Otobus(Arac):
    def gitmek(self):
        print("Otobüs gidiyoor")








