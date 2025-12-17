import sqlalchemy as sa
#sql bağlantısı için yukarıdaki işe yarar
import pandas as pd

SERVER = ""
DATABASE = ""
USERNAME = ""
PASSWORD = ""
DRIVER = "ODBS DRIVER "

#önce kaynak + hangi kütüphane ile istek atıalcak + 
conn_str = f"mssql+pyodbc://{USERNAME}:"

conn_str = f"mssql+pyodbc://{SERVER}/{DATABASE}?driver={DRIVER}&trusted_connection=yes"


try:
    engine = sa.create_engine(conn_str)
    engine.connect()
    print("Connection kuruldu")
except Exception as e:
    print("Hata bulundu, hatanın mesajı ise;",e)
    
#bağlantı tamamlandı


#veri okuma
querry = "SELECT * FROM []"
df_sql = pd.read_sql(querry, engine)


ogrenciler = pd.DataFrame({
    "Ad":["Ali","Ahmet","Veli"],
    "Notlar":[90,95,100]
    })


ogrenciler.to_sql(name="Öğrenciler",con=engine,index=False,if_exists="replace")

yeni_ogrenciler = pd.DataFrame({
    "Ad":["Ali","Ahmet","Veli"],
    "Notlar":[90,95,100]
    })

yeni_ogrenciler.to_sql(name="Öğrenciler",con=engine,index=False,if_exists="append",chunksize=2)

    










