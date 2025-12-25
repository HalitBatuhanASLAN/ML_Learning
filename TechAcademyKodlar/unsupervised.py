import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("spotfy_songs.csv")

col = ["danceability","energy","liveness","loudness","tempo"]

X = df[col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

optimal_k = 4

kmeans = KMeans(n_clusters= optimal_k,init="k-means++",n_init=10)

# model eğitimi yok, model çalışırken o an datayı ayıracak ve anlayacak
y_kmeans = kmeans.fit_predict(X_scaled)

df["tur"] = y_kmeans


