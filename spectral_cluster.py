from sklearn import cluster
import numpy as np
import pandas as pd

file_path = input("input file: ")
data = pd.read_csv(file_path)

data = data.values
songs = data[:,-1]
data = data[:,1:-1]
rows = data.shape[0]
cols = data.shape[1]

n_clusters = int(input("n of clusters: "))
sc = cluster.SpectralClustering(n_clusters=n_clusters).fit(data)
labels = sc.labels_

for i in range(n_clusters):
    print("cluster", i)
    print(songs[labels==i])