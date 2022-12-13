import pickle
import pandas as pd
import numpy as np
import re
import os
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


data = pd.read_csv('song_feature_file_full.csv')
chords = data['chords']
chords = chords.apply(lambda row: row.replace('[',"").replace(']',"").replace('\'',"").replace(",",""))

labeledSong1 = gensim.models.doc2vec.TaggedDocument
all_content_train = []
j = 0

for em in chords.values:
    all_content_train.append(labeledSong1(em,[j]))
    j += 1

print("Songs Processed: ", j)

d2v_model = Doc2Vec(all_content_train,window=10,min_count = 500,workers=1,dm=1,alpha=0.025,min_alpha=0.001)

d2v_model.train(all_content_train,total_examples=d2v_model.corpus_count,epochs=10,start_alpha=0.002,end_alpha=-0.016)

kmeans_model = KMeans(n_clusters=4,init='k-means++',max_iter=1000)
x = kmeans_model.fit(d2v_model.dv.vectors)
labels = kmeans_model.labels_.tolist()

l = kmeans_model.fit_predict(d2v_model.dv.vectors)
pca = PCA(n_components=2).fit(d2v_model.dv.vectors)
datapoint = pca.transform(d2v_model.dv.vectors)

plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker="^", s=150, c="#000000")
plt.show()