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


def compute_tf(chord_dict,lists):
    tf = {}
    sum_nk = len(lists)
    for chord,count in chord_dict.items():
        tf[chord] = count/sum_nk
    return tf

def compute_idf(song_list):
    n = len(song_list)
    idf = dict.fromkeys(song_list[0],0)
    for song in song_list:
        for chord,count in song.items():
            if count > 0:
                idf[chord] += 1
            
    for chord,v in idf.items():
        idf[chord] = np.log(n/float(v))
    return idf

def compute_tf_idf(tf,idf):
    tf_idf = dict.fromkeys(tf.keys(),0)
    for chord, v in tf.items():
        tf_idf[chord] = v * idf[chord]

    return tf_idf

data = pd.read_csv('song_feature_file_full.csv')
chords = data['chords']
chords = chords.apply(lambda row: row.replace('[',"").replace(']',"").replace('\'',"").replace(",","").replace("-",''))
chords = chords.to_numpy()
chord_list = []
for i in range(0,len(chords)):
    chord_list += chords[i].split()
    chords[i] = np.asarray(chords[i].split())

bow = []
for i in range(0,len(chords)):    
    chord_list = dict.fromkeys(chord_list,0)
    for chord in chords[i]:
        chord_list[chord] += 1

    bow.append(chord_list)

tf_chords = []
for i in range(0,len(chords)):
    tf_chords.append(compute_tf(bow[i],chords[i]))
    
idf = compute_idf(bow)

tf_idf_chords = []
for i in range(0,len(chords)):
    tf_idf_chords.append(compute_tf_idf(tf_chords[i],idf))

print(pd.DataFrame(tf_idf_chords))