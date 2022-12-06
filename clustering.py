import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from networkx.algorithms.community.label_propagation import label_propagation_communities


df = pd.read_csv('inputs.csv')

#calculates similarity for two songs bpm
def bpm_sim(bpm1,bpm2):
    return 1 - abs(bpm1 - bpm2)/(bpm1 + bpm2)

def chord_sim(c1,c2):
    c1 = c1.replace('[',"")
    c1 = c1.replace(']',"")
    c1 = c1.replace(' ',"")
    
    c2 = c2.replace('[',"")
    c2 = c2.replace(']',"")
    c2 = c2.replace(' ',"")

    c1 = c1.split(",")
    c2 = c2.split(",")

    c1 = np.asarray([int(i) for i in c1])
    c2 = np.asarray([int(i) for i in c2])

    return np.dot(c1,c2)/(np.multiply(np.linalg.norm(c1),np.linalg.norm(c2)))

#calculates similarity of bpm between all songs in a dataset
def bpm_all(data):
    bpms = data['bpm'].to_numpy()
    bpm_vec = np.vectorize(bpm_sim)
    bpm_score = np.zeros((len(bpms),len(bpms)))
    bpm_score = bpm_vec(bpms[:,None],bpms[None,:])
    return bpm_score

def chord_all(data):
    chords = data['chords'].to_numpy()

    chord_vec = np.vectorize(chord_sim)
    chord_score = np.zeros((len(chords),len(chords)))
    chord_score = chord_vec(chords[:,None],chords[None,:])
    print(chord_score)
    return chord_score

def bpm_graph(data,sim_scores):
    G = nx.Graph()
    threshold = 0.99
    
    for index,song in data.iterrows():
        G.add_node(song['song_name'])
        edges = np.where(sim_scores[index] > threshold)
        for e in edges[0]:
            if e != index:
                G.add_edge(song['song_name'],data['song_name'][e])

    communities = label_propagation_communities(G)
    return G, communities

def chord_graph(data,sim_scores):
    G = nx.Graph()
    threshold = 0.50

    for index,song in data.iterrows():
        G.add_node(song['song_name'])
        edges = np.where(sim_scores[index] > threshold)
        for e in edges[0]:
            if e != index:
                G.add_edge(song['song_name'],data['song_name'][e])

    communities = label_propagation_communities(G)
    return G,communities

bpm_scores = bpm_all(df)
chord_scores = chord_all(df)

G_bpm,bpm_communities = bpm_graph(df,bpm_scores)
G_chord,chord_communities = chord_graph(df,chord_scores)


plt.figure("Chords")
nx.draw(G_chord,with_labels=True,font_size=7)
plt.figure("Bpm")
nx.draw(G_bpm,with_labels=True,font_size=7)

plt.show()
