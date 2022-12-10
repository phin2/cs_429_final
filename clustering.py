import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from networkx.algorithms.community.label_propagation import label_propagation_communities
import random
import math

df = pd.read_csv('inputs_test.csv')




def to_list(string):
    l = string.replace('[',"")
    l = l.replace(']',"")
    l = l.replace( ' ',"")

    l = l.split(",")
    l = np.asarray([int(i) for i in l])

    return l

def motion_sim(m1,m2):
    return 0

#calculates similarity for two songs bpm
def bpm_sim(bpm1,bpm2):
    return 1 - abs(bpm1 - bpm2)/(bpm1 + bpm2)

def cos_sim(c1,c2):
    if type(c1) == str:
        c1 = to_list(c1)
    if type(c2) == str:
        c2 = to_list(c2)

    return np.abs(np.dot(c1,c2)/(np.multiply(np.linalg.norm(c1),np.linalg.norm(c2))))

#calculates similarity of bpm between all songs in a dataset
def bpm_all(data):
    bpms = data['bpm'].to_numpy()
    bpm_vec = np.vectorize(bpm_sim)
    bpm_score = np.zeros((len(bpms),len(bpms)))
    bpm_score = bpm_vec(bpms[:,None],bpms[None,:])
    return bpm_score

def chord_all(data):
    chords = data['chords'].to_numpy()

    chord_vec = np.vectorize(cos_sim)
    chord_score = np.zeros((len(chords),len(chords)))
    chord_score = chord_vec(chords[:,None],chords[None,:])
    return chord_score

def motion_all(data):
    motions = data['melodic_motion'].to_numpy()
    
    max_len = 0
    for i in range(0,len(motions)):
        motions[i] = to_list(motions[i])                                                
    
    for m in motions:
        if len(m) > max_len:
            max_len = len(m)

    for i in range(0,len(motions)):
        motions[i] =  np.concatenate((motions[i],np.zeros(max_len - len(motions[i]))))

    motions_vec = np.vectorize(cos_sim)
    motion_score = np.zeros((max_len,max_len))
    motion_score = motions_vec(motions[:,None],motions[None,:])
    #print(motion_score)
    return motion_score


def graph(data,sim_scores,threshold):
    G = nx.Graph()

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
motion_scores = motion_all(df)

G_bpm,bpm_communities = graph(df,bpm_scores,0.999)
G_chord,chord_communities = graph(df,chord_scores,0.55)
G_motion, motion_communities = graph(df,motion_scores,0.1)

#color_map_bpm = np.zeros(len())
color_map_bpm = np.zeros(len(G_bpm.nodes))
for i in bpm_communities:
    for j in i:
        #print(j)
        #print(list(G_bpm.nodes).index(j))
        color_map_bpm[list(G_bpm.nodes).index(j)] = list(bpm_communities).index(i)

color_map_chord = np.zeros(len(G_bpm.nodes))
for i in chord_communities:
    for j in i:
        #print(j)
        #print(list(G_bpm.nodes).index(j))
        color_map_chord[list(G_chord.nodes).index(j)] = list(chord_communities).index(i)

color_map_motion = np.zeros(len(G_bpm.nodes))
for i in motion_communities:
    for j in i:
        #print(j)
        #print(list(G_bpm.nodes).index(j))
        color_map_motion[list(G_motion.nodes).index(j)] = list(motion_communities).index(i)

print('BPM Communities:')
for i in bpm_communities:
    print(i)
print("------------------------------------------------------------------------------")
print('Chord Communities:')
for i in chord_communities:
    print(i)
print("------------------------------------------------------------------------------")
print('Motion Communities:')
for i in motion_communities:
    print(i)


pos_bpm = nx.spring_layout(G_bpm,k=5/math.sqrt(G_bpm.order()))
pos_chord = nx.spring_layout(G_chord,k=5/math.sqrt(G_chord.order()))
pos_motion = nx.spring_layout(G_motion,k=5/math.sqrt(G_motion.order()))

plt.figure("Chords")
nx.draw(G_chord,pos_chord,with_labels=True,node_color = color_map_chord,font_size=7)
plt.figure("Bpm")
nx.draw(G_bpm,pos_bpm,with_labels=True,node_color = color_map_bpm,font_size=7)
plt.figure("Motion")
nx.draw(G_motion,pos_motion,with_labels=True,node_color = color_map_motion,font_size=7)


plt.show()
