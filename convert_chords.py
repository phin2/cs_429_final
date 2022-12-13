import numpy as np
import pandas as pd
from itertools import chain


data = pd.read_csv('./song_feature_file_full.csv')
print(data)

def to_list(string):
    l = string.replace('[',"")
    l = l.replace(']',"")
    l = l.replace( ' ',"")

    l = l.split(",")

    return l


chords = data['chords'].to_numpy()
chords = [to_list(x) for x in chords]
chords_list = list(chain.from_iterable(chords))


new_chord = []

for song in chords:
    print(chords.index(song))
    song_dict = dict.fromkeys(set(chords_list),0)
    for chord in song:
            song_dict[chord] += 1
    
    new_chord.append(list(song_dict.values()))

chord_df = pd.DataFrame(new_chord,columns = song_dict.keys())


print(chord_df)
print(data)
data = pd.concat([chord_df,data],axis=1)
data = data.drop(['Unnamed: 0'],axis=1)
data = data.drop(['Unnamed: 0.1'],axis=1)
data = data.drop(['chords'],axis=1)

print(data)

data.to_csv('song_features_chord_hist_1.csv')