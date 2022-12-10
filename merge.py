import pandas as pd
import os

dfs = []
for root,dirs,file in os.walk('./feature_files'):
    for name in file:
        dfs.append(pd.read_csv(os.path.join(root,name)))

df = pd.concat(dfs)
df.to_csv('./feature_files/song_features_full.csv')