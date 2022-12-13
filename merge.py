import pandas as pd
import os

dfs = []
for root,dirs,file in os.walk('./feature_files'):
    for name in file:
        dfs.append(pd.read_csv(os.path.join(root,name)))

df = pd.concat(dfs,ignore_index=True)
df.to_csv('./song_feature_file_full.csv')