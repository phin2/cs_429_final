import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("WebAgg")
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import math

data = pd.read_csv('./song_features_chord_hist.csv')
labels = data['song_name']
data = data.iloc[:,1:len(data.axes(1)) - 2]
print(data)
x = data.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
data =  pd.DataFrame(x_scaled)

pca = PCA(n_components = 2)
pca.fit (data)
data_pca = pca.transform(data)
data_pca = pd.DataFrame(data_pca,columns=['X','Y'])

data_arr = data_pca.to_numpy()
np.savetxt('song_data.txt',data_arr)

tsne = TSNE(n_components=2,verbose=1,perplexity=40,n_iter=300)
tsne.fit(data)
data_tsne = tsne.fit_transform(data)
data_tsne = pd.DataFrame(data_tsne,columns=['X','Y'])

data_arr = data_tsne.to_numpy()
np.savetxt('song_data_tsne.txt',data_arr)
#fig = plt.figure()
#ax = fig.add_subplot()


#print(data_pca)
#ax.scatter(data_pca['X'],data_pca['Y'])
#plt.show()
