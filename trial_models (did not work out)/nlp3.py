import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data = pd.read_csv('song_feature_file_full.csv')
chords = data['chords']
chords = chords.apply(lambda row: row.replace('[',"").replace(']',"").replace('\'',"").replace(",","").replace("-",''))


tfidf = TfidfVectorizer()
tfidf.fit(chords)

song = tfidf.transform(chords)

def find_optimal_clusters(data,max_k):
    iters = range(2,max_k + 1,2)
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k,init_size = 1024,batch_size = 2048,random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))

    f, ax = plt.subplots(1,1)
    ax.plot(iters,sse,marker = 'o')
    plt.show()

#find_optimal_clusters(song,100)

clusters = MiniBatchKMeans(n_clusters=18,init_size=1024,batch_size=2048,random_state=20).fit_predict(song)

def plot(data,labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]),size=3000,replace=False)

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))

    idx = np.random.choice(range(pca.shape[0]),size=300,replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    plt.show()
    
plot(song, clusters)