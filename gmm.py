from sklearn import mixture
import numpy as np
import pandas as pd

file_path = input("input file: ")
data = pd.read_csv(file_path)

minMax = lambda X : (X - np.min(X, axis=0) / (np.max(X, axis=0) - np.min(X, axis=0)))

data = data.values
songs = data[:,-1]
data = minMax(data[:,1:-1])
rows = data.shape[0]
cols = data.shape[1]

n_clusters = int(input("n of clusters: "))
gmm = mixture.BayesianGaussianMixture(n_components=n_clusters).fit(data)
labels = gmm.predict_proba(data)

print(labels)
