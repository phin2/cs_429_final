import numpy as np
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import math
import sys
warnings.filterwarnings("ignore")

names = pd.read_csv('song_features_chord_hist.csv')['song_name'].to_numpy()

#caluclates distance between all points
def dist(x,y,x_c,y_c):
    dist_x = np.abs(x[:,None] - x_c[None,:])
    dist_y = np.abs(y[:,None] - y_c[None,:])

    dist_x_square = np.square(dist_x)
    dist_y_square = np.square(dist_y)

    dist = dist_x_square + dist_y_square
    dist = np.sqrt(dist)

    return dist

#initializes centroids using kmeans++ algorith
def init_centroids(x,y,num_c):
    centroids = []
    centroids.append(np.random.randint(len(x)))

    for i in range(num_c - 1):
        dist_c = dist(x,y,x[centroids],y[centroids])
        min_dist = [min(row) for row in dist_c]
        prob_c = min_dist/sum(min_dist)

        centroids.append(np.argmax(prob_c))
    return centroids

#initializes centroids randomly
def init_centroids_rand(x,num_c):
    centroids = np.random.randint(len(x),size=num_c)
    return centroids

NUM_CENTROIDS_ARR = [10]
n_epochs = int(sys.argv[1])
data = pd.read_csv(str(sys.argv[2]), delimiter=' ', header=None, names=['X','Y'])

X = data['X'].to_numpy()
Y = data['Y'].to_numpy()
#randomly generate centroid indexes
#C = np.random.choice(np.shape(X)[0], NUM_CENTROIDS)
for NUM_CENTROIDS in NUM_CENTROIDS_ARR:
    C = init_centroids(X,Y,NUM_CENTROIDS)
    #C = init_centroids_rand(X,NUM_CENTROIDS)
    c_x = X[C]
    c_y = Y[C]
    c_entropy = []

    for j in range(n_epochs):
        #print(j)

        distances =  dist(X,Y,c_x,c_y)

        #assigns points to clusters
        assignments = np.argmin(distances,axis = 1) 
        #print(assignments)

        #update centroids
        for i in range(NUM_CENTROIDS):
            #gets all points in a cluster and takes their avg
            cluster_idx = np.where(assignments == i) #get index of all points in cluster
            
            avg_x = np.sum(X[cluster_idx])/len(cluster_idx[0]) #get avg of x in cluster
            avg_y = np.sum(Y[cluster_idx])/len(cluster_idx[0]) #get avg of y in cluster

            #convert to array
            avg_x = np.array([avg_x])
            avg_y = np.array([avg_y])

            #find closest point to (avg_x,avg_y)
            dist_new_c = dist(avg_x,avg_y,X,Y)
            c_x[i] = X[np.argmin(dist_new_c)]
            c_y[i] = Y[np.argmin(dist_new_c)]

        min_dist = []
        #cakculate
        for i in range(len(data)):
            min_dist.append(distances[i][assignments[i]])
        
        mean_entropy = np.sum(min_dist)/len(min_dist)
        #print(mean_entropy)
        c_entropy.append(mean_entropy)

    #plots centroids
    plt.scatter(X,Y,c = assignments)
    plt.scatter(c_x,c_y,c='r',marker = 'x')
    for i in range(0,NUM_CENTROIDS):
        file = open("results.txt",'a')
        file.write('\n')
        file.write("-------------------------------------------------")
        file.write('\n')
        file.write("Cluster " + str(i))
        file.write('\n')



        cluster_idx = np.where(assignments == i)
        file.write('\n'.join(names[cluster_idx]) + '\n')
        file.write('\n')
        file.write("-------------------------------------------------")
        file.write('\n')
        file.close()

    #graphs cluster_entropy vs iterations
    #plt.ylabel("Mean Cluster Entropy")
    #plt.xlabel("Num Iterations")
    #plt.title(str(sys.argv[2]))
    #plt.plot(range(n_epochs),c_entropy,label = 'Num_centroid = ' + str(NUM_CENTROIDS))



plt.legend(NUM_CENTROIDS_ARR,title = 'Num Centroids',fontsize ='small')


plt.show()