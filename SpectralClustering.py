'''
Name: Olugbenga Abdulai
CWID: A20447331
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.neighbors as nbrs
import math

'''
3(a)
'''
# reading data
data = pd.read_csv(r"C:\Users\abdul\Desktop\CS 584\HW\HW 2\FourCircle.csv")

# scatterplot
plt.figure(figsize=[8,8])
sns.scatterplot(x=data['x'], y=data['y'], data=data)
plt.show()

'''
There are four clusters
'''

'''
3(b)
'''
train_data = data[['x', 'y']]
kmeans = cluster.KMeans(n_clusters=4, random_state=60616).fit(train_data)
data['kmean_cluster'] = kmeans.labels_

# scatter plot
plt.figure(figsize=[8,8])
sns.scatterplot(x=data['x'], y=data['y'], hue=data.kmean_cluster)
plt.show()

# observations for each cluster
for i in range(4):
    print("\ncluster label ", i)
    print(data.loc[data.kmean_cluster == i])

'''
The kmeans algorithm separates cluster 0 and cluster 1 along the x-axis. i.e. cluster 0 has x-values
ranging roughly from 0-10 while cluster 1 has x-values ranging from -10 to 0. The algorithm separates cluster 2
and cluster 3 along the y-axis with y-values for cluster 2 ranging roughly from 3-10 and y-values for cluster 3 
ranging from -3 to -10. The result is a cluster chart resembling a pie chart split in four segments.
'''

'''
3(c)
'''
'''
This function prints the adjacency matrix, degree matrix, laplacian matrix and 
returns the eigenvalues and eigenvectors from the laplacian matrix of spectral
clustering analysis.Takes the number of neighbors as input
'''
def eigenval_eigenvec(n):
    knn = nbrs.NearestNeighbors(n_neighbors=n, algorithm="brute", metric='euclidean')
    knn_fit = knn.fit(train_data)
    d, i = knn_fit.kneighbors(train_data)

    # distances among observations
    dist = nbrs.DistanceMetric.get_metric('euclidean')
    distances = dist.pairwise(train_data)

    # adjacency matrix
    n_obs = data.shape[0]
    adj = np.zeros((n_obs, n_obs))
    for r in range(n_obs):
        for j in i[r]:
            adj[r, j] = math.exp(-(distances[r][j]) ** 2)

    # making the adjacency matrix symmetric
    adj = 0.5 * (adj + adj.transpose())
    print("\nadjacency matrix:\n", adj)

    # Degree matrix
    degree = np.zeros((n_obs, n_obs))
    for j in range(n_obs):
        s = 0
        for k in range(n_obs):
            s += adj[j, k]
        degree[j, j] = s

    print("\ndegree matrix:\n", degree)

    # Laplacian matrix
    laplacian = degree - adj
    print("\nlaplacian matrix:\n", laplacian)

    # eigenvalues and eigenvectors of laplacian matrix
    from numpy import linalg as lin
    evals, evecs = lin.eigh(laplacian)

    print("\nfirst seven eigenvalues with {} nearest neighbors: \n".format(n), evals[:8])
    return (evals, evecs)

# trying various values for number of neighbors
for j in range(1,16):
    print(eigenval_eigenvec(j))

'''
From the eigenvalues analysis and our visual inspection that we require 4 clusters, we can see that
a minimum of 6 nearest neighbors is needed to correctly achieve 4 clusters. Let us confirm this with
the eigenvalue plot
'''
def plot_eigen_values(evals):
    # determining number of clusters with 7 smallest eigenvalues
    seq = np.arange(1,8,1)
    plt.figure(figsize=[8,8])
    sns.scatterplot(x=seq, y=evals[0:7,])
    plt.xticks(seq)
    plt.grid('both')
    plt.show()

evals, evecs = eigenval_eigenvec(6)
plot_eigen_values(evals)

'''
3(d)
'''
# with 6 nearest neighbors
evals, evec = eigenval_eigenvec(6)

print('\neigenvalues for 6 nearest neighbors: ', evals[:4])
'''
There are four eigenvalues that are practically zero
In scientific notation, 
[-1.24369 x 10^-15, 1.4879 x 10^-16, 4.4804 x 10^-16, 2.9175 x 10^-15]
'''

'''
3(e)
'''
# determining the 'practical' number of neighbors
# plotting the first 20 eigenvals
seq = np.arange(1,21,1)
plt.figure(figsize=[8,8])
sns.scatterplot(x=seq, y=evals[0:20,])
plt.xticks(seq)
plt.grid('both')
plt.show()

'''
The first 'jump' occurs at the 10th eigenvalue so we choose
10 as our optimal number of nearest neighbors
'''
evals, evec = eigenval_eigenvec(10)
z = evecs[:, [0,1,2,3]]

# 4-cluster k-mean on first four eigenvectors
kmeans_spectral = cluster.KMeans(n_clusters=4, random_state=60616).fit(z)
data['spectral_cluster'] = kmeans_spectral.labels_

# scatter plot
plt.figure(figsize=[8,8])
sns.scatterplot(x=data['x'], y=data['y'], hue=data['spectral_cluster'])
plt.grid()
plt.show()