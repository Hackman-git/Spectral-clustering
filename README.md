# Spectral-clustering
Apply the Spectral Clustering method to the FourCircle.csv.  Your input fields are x and y. Wherever needed, specify random_state = 60616 in calling the KMeans function.
g)	(5 points) Plot y on the vertical axis versus x on the horizontal axis.  How many clusters are there based on your visual inspection?

h)	(5 points) Apply the K-mean algorithm directly using your number of clusters that you think in (a). Regenerate the scatterplot using the K-mean cluster identifiers to control the color scheme. Please comment on this K-mean result.

i)	(10 points) Apply the nearest neighbor algorithm using the Euclidean distance.  We will consider the number of neighbors from 1 to 15.  What is the smallest number of neighbors that we should use to discover the clusters correctly?  Remember that we may need to try a couple of values first and use the eigenvalue plot to validate our choice.

j)	(5 points) Using your choice of the number of neighbors in (c), calculate the Adjacency matrix, the Degree matrix, and finally the Laplacian matrix. How many eigenvalues do you determine are practically zero?  Please display their calculated values in scientific notation.

k)	(10 points) Apply the K-mean algorithm on the eigenvectors that correspond to your “practically” zero eigenvalues.  The number of clusters is the number of your “practically” zero eigenvalues. Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme.
