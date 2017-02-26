import numpy as np

def kMeans(X, K, maxIters = 100):
	## inputs:
	#K: number of cluster
	#X: input data
	#maxIters : maximale iteration
	##outputs:
	#c: labels of data
	#centroids: center of clustering
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
   
	
    return np.array(centroids) , C
