import numpy as np
from scipy.spatial import distance
from scipy.linalg import norm


def kMeans(X, K, maxIters = 100,tol = 5):
	## inputs:
	#K: number of cluster
	#X: input data
	#maxIters : maximale iteration
        #tol: the tolerance for the total distance of point to the center

	##outputs:
	#c: labels of data
	#centroids: center of clustering

    #random initialization
    centroids =np.array(X[np.random.choice(np.arange(len(X)), K), :])
   
    tt_0 = 0
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([distance.cdist([X[j,:]],centroids,'euclidean')]) for j in range(len(X))])

        # Move centroids step
        centroids = np.array([X[C == k].mean(axis = 0) for k in range(K)])
        
	#calculate the total distance of each point to center
	tt_1 = norm(X-centroids[C,:],axis=1).mean(axis =0)
	print tt_1

        # when tolerance or converge arrives, stop 
        if tt_1<tol or tt_1-tt_0==0:
		print "tolerance  or converge is achieve "
        	break
        tt_0=tt_1
    
    return np.array(centroids) , C

