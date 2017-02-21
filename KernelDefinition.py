# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:41:16 2017

@author: Admin
"""
import scipy
import numpy as np

def GramGaussianKernel(X,gamma) :
    from scipy.spatial.distance import pdist, squareform
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = scipy.exp(-pairwise_sq_dists *gamma)
    return K
    
def GramGaussianDistance(gammaP) :
    """Computes pairwise distance over all pixels normalized to [0,1["""
    X=[]
    for i in range(0,32) :
        X.append(np.array([[i]*32,range(0,32)])/32)
    return GramGaussianKernel(np.reshape(np.concatenate(np.asarray(X)).T,[-1,2]),gammaP)

def GradMagnitudeOrientation(img) :
    """Computes orientation and gradient matrix over whole image (looping on the edges)"""
    GradOpY=scipy.linalg.circulant([-1]+[0]*30+[1])
    GradImgY=np.sqrt(np.square(np.dot(GradOpY,img[:,:,0]))+np.square(np.dot(GradOpY,img[:,:,1]))+np.square(np.dot(GradOpY,img[:,:,2])))
    GradImgX=np.sqrt(np.square(np.dot(GradOpY,img[:,:,0].T))+np.square(np.dot(GradOpY,img[:,:,1].T))+np.square(np.dot(GradOpY,img[:,:,2].T)))
    GradMag=np.sqrt(np.square(GradImgY)+np.square(GradImgX))
    GradOr=np.arctan(GradImgY/(GradImgX+1E-15))
    return GradMag, GradOr 
    
def ValueGradKernel(xlist1,ylist1,xlist2,ylist2,img1,img2,gammaO,gammaP,epsG) :
    """Computes gradient match kernel between pacthes xlist1*ylist1 of img1 and xlist2*ylist2 of img2"""
    GradMag1, GradOr1=GradMagnitudeOrientation(img1)
    GradMag2, GradOr2=GradMagnitudeOrientation(img2)
    subimg1GradMag=GradMag1[np.ix_(xlist1,ylist1)].flatten()
    subimg2GradMag=GradMag2[np.ix_(xlist2,ylist2)].flatten()
    subimg1GradOr=GradOr1[np.ix_(xlist1,ylist1)].flatten()
    subimg2GradOr=GradOr2[np.ix_(xlist2,ylist2)].flatten() 
    normalize1=np.sqrt(np.sum(np.square(subimg1GradMag))+epsG)
    normalize2=np.sqrt(np.sum(np.square(subimg2GradMag))+epsG)   
    XP=[]
    XO=[]
    """Missing the positional kernel weighting, elegant way to compute it on the patches rather than extracting from the whole GramGaussianPMat ?"""
    for i in range(len(subimg1GradOr)) :
        XO.append(np.square(subimg2GradOr-subimg1GradOr[i]))                   
    KgradPQ = np.dot(subimg1GradMag,np.dot(np.asarray(XO),subimg2GradMag.T))/(normalize1*normalize2)
    return KgradPQ


vec=X[0,:]
img = np.zeros((32,32,3), dtype='float32')
img[:,:,0] = vec[0:1024].reshape((32,32))
img[:,:,1] = vec[1024:2048].reshape((32,32))
img[:,:,2] = vec[2048:].reshape((32,32))
for i in range(3):
	minval = img[:, :, i].min()
	maxval = img[:, :, i].max()
	if minval != maxval:
		img[:,:,i] -= minval
		img[:,:,i] *= (1.0/(maxval-minval))
xlist1=np.array(range(0,3))
xlist2=np.array(range(0,3))
ylist1=np.array(range(0,3))
ylist2=np.array(range(3,6))   
img1=img2=img
gammaP=.5
gammaO=.5
epsG=1E-10
GramGaussianPMat=GramGaussianDistance(gammaP)  
KgradPQ=ValueGradKernel(xlist1,ylist1,xlist2,ylist2,img1,img2,gammaO,gammaP,epsG)