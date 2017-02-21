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
    XP=[]
    for i in range(0,32) :
        XP.append(np.array([[i]*32,range(0,32)])/32)
    return GramGaussianKernel(np.reshape(np.concatenate(np.asarray(XP)).T,[-1,2]),gammaP)

def GradMagnitudeOrientation(img) :
    """Computes orientation and gradient matrix over whole image (looping on the edges)"""
    GradOpY=scipy.linalg.circulant([-1]+[0]*30+[1])
    GradImgY=np.sqrt(np.square(np.dot(GradOpY,img[:,:,0]))+np.square(np.dot(GradOpY,img[:,:,1]))+np.square(np.dot(GradOpY,img[:,:,2])))
    GradImgX=np.sqrt(np.square(np.dot(GradOpY,img[:,:,0].T))+np.square(np.dot(GradOpY,img[:,:,1].T))+np.square(np.dot(GradOpY,img[:,:,2].T)))
    GradMag=np.sqrt(np.square(GradImgY)+np.square(GradImgX))
    GradOr=np.arctan(GradImgY/(GradImgX+1E-15))
    return GradMag, GradOr 

"We take the classical convention for images x horizontally and y vertically (which is the transpose for matrices)"
"test part"
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
gammaC=.5
epsG=1E-10
GramGaussianPMat=GramGaussianDistance(gammaP)  
    
def ValueGradKernel(xlist1,ylist1,xlist2,ylist2,img1,img2,gammaO,gammaP,epsG) :
    """Computes gradient match kernel between pacthes xlist1*ylist1 of img1 and xlist2*ylist2 of img2"""
    patch1=np.ix_(ylist1,xlist1)
    patch1IndMat=np.matlib.repmat(32*ylist1, len(xlist1), 1).T+np.matlib.repmat(xlist1, len(ylist1), 1)
    patch2=np.ix_(ylist2,xlist2)
    patch2IndMat=np.matlib.repmat(32*ylist2, len(xlist2), 1).T+np.matlib.repmat(xlist2, len(ylist2), 1)   
        
    GradMag1, GradOr1=GradMagnitudeOrientation(img1)
    GradMag2, GradOr2=GradMagnitudeOrientation(img2)
    subimg1GradMag=GradMag1[patch1].flatten()
    subimg2GradMag=GradMag2[patch2].flatten()
    subimg1GradOr=GradOr1[patch1].flatten()
    subimg2GradOr=GradOr2[patch2].flatten() 
    normalize1=np.sqrt(np.sum(np.square(subimg1GradMag))+epsG)
    normalize2=np.sqrt(np.sum(np.square(subimg2GradMag))+epsG)   
    """Is there an elegant way to compute it on the patches rather than extracting from the whole GramGaussianPMat ?"""
    XO=[]    
    for i in range(len(subimg1GradOr)) :
        XO.append(-np.square(subimg2GradOr-subimg1GradOr[i])*gammaO)
    XP=GramGaussianPMat[np.ix_(patch1IndMat.flatten(),patch2IndMat.flatten())]                   
    KgradPQ = np.dot(subimg1GradMag,np.dot(scipy.exp(np.asarray(XO))*XP,subimg2GradMag.T))/(normalize1*normalize2)
    return KgradPQ

def ValueColorKernel(xlist1,ylist1,xlist2,ylist2,img1,img2,gammaC,gammaP,*epsG) :
    """Computes gradient match kernel between pacthes xlist1*ylist1 of img1 and xlist2*ylist2 of img2"""
    patch1=np.ix_(ylist1,xlist1)
    patch1IndMat=np.matlib.repmat(32*ylist1, len(xlist1), 1).T+np.matlib.repmat(xlist1, len(ylist1), 1)
    patch2=np.ix_(ylist2,xlist2)
    patch2IndMat=np.matlib.repmat(32*ylist2, len(xlist2), 1).T+np.matlib.repmat(xlist2, len(ylist2), 1)   
    
    subimg1R=img[:,:,0][patch1].flatten()
    subimg2R=img[:,:,0][patch2].flatten()
    subimg1B=img[:,:,1][patch1].flatten()
    subimg2B=img[:,:,1][patch2].flatten()   
    subimg1G=img[:,:,2][patch1].flatten()
    subimg2G=img[:,:,2][patch2].flatten()       
    XCR=[]
    XCB=[]
    XCG=[]    
    for i in range(len(subimg1R)) :
        XCR.append(-np.square(subimg2R-subimg1R[i])*gammaC)
        XCB.append(-np.square(subimg2B-subimg1B[i])*gammaC)
        XCG.append(-np.square(subimg2G-subimg1G[i])*gammaC)
    XP=GramGaussianPMat[np.ix_(patch1IndMat.flatten(),patch2IndMat.flatten())]                   
    KcolPQ = ((scipy.exp(np.asarray(XCR))+scipy.exp(np.asarray(XCG))+scipy.exp(np.asarray(XCB)))*XP).sum()
    return KcolPQ
    
KgradPQ=ValueGradKernel(xlist1,ylist1,xlist2,ylist2,img1,img2,gammaO,gammaP,epsG)
KcolPQ=ValueColorKernel(xlist1,ylist1,xlist2,ylist2,img1,img2,gammaC,gammaP)