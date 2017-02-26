import numpy as np 
from utils_fast import conv2

def gaussian_filter_1d(shape, sigma):
	m = (shape - 1.)/2.
	filt = np.arange(-m,m+1)
	filt = np.exp(-np.square(filt)/(2.*sigma**2))
	return filt/np.sum(filt)

def conv2_gaussian(in1, shape, sigma=0.5):
	"""convolve given image with a gaussian filter with given shape
	sigma: sigma for gaussian
	"""
	g_filter = gaussian_filter_1d(shape, sigma)
	g_filter = np.outer(g_filter, g_filter)
	return conv2(in1, g_filter)

if __name__ == '__main__':
	x = np.ones((32,32))
	print(conv2_gaussian(x, 9, np.sqrt(2)))