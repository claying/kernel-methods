import numpy as np 

def load_data(filepath='data/', split='train'):
	if split == 'train':
		X = np.genfromtxt(filepath+'Xtr.csv', delimiter=',', dtype='float32')
		X = X[:, :-1]
		y = np.genfromtxt(filepath+'Ytr.csv', delimiter=',', skip_header=1, dtype=int)
		return X, y[:,-1]
	elif split == 'test':
		X = np.genfromtxt(filepath+'Xte.csv', delimiter=',', dtype='float32')
		return X[:, :-1]

def imshow(vec):
	"""visualize image of given vec: 1 x 3072"""
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
	import matplotlib.pyplot as plt
	plt.imshow(img)
	plt.axis("off")
	plt.show()

def distribution_y(filepath='data/'):
	y = np.genfromtxt(filepath+'Ytr.csv', delimiter=',', skip_header=1, dtype=int)[:, -1]
	import matplotlib.pyplot as plt
	print y 
	vmin = y.min()
	vmax = y.max()
	plt.hist(y, bins=np.arange(vmin-0.5, vmax+1.5))
	plt.show()

# distribution_y()