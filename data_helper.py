import numpy as np 

def load_data(filepath='data/', split='train', reshape_images=False):
	"""
	if reshape_images
	  X: images array dim: n x sx x sx x n_channels
	else
	  X: images array dim: n x d
	y: 1d array
	"""
	if split == 'train':
		X = np.genfromtxt(filepath+'Xtr.csv', delimiter=',', dtype='float64')
		X = X[:, :-1]
		if reshape_images:
			X = reshape(X)
		y = np.genfromtxt(filepath+'Ytr.csv', delimiter=',', skip_header=1, dtype=int)
		return X, y[:,-1]
	elif split == 'test':
		X = np.genfromtxt(filepath+'Xte.csv', delimiter=',', dtype='float64')
		X = X[:, :-1]
		if reshape_images:
			X = reshape(X)
		return X 

def save_pred(y_pred, outpath='output/'):
	ids = np.arange(1, 2001, dtype=int)
	arr = np.hstack((ids.reshape(-1,1), y_pred.reshape(-1,1)))
	arr = arr.astype(int)
	np.savetxt(outpath+'Yte.csv', arr, header='Id,Prediction', fmt='%d,%d', delimiter=',', comments='')

def reshape(images, sx=32, n_channels=3):
	return images.reshape((-1, n_channels, sx, sx)).swapaxes(1, 2).swapaxes(2,3)

def imshow(img):
	"""visualize image of given vec: 1 x 3072"""
	if img.ndim < 3:
		print ('ok')
		img = img.reshape((32,32,3), order='F')
		img = img.swapaxes(0, 1)
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


# y = np.random.randint(1, 2001, size=2000)
# print y.dtype
# save_pred(y) 

# X,y= load_data(reshape_images=1)
# img = X[1][:,:,1]
# dx, dy = np.gradient(img)
# print dx 
# import matplotlib.pyplot as plt
# plt.imshow(dx, cmap=plt.cm.binary)
# plt.axis("off")
# plt.show()
# print np.mean(x)
# x = reshape(x)
# print x.shape 
# print y.shape
# imshow(x[1])
# imshow(x[10])
# distribution_y()