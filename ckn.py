import numpy as np 
from utils import extract_patches_2d
from scipy.optimize import fmin_l_bfgs_b
from filters import conv2_gaussian
from scipy.spatial.distance import cdist
from utils_fast import compute_kernel
from utils import normalize_row
import pickle

class CKN(object):
	"""CKN model"""
	def __init__(self, patch_sizes, subsamplings, map_dims, layer_type):
		self.patch_sizes = patch_sizes
		self.subsamplings = subsamplings
		self.map_dims = map_dims
		self.layer_type = layer_type
		self.n_layers = len(patch_sizes)

	def train(self, images):
		"""
		train CKN model for given images
		images: n x sx x sx x channels
		"""
		self.layers = []
		psis = images
		n = images.shape[0]
		for i in range(self.n_layers):
			centering = (i==0) and (self.layer_type=='patch')
			layer = CKNLayer(i, centering, self.layer_type, self.patch_sizes[i], self.map_dims[i], self.subsamplings[i])
			layer.train(psis)
			self.layers.append(layer)
			if i < self.n_layers - 1:
				for j in range(n):
					psi = layer.forward(psis[j])
					if j == 0:
						out_psis = np.zeros((n,)+psi.shape)
					out_psis[j] = psi 
				psis = out_psis

	def forward(self, image, layer=None):
		"""output map of the i-th layer
		layer: int < n_layers
		image: sx x sx x channels
		"""
		if layer is None:
			layer = self.n_layers
		psi = image
		for i in range(layer):
			psi = self.layers[i].forward(psi)
		return psi 

	def out_maps(self, images):
		n = images.shape[0]
		for i in range(n):
			output_map = self.forward(images[i])
			output_map = output_map.flatten()
			if i == 0:
				res =  np.zeros((n, output_map.shape[0]))
			res[i] = output_map
		return res

def save(model, filename):
	with open(filename, 'wb') as f:
		pickle.dump(model, f)

def load(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)
	
class CKNLayer(object):
	"""one layer for CKN model"""
	def __init__(self, order, centering, layer_type, patch_size, map_dim, subsampling):
		self.order = order
		self.centering = centering
		self.layer_type = layer_type
		self.patch_size = patch_size
		self.map_dim = map_dim
		self.subsampling = subsampling
		self.n_patches_per_image = 10

	def train(self, input_maps):
		if self.order == 0 and self.layer_type == 'gradient':
			# no parameter training for gradient map
			return 
		else:
			n = input_maps.shape[0]
			for i in range(n):
				psi = input_maps[i]
				patch = extract_patches_2d(psi, self.patch_size)
				# print patch.shape
				patch = patch.reshape((patch.shape[0], -1))
				if i == 0:
					n_patches_per_image = np.minimum(2*self.n_patches_per_image, patch.shape[0])
					X = np.zeros((n*n_patches_per_image, patch.shape[1]))
				# discard patches with no variance
				idx = np.any(patch != patch[:,[0]], axis = 1)
				if n_patches_per_image <= idx.sum():
					patch = patch[idx]
				np.random.shuffle(patch)
				X[i*n_patches_per_image:(i+1)*n_patches_per_image] = patch[:n_patches_per_image]
			size = X.shape[0]//2
			np.random.shuffle(X)
			X = normalize_row(X, norm='l2')
			w, eta, sigma2 = nystrom_gaussian(X[:size], X[size:], self.map_dim, sigma2=None)
			self.w = w 
			self.eta = eta
			self.sigma2 = sigma2

	def forward(self, input_map):
		if self.order == 0 and self.layer_type == 'gradient':
			if input_map.ndim == 3:
				input_map = input_map[:,:,1]
			dx, dy = np.gradient(input_map)
			rho = np.sqrt(np.square(dx)+np.square(dy))
			idx = rho > 0
			dx[idx] = dx[idx]/rho[idx]
			dy[idx] = dy[idx]/rho[idx]
			# sample theta between [0, 2*pi]
			theta = np.linspace(0, 2.0*np.pi, self.map_dim+1)[:-1]
			delta_theta = 2.0*np.pi/self.map_dim
			sigma2 = np.square(1 - np.cos(delta_theta))+np.square(np.sin(delta_theta))
			self.sigma2 = sigma2
			output_map = np.zeros(input_map.shape + (self.map_dim, ))
			for i in range(self.map_dim):
				output_map[:,:,i] = rho * np.exp(-1.0/sigma2*(np.square(dx-np.cos(theta[i]))+np.square(dy-np.sin(theta[i]))))
		else:
			patch = extract_patches_2d(input_map, self.patch_size)
			size = input_map.shape[0] - self.patch_size + 1
			# TODO: centering for patch map

			patch = patch.reshape((patch.shape[0], -1))
			patch, rho = normalize_row(patch, norm='l2', return_norm=True)
			output_map = eval_psi(patch, self.w, self.sigma2)
			output_map = np.diag(rho).dot(output_map.dot(np.diag(np.sqrt(self.eta))))
			output_map = output_map.reshape((size, size, -1))
		# gaussian smoothing
		for i in range(output_map.shape[-1]):
			output_map[:,:,i] = conv2_gaussian(output_map[:,:,i], 2*self.subsampling+1, self.subsampling/np.sqrt(2))
		output_map = output_map[(self.subsampling-1)//2::self.subsampling,(self.subsampling-1)//2::self.subsampling, :]
		return output_map


def gaussian_kernel(X, Y, sigma2):
	aux = np.sum(np.square(X-Y), axis=1)
	if sigma2 is None:
		sigma2 = np.percentile(aux, 10, interpolation='midpoint')
	return np.exp(-aux/(2.0*sigma2)), sigma2

def eval_psi(X, w, sigma2):
	"""
	X : n x m
	w : dim x m
	output: n x dim
	"""
	n = X.shape[0]
	dim = w.shape[0]
	psi = cdist(X, w)
	psi = np.square(psi)
	return np.exp(-psi/sigma2)

# def eval_psi2(X, w, sigma2):
# 	"""
# 	X : n x m
# 	w : dim x m
# 	output: n x dim
# 	"""
# 	n = X.shape[0]
# 	dim = w.shape[0]
# 	psi = compute_kernel(X, w, sigma2/2.0, kernel='gaussian')
# 	return psi 

def nystrom_gaussian(X, Y, dim, sigma2=None):
	"""nystrom approximation for gaussian kernel"""
	# compute and normalize gaussian kernel
	K, sigma2 = gaussian_kernel(X, Y, sigma2)
	print sigma2
	sumK = np.sqrt(np.sum(np.square(K)))
	K = K/sumK
	# print K
	n = X.shape[0]
	sumXY = X + Y 
	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=dim, random_state=0).fit(X)

	# w = np.ones((dim, X.shape[1]))
	w = kmeans.cluster_centers_
	eta = np.zeros(dim)
	x0 = np.hstack((w, eta.reshape(-1,1)))
	print X.shape
	# print Y.shape
	print w.shape
	# print eta.shape
	def func(x, compute_grad=True):
		x = x.reshape((dim,-1))
		# print x.shape
		w = x[:,:-1]
		eta = x[:,-1]
		psiX = eval_psi(X, w, sigma2)
		psiY = eval_psi(Y, w, sigma2)
		psiXY = psiX * psiY
		R = K - psiXY.dot(eta)
		obj = np.mean(np.square(R))
		# print psiY
		if not compute_grad:
			return obj
		# gradient
		aux = psiXY.T.dot(np.diag(R)) # dim x n
		sum_aux = np.sum(aux, axis=1)
		grad_eta = -2.0/n*sum_aux # dim x 1
		grad_w = -4.0/(n*sigma2)*np.diag(eta).dot(aux.dot(sumXY) - 2*np.diag(sum_aux).dot(w))
		grad = np.hstack((grad_w, grad_eta.reshape(-1, 1)))
		grad = grad.flatten()
		# print grad
		return obj, grad
	print("initial objective function: %f" % func(x0,False))
	# u = np.ones(x0.shape)*np.inf
	l = np.zeros(x0.shape)
	l[:,:-1] = -np.inf 
	l = l.flatten()
	x0 = x0.flatten()
	u = np.ones(x0.shape)*np.inf

	x, obj, information = fmin_l_bfgs_b(func, x0, iprint=-1, maxiter=5000, pgtol=1e-8, bounds=zip(l, u), m=1000)
	print("final objective function: %f" % obj)
	print information
	x = x.reshape((dim, -1))
	w = x[:,:-1]
	eta = x[:,-1]
	print eta
	print w 
	return w, eta, sigma2
