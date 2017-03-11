import numpy as np 
from filters import conv2_gaussian
from utils_fast import compute_kernel
from utils import extract_patches_2d, centering_patch, normalize_row, spherical_kmeans, shape_patch
import pickle

LAYERTYPES = ['gradient', 'patch', 'shape']

class CKN(object):
	"""docstring for CKN"""
	def __init__(self, patch_sizes, subsamplings, map_dims, layer_type):
		self.patch_sizes = patch_sizes
		self.subsamplings = subsamplings
		self.map_dims = map_dims
		if layer_type in LAYERTYPES:
			self.layer_type = layer_type
		else:
			raise ValueError("unknown layer type")
		self.n_layers = len(patch_sizes)

	def train(self, images):
		"""
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
		elif self.order == 0 and self.layer_type == 'shape':
			n = input_maps.shape[0]
			for i in range(n):
				psi = input_maps[i]
				patch = extract_patches_2d(psi, self.patch_size)
				
				patch = shape_patch(patch)
				if i == 0:
					n_patches_per_image = np.minimum(2*self.n_patches_per_image, patch.shape[0])
					X = np.zeros((n*n_patches_per_image, patch.shape[1]))
				# discard patches with no variance
				idx = np.any(patch != patch[:,[0]], axis = 1)
				if n_patches_per_image <= idx.sum():
					patch = patch[idx]
				np.random.shuffle(patch)
				X[i*n_patches_per_image:(i+1)*n_patches_per_image] = patch[:n_patches_per_image]
			np.random.shuffle(X)
			X = normalize_row(X, norm='l2')
			Z = spherical_kmeans(X, self.map_dim)[0]
			self.Z = Z  
			self.sigma2 = 0.25
			lin_tran = gaussian_func(Z.dot(Z.T), self.sigma2)
			# svd decomposition to compute lin_tran^(-0.5)
			w, v = np.linalg.eigh(lin_tran)
			w = (w + 1e-8)**(-0.5)
			lin_tran = v.dot(np.diag(w).dot(v.T))
			self.lin_tran = lin_tran
		else:
			n = input_maps.shape[0]
			for i in range(n):
				psi = input_maps[i]
				patch = extract_patches_2d(psi, self.patch_size)
				if self.centering:
					patch = centering_patch(patch)
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
			np.random.shuffle(X)
			X = normalize_row(X, norm='l2')
			Z = spherical_kmeans(X, self.map_dim)[0]
			self.Z = Z  
			self.sigma2 = 0.25
			lin_tran = gaussian_func(Z.dot(Z.T), self.sigma2)
			# svd decomposition to compute lin_tran^(-0.5)
			w, v = np.linalg.eigh(lin_tran)
			w = (w + 1e-8)**(-0.5)
			lin_tran = v.dot(np.diag(w).dot(v.T))
			self.lin_tran = lin_tran

	def forward(self, input_map):
		if self.order == 0 and self.layer_type == 'gradient':
			n_channels = input_map.shape[-1]
			output_map = np.zeros(input_map.shape[:-1] + (self.map_dim*n_channels, ))
			theta = np.linspace(0, 2.0*np.pi, self.map_dim+1)[:-1]
			delta_theta = 2.0*np.pi/self.map_dim
			sigma2 = np.square(1 - np.cos(delta_theta))+np.square(np.sin(delta_theta))
			self.sigma2 = sigma2
			for k in range(n_channels):
				input_map_k = input_map[:,:,k]
				dx, dy = np.gradient(input_map_k)
				rho = np.sqrt(np.square(dx)+np.square(dy))
				idx = rho > 0
				dx[idx] = dx[idx]/rho[idx]
				dy[idx] = dy[idx]/rho[idx]
				# sample theta between [0, 2*pi]
				for i in range(self.map_dim):
					output_map[:,:,i+self.map_dim*k] = rho * gaussian_func(dx*np.cos(theta[i])+dy*np.sin(theta[i]), sigma2)
		elif self.order == 0 and self.layer_type == 'shape':
			# we suppose that patch_size is odd
			n_channels = input_map.shape[-1]
			patch = extract_patches_2d(input_map, self.patch_size)
			size = input_map.shape[0] - self.patch_size + 1
			patch = shape_patch(patch)
			patch, rho = normalize_row(patch, norm='l2', return_norm=True)

			output_map = gaussian_func(patch.dot(self.Z.T), self.sigma2)
			output_map = np.diag(rho).dot(output_map.dot(self.lin_tran))
			output_map = output_map.reshape((size, size, -1))
		else:
			patch = extract_patches_2d(input_map, self.patch_size)
			size = input_map.shape[0] - self.patch_size + 1
			# centering
			if self.centering:
				patch = centering_patch(patch)

			patch = patch.reshape((patch.shape[0], -1))

			patch, rho = normalize_row(patch, norm='l2', return_norm=True)
			output_map = gaussian_func(patch.dot(self.Z.T), self.sigma2)
			output_map = np.diag(rho).dot(output_map.dot(self.lin_tran))
			output_map = output_map.reshape((size, size, -1))
		# gaussian smoothing
		for i in range(output_map.shape[-1]):
			output_map[:,:,i] = conv2_gaussian(output_map[:,:,i], 2*self.subsampling+1, self.subsampling/np.sqrt(2))
		output_map = output_map[(self.subsampling-1)//2::self.subsampling,(self.subsampling-1)//2::self.subsampling, :]
		return output_map

def gaussian_func(x, sigma2):
	return np.exp(1/sigma2*(x-1))

