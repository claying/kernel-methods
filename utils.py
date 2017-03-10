import numbers
from numpy.lib.stride_tricks import as_strided
import numpy as np 

def check_random_state(seed):
	"""Turn seed into a np.random.RandomState instance
	If seed is None, return the RandomState singleton used by np.random.
	If seed is an int, return a new RandomState instance seeded with seed.
	If seed is already a RandomState instance, return it.
	Otherwise raise ValueError.
	"""
	if seed is None or seed is np.random:
		return np.random.mtrand._rand
	if isinstance(seed, (numbers.Integral, np.integer)):
		return np.random.RandomState(seed)
	if isinstance(seed, np.random.RandomState):
		return seed
	raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
					 ' instance' % seed)

def extract_patches_2d(image, patch_size, step=1):
	"""image : nx x ny x n_channels
	output: n_patches x patch_size x patch_size x n_channels
	"""
	n_channels = image.shape[-1]
	patches = extract_patches(image, (patch_size, patch_size, n_channels), step)
	return patches.reshape(-1, patch_size, patch_size, n_channels)

def extract_patches(arr, patch_shape, extraction_step):
	if isinstance(extraction_step, int):
		extraction_step = tuple([extraction_step] * arr.ndim)
	patch_strides = arr.strides
	slices = [slice(None, None, st) for st in extraction_step]
	indexing_strides = arr[slices].strides

	patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
						   np.array(extraction_step)) + 1

	shape = tuple(list(patch_indices_shape) + list(patch_shape))
	strides = tuple(list(indexing_strides) + list(patch_strides))

	patches = as_strided(arr, shape=shape, strides=strides)
	return patches

def centering_patch(patch):
	return patch - np.mean(patch, axis=(1, 2), keepdims=1)

def shape_patch(patch, threshold=1e-6):
	"""patch: n_patches x size x size x n_channels
	output: n_patches x (size^2-1)n_channels
	"""
	n_channels = patch.shape[-1]
	patch = patch.reshape((patch.shape[0], -1, patch.shape[-1]))
	center = patch.shape[1]//2
	patch = patch - patch[:,[center], :]
	patch = np.delete(patch, center, 1)  # remove center
	patch = patch.reshape((patch.shape[0],-1))
	# rho = np.linalg.norm(patch, axis=1)
	# rho[rho<threshold] = 0
	# # binarize the difference
	# cond = patch > 0
	# non_cond = patch <= 0 
	# patch[cond] = 1.0
	# patch[non_cond] = 0.0
	return patch


def normalize_row(X, norm='l2', threshold=1e-5, return_norm=False):
	if norm == 'l2':
		norms = np.linalg.norm(X, axis=1)
		norms[norms<threshold] = threshold
		X /= norms[:, np.newaxis]
	if return_norm:
		return X, norms
	return X 

def kmeans(X, k, max_iter=1000, init=None):
	"""normal kmeans"""
	n_samples, n_features = X.shape
	if init is None:
		perm =  np.random.choice(np.arange(n_samples), k)
		centroids = X[perm]

	norm2 = np.linalg.norm(X, axis=1, keepdims=1)**2
	prev_obj = np.inf 

	for n_iter in range(max_iter):
		dist2 = np.linalg.norm(centroids, axis=1)**2 - 2.0*np.dot(X, centroids.T) + norm2
		assign = np.argmin(dist2, axis=1)
		obj = dist2[np.unravel_index(assign, dist2.shape)].mean()
		# print dist.shape
		if (n_iter+1)%10 == 0:
			print("kmeans iter %d, objective: %f"%(n_iter+1, obj))

		for j in range(k):
			Xj = X[assign==j]
			if Xj.shape[0] == 0:
				centroids[j] = X[np.random.randint(n_samples)]
			else:
				centroids[j] = np.mean(Xj, axis=0)

		if np.abs(prev_obj-obj)/(np.abs(obj)+1e-20) < 1e-8:
			break
		prev_obj = obj

		# stop criteria
	return centroids, assign

def spherical_kmeans(X, k, max_iter=1000, init=None):
	"""X: n x d points with unit-norm
	"""
	n_samples, n_features = X.shape
	if init is None:
		perm =  np.random.choice(np.arange(n_samples), k)
		centroids = X[perm]

	prev_obj = np.inf 
	for n_iter in range(max_iter):
		cos_sim = np.dot(X, centroids.T)
		assign = np.argmax(cos_sim, axis=1)
		obj = cos_sim[np.unravel_index(assign, cos_sim.shape)].mean()

		if (n_iter+1)%10 == 0:
			print("spherical kmeans iter %d, objective: %f"%(n_iter+1, obj))

		for j in range(k):
			Xj = X[assign==j]
			if Xj.shape[0] == 0:
				centroids[j] = X[np.random.randint(n_samples)]
			else:
				centroids[j] = np.sum(Xj, axis=0)
				norm = np.linalg.norm(centroids[j])
				centroids[j] /= norm 
		if np.abs(prev_obj-obj)/(np.abs(obj)+1e-20) < 1e-8:
			break
		prev_obj = obj

		# stop criteria
	return centroids, assign



if __name__ == "__main__":
	x = np.array([[ 0,  1,  2,  3],
			   [ 4,  5,  6,  7],
			   [ 8,  9, 10, 11],
			   [12, 13, 14, 15]])

	y = np.zeros((4,4,3))
	y[:,:,0] = x

	res = extract_patches_2d(y, 3, 1)
	print res.shape
	print res[0][:,:,0]
	print res[1][:,:,0]
	print res.reshape((res.shape[0], -1))
	patch = res.reshape((res.shape[0], -1, res.shape[-1]))
	center = patch.shape[1]//2
	bin_patch = patch-patch[:,[center],:]
	# print np.delete(bin_patch, center, 1) >0.0
	print binarize_patch(res)
	# res = res.reshape((res.shape[0],-1))
	# print res 
	res = res - np.mean(res, axis=(1, 2), keepdims=1)
	print res[0][:,:,0]
	print res.reshape((res.shape[0], -1))
	# print res.shape
	# print res.reshape((res.shape[0],-1))[:,[0,3,6,9]].mean(0)
	# print np.mean(res, axis=0)
	# res = res - np.mean(res, axis=0, keepdims=1)
	# print res

	# np.random.seed(0)
	# x = np.random.rand(30000,300)
	# x = normalize_row(x)
	# # print np.linalg.norm(x, axis=1)
	# print "1"
	# c, assign = spherical_kmeans(x, 100)
	# print c 
	# print np.linalg.norm(c, axis=1)
