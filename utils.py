from numpy.lib.stride_tricks import as_strided
import numpy as np 

def extract_patches_2d(image, patch_size, step=1):
	"""image : nx x ny x n_channels"""
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

def centering(patch):
	return patch - np.mean(patch, axis=0, keepdims=1)

def normalize_row(X, norm='l2', threshold=1e-5, return_norm=False):
	if norm == 'l2':
		norms = np.linalg.norm(X, axis=1)
		norms[norms<threshold] = threshold
		X /= norms[:, np.newaxis]
	if return_norm:
		return X, norms
	return X 

if __name__ == "__main__":
	x = np.array([[ 0,  1,  2,  3],
			   [ 4,  5,  6,  7],
			   [ 8,  9, 10, 11],
			   [12, 13, 14, 15]])

	y = np.zeros((4,4,3))
	y[:,:,0] = x

	res = extract_patches_2d(y, 2, 1)
	print res[0][:,:,0]
	print res[1][:,:,0]
	# res = res.reshape((res.shape[0],-1))
	# print res 
	print res.shape
	print res.reshape((res.shape[0],-1))[:,[0,3,6,9]].mean(0)
	print np.mean(res, axis=0)
	res = res - np.mean(res, axis=0, keepdims=1)
	print res 
