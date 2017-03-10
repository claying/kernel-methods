# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cdef int ONE = 1

import numpy as np
cimport numpy as np
from cython cimport floating
from numpy.math cimport INFINITY
from libc.math cimport fabs
from scipy.linalg.cython_blas cimport saxpy, daxpy, sdot, ddot
from libc.stdio cimport printf

ctypedef np.uint32_t UINT32_t

ctypedef floating (*DOT)(int* N, floating* X, int* incX, floating* Y,
						 int* incY) nogil
ctypedef void (*AXPY)(int* N, floating* alpha, floating* X, int* incX,
					  floating* Y, int* incY) nogil

cdef enum:
	# Max value for our rand_r replacement (near the bottom).
	# We don't use RAND_MAX because it's different across platforms and
	# particularly tiny on Windows/MSVC.
	RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
	seed[0] ^= <UINT32_t>(seed[0] << 13)
	seed[0] ^= <UINT32_t>(seed[0] >> 17)
	seed[0] ^= <UINT32_t>(seed[0] << 5)

	return seed[0] % (<UINT32_t>RAND_R_MAX + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
	"""Generate a random integer in [0; end)."""
	return our_rand_r(random_state) % end

cdef inline floating fmax(floating x, floating y) nogil:
	if x > y:
		return x
	return y

cdef inline floating fmin(floating x, floating y) nogil:
	if x < y:
		return x
	return y

cdef inline void swap(int* x, int* y) nogil:
	cdef int t = x[0]
	x[0] = y[0] 
	y[0] = t 

# cdef extern from "cblas.h":
# 	void daxpy "cblas_daxpy"(int N, double alpha, double *X, int incX,
# 							 double *Y, int incY) nogil
# 	void saxpy "cblas_saxpy"(int N, float alpha, float *X, int incX,
# 							 float *Y, int incY) nogil

cdef void coordinate_descent_svc(floating[:] w, floating[:,::1] X, 
	floating[:] y, floating C, floating tol, object rng):
	"""Cython implementation for dual coordinate descent with shrinking for l2-SVM
	Reference
	Hsieh et al., ICML 2008
	https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf
	"""

	cdef AXPY axpy
	cdef DOT dot

	if floating is float:
		dtype = np.float32
		axpy = saxpy
		dot = sdot
	else:
		dtype = np.float64
		axpy = daxpy
		dot = ddot

	cdef int n_samples = y.shape[0]
	cdef int n_features = X.shape[1]
	cdef int max_iter = 3000
	cdef floating G
	cdef floating D = 0.5/C 
	cdef int[:] active_set = np.empty(n_samples, dtype=np.int32)
	cdef int active_size = n_samples
	# cdef floating buf = 0
	cdef int s = 0

	
	cdef floating proj_grad
	cdef floating proj_grad_max_old = INFINITY
	cdef floating proj_grad_min_old = -INFINITY
	cdef floating proj_grad_max_new
	cdef floating proj_grad_min_new

	# initialization 
	cdef floating[:] Q_diag = np.empty(n_samples, dtype=dtype)
	cdef floating alpha_old
	cdef floating[:] alpha = np.zeros(n_samples, dtype=dtype)
	# w = np.zeros(n_features, dtype=dtype)

	cdef floating* w_ptr = <floating*> &w[0]
	cdef floating* X_ptr = <floating*> &X[0,0] 
	cdef floating y_alpha
	cdef floating alpha_y
	cdef int n_iter = 0
	cdef int idx
	cdef int ii = 0
	cdef int i = 0
	cdef int j
	cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
	cdef UINT32_t* rand_r_state = &rand_r_state_seed


	for ii in range(n_samples):
		active_set[ii] = ii 
		Q_diag[ii] = dot(&n_features, &X_ptr[ii * n_features], &ONE, &X_ptr[ii * n_features], &ONE) + D 
		# w += y[i]*alpha[i]*X[i]
		y_alpha = y[ii]*alpha[ii]
		axpy(&n_features, &y_alpha, &X_ptr[ii * n_features], &ONE, w_ptr, &ONE)

	with nogil:
		for n_iter in range(max_iter):
			proj_grad_max_new = -INFINITY
			proj_grad_min_new = INFINITY

			for i in range(active_size):
				j = i + rand_int(<UINT32_t> (active_size - i), rand_r_state)
				swap(&active_set[i], &active_set[j])

			s = 0
			while s < active_size:
				idx = active_set[s]
				G = y[idx] * dot(&n_features, w_ptr, &ONE, &X_ptr[idx * n_features], &ONE) - 1 + alpha[idx] * D 
				# print ('alpha_i: %f' % alpha[idx])

				proj_grad = 0.0
				if alpha[idx] == 0.0:
					if G > proj_grad_max_old:
						active_size -= 1
						swap(&active_set[idx], &active_set[active_size])
						# s -= 1
						continue
					elif G < 0:
						proj_grad = G 
				else:
					proj_grad = G 

				proj_grad_max_new = fmax(proj_grad_max_new, proj_grad)
				proj_grad_min_new = fmin(proj_grad_min_new, proj_grad)
				# print fabs(proj_grad)

				if fabs(proj_grad) > 1e-12:
					alpha_old = alpha[idx]
					alpha[idx] = fmax(alpha[idx] - G/Q_diag[idx], 0.0)
					# w += (alpha[idx] - alpha_old) * y[idx] * X[idx]
					alpha_y = (alpha[idx] - alpha_old) * y[idx]
					axpy(&n_features, &alpha_y, &X_ptr[idx * n_features], &ONE, w_ptr, &ONE)

				s += 1

			# if n_iter % 10 == 0:
			# 	printf('.')

			if proj_grad_max_new - proj_grad_min_new <= tol:
				if active_size == n_samples:
					break 
				else:
					active_size = n_samples
					# printf("*")
					proj_grad_max_old = INFINITY
					proj_grad_min_old = -INFINITY
					continue

			if proj_grad_max_new <= 0.0:
				proj_grad_max_old = INFINITY
			else:
				proj_grad_max_old = proj_grad_max_new

			if proj_grad_min_new >= 0.0:
				proj_grad_min_old = -INFINITY
			else:
				proj_grad_min_old = proj_grad_min_new

	printf("\noptimization finished, #iter = %d \n" % (n_iter+1))
	# calculate objective value
	cdef floating v = 0.0
	cdef int nSV = 0
	cdef int iii = 0
	v += dot(&n_features, w_ptr, &ONE, w_ptr, &ONE)
	for iii in range(n_samples):
		v += alpha[iii] * (alpha[iii]*D - 2)
		if alpha[iii] > 0:
			nSV += 1
	v = v/2
	printf('objective value : %f \n' % v)
	printf('nSV: %d \n' % nSV)


def train_mcsvc(floating[:, ::1] X, int[:] y, floating C, floating tol, int n_classes, object rng):
	"""one-vs-rest strategy to solve multiclass SVC
	"""
	if floating is float:
		dtype = np.float32
	else:
		dtype = np.float64

	cdef int n_samples = y.shape[0]
	cdef int n_features = X.shape[1]

	cdef floating[:,::1] w = np.zeros((n_classes, n_features), dtype=dtype)
	cdef floating[:] wk
	cdef floating[:] yk = np.zeros(n_samples, dtype=dtype)
	cdef int k = 0

	for k in range(n_classes):
		for i in range(n_samples):
			if y[i] == k:
				yk[i] = +1.0
			else:
				yk[i] = -1.0
		wk = w[k]
		# print np.array(yk)
		coordinate_descent_svc(wk, X, yk, C, tol, rng)
	return w

def predict_mcsvc(floating[:,::1] w, floating[:] xi):
	"""predict mcsvc
	"""
	cdef int n_classes = w.shape[0]
	cdef int n_features = w.shape[1]
	cdef int yi_pred
	cdef floating dec_current_val
	cdef floating dec_max_val = -INFINITY
	cdef int k = 0

	for k in range(n_classes):
		dec_current_val = np.dot(w[k], xi)
		if dec_current_val > dec_max_val:
			yi_pred = k
			dec_max_val = dec_current_val

	return yi_pred

def cross_validation(floating[:, ::1] X, int[:] y, floating C, int n_folds, floating tol, int n_classes, object rng):
	"""cross validation for mcsvc
	"""
	if floating is float:
		dtype = np.float32
	else:
		dtype = np.float64

	cdef int n_samples = y.shape[0]
	cdef int n_features = X.shape[1]
	cdef int[:] perm = np.empty(n_samples, dtype=np.int32)
	cdef int[:] fold_start = np.empty(n_folds+1, dtype=np.int32)

	cdef int ii = 0
	cdef int jj = 0
	cdef int i = 0
	cdef int j
	cdef int begin, end 
	cdef idx

	cdef floating[:, ::1] X_sub
	cdef int[:] y_sub

	cdef floating[:,::] w 
	cdef int[:] y_pred = np.empty(n_samples, dtype=np.int32)
	cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
	cdef UINT32_t* rand_r_state = &rand_r_state_seed

	for ii in range(n_samples):
		perm[ii] = ii 

	for i in range(n_samples):
		j = i + rand_int(<UINT32_t> (n_samples - i), rand_r_state)
		swap(&perm[i], &perm[j])

	for ii in range(n_folds+1):
		fold_start[ii] = ii * n_samples / n_folds

	for ii in range(n_folds):
		begin = fold_start[ii]
		end = fold_start[ii+1]

		n_samples_sub = n_samples - (end - begin)
		X_sub = np.empty((n_samples_sub, n_features), dtype=dtype)
		y_sub = np.empty(n_samples_sub, dtype=np.int32)

		idx = 0
		for jj in range(begin):
			X_sub[idx] = X[perm[jj]]
			y_sub[idx] = y[perm[jj]]
			idx += 1

		for jj in range(end, n_samples):
			X_sub[idx] = X[perm[jj]]
			y_sub[idx] = y[perm[jj]]
			idx += 1

		w = train_mcsvc(X_sub, y_sub, C, tol, n_classes, rng)

		for jj in range(begin, end):
			y_pred[perm[jj]] = predict_mcsvc(w, X[perm[jj]])

	cdef double acc
	acc = accuracy(y, y_pred)
	return acc 

def accuracy(int[:] y_true, int[:] y_pred):
	cdef double acc = 0.0
	cdef int n_samples = y_true.shape[0]
	cdef int i = 0

	for i in range(n_samples):
		if y_true[i] == y_pred[i]:
			acc += 1

	acc = 100.0 * acc/n_samples
	return acc









