# encoding: utf-8
from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport exp

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cimport cython 
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def conv2(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[DTYPE_t, ndim=2] g):
	if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
		raise ValueError("Only odd dimensions on filter supported")
	assert f.dtype == DTYPE and g.dtype == DTYPE
	cdef int vmax = f.shape[0]
	cdef int wmax = f.shape[1]
	cdef int smax = g.shape[0]
	cdef int tmax = g.shape[1]
	cdef int smid = smax // 2
	cdef int tmid = tmax // 2
	cdef int xmax = vmax + 2*smid
	cdef int ymax = wmax + 2*tmid
	cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros([xmax, ymax], dtype=DTYPE)
	cdef int x, y, s, t, v, w

	cdef int s_from, s_to, t_from, t_to

	cdef DTYPE_t value
	for x in range(xmax):
		for y in range(ymax):
			s_from = max(smid - x, -smid)
			s_to = min((xmax - x) - smid, smid + 1)
			t_from = max(tmid - y, -tmid)
			t_to = min((ymax - y) - tmid, tmid + 1)
			value = 0
			for s in range(s_from, s_to):
				for t in range(t_from, t_to):
					v = x - smid + s
					w = y - tmid + t
					value += g[smid - s, tmid - t] * f[v, w]
			h[x, y] = value
	return h[smid:vmax+smid, tmid:wmax+tmid]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_kernel(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y, sigma2=1.0, kernel='gaussian'):
	cdef:
		int mx = X.shape[0]
		int nx = X.shape[1]
		int my = Y.shape[0]
		int ny = Y.shape[1]
		DTYPE_t tmp, d 
		np.ndarray[DTYPE_t, ndim=2] K = np.zeros([mx, my], dtype=DTYPE) 
		int i, j
	assert nx == ny

	if kernel == 'gaussian':
		for i in range(mx):
			for j in range(my):
				d = 0.0
				for k in range(nx):
					tmp = X[i, k] - Y[j, k]
					d += tmp * tmp
				K[i, j] = exp(-d/(2.0*sigma2))
	else:
		raise ValueError("kernel not supported")
	return K 
