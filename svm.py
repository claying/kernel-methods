import numpy as np 
from svm_solver import train_mcsvc, predict_mcsvc, cross_validation
from utils import check_random_state

class SVM(object):
	def __init__(self, C, seed=None, n_folds=5, tol=0.1):
		"""C parameter for svm"""
		self.rng = check_random_state(seed)
		self.C = C 
		self.w = None
		self.tol = tol 
		self.n_folds = n_folds # number of folds for cross validation

	def fit(self, Xtr, Ytr):
		self.n_classes = np.max(Ytr) + 1
		Xtr = np.asarray(Xtr, order='C', dtype=np.float32)
		Ytr = np.asarray(Ytr, order='C', dtype=np.int32)
		# Ytr = Ytr.astype(Xtr.dtype)
		self.w = train_mcsvc(Xtr, Ytr, self.C, self.tol, self.n_classes, self.rng)

	def cross_val(self, Xtr, Ytr):
		self.n_classes = np.max(Ytr) + 1
		Xtr = np.asarray(Xtr, order='C', dtype=np.float32)
		Ytr = np.asarray(Ytr, order='C', dtype=np.int32)
		accuracy = cross_validation(Xtr, Ytr, self.C, self.n_folds, self.tol, self.n_classes, self.rng)
		print ("cross validation accuracy: %.1f%%" % accuracy)
		return accuracy

	def predict(self, Xte):
		if self.w is not None:
			Xte = np.asarray(Xte, order='C', dtype=np.float32)
			Ypr = np.zeros(Xte.shape[0])
			for i in range(Xte.shape[0]):
				xi = Xte[i,:]
				Ypr[i] = predict_mcsvc(self.w, xi)
		else:
			raise ValueError('Error! Run fit first!')
		return Ypr.astype(np.int32)




if __name__ == "__main__":
	np.random.seed(0)
	rng = np.random.RandomState(0)
	Xtr = np.random.rand(200,20)
	Ytr = np.random.randint(10,size=200)
	Ytr = Ytr.astype(np.int32)
	print np.sum(Xtr**2, axis=1)


	svm2 = SVM(C=1, seed=rng)
	print svm2.cross_val(Xtr, Ytr)
	# print np.array(svm2.w)
	# print svm2.predict(Xtr)