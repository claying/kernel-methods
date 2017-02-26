import numpy as np 
import os, sys
sys.path = [os.path.dirname(os.path.abspath(__file__))+'/liblinear-2.1/python'] + sys.path 
from liblinearutil import *

class SVM(object):
	"""docstring for SVM"""
	def __init__(self, C):
		"""C parameter for svm"""
		self.C = C 
		self.classifier = None

	def fit(self, Xtr, Ytr):
		prob = problem(Ytr.tolist(), Xtr.tolist())
		param = parameter('-s 1 -c %f -s 2' % self.C)
		classifier = train(prob, param)
		self.classifier = classifier

	def cross_val(self, Xtr, Ytr):
		prob = problem(Ytr.tolist(), Xtr.tolist())
		param = parameter('-s 1 -c %f -v 5 -s 2' % self.C)
		accuracy = train(prob, param)
		return accuracy

	def predict(self, Xte):
		if self.classifier is not None:
			Yte = [1]*Xte.shape[0]
			Ypr, accuracy, decision = predict(Yte, Xte.tolist(), self.classifier, '-q')
		else:
			raise ValueError('Error! Run fit first!')
		return np.array(Ypr, dtype=int)

	def evaluate(self, Xte, Yte):
		if self.classifier is not None:
			Ypr, accuracy, decision = predict(Yte.tolist(), Xte.tolist(), self.classifier)
		else:
			raise ValueError('Error! Run fit first!')
		return accuracy[0]


