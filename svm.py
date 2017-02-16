import numpy as np 
import os, sys
sys.path = [os.path.dirname(os.path.abspath(__file__))+'/liblinear-2.1/python'] + sys.path 
from liblinearutil import *

class SVM(object):
	"""docstring for SVM"""
	def __init__(self, C):
		self.C = C 
		# self.kernel = 'linear'
		self.classifier = None

	def fit(self, Xtr, Ytr):
		prob = problem(Ytr.tolist(), Xtr.tolist())
		param = parameter('-s 1 -c %d' % self.C)
		classifier = train(prob, param)
		self.classifier = classifier

	def predict(self, Xte):
		if self.classifier is not None:
			Yte = [1]*Xte.shape[0]
			Ypr, accuracy, decision = predict(Yte, Xte.tolist(), self.classifier)
		else:
			raise ValueError('Error! Run fit first!')
		return np.array(Ypr, dtype=int)

	def evaluate(self, Xte, Yte):
		if self.classifier is not None:
			Ypr, accuracy, decision = predict(Yte.tolist(), Xte.tolist(), self.classifier)
			# Ypr = self.classifier(Xte)
		else:
			raise ValueError('Error! Run fit first!')
		return accuracy[0]


