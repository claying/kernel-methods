from ckn2 import CKN, save, load 
from data_helper import load_data, save_pred
import numpy as np
import time
from svm import SVM 

def single_model(model, train=True, save_model=False, prediction=False, save_svm_model=False):
	X, y = load_data(split='train', reshape_images=True)

	if train:
		tic = time.time()
		model.train(X)
		toc = time.time()
		print("ckn training time: %.2f s" % (toc - tic))


	### uncomment if you want to save the model
	if save_model:
		save(model, 'model/'+model.layer_type+'_'+str(model.n_layers))
	# model = load('model/patch_100_200')

	X_ckn = model.out_maps(X)
	del X 
	print X_ckn.shape

	X_ckn = X_ckn - X_ckn.mean(axis=1, keepdims=1)
	norm = np.mean(np.sqrt(np.sum(np.square(X_ckn), axis=1)))
	X_ckn = X_ckn/norm

	### cross validation
	print('Cross validation ...')
	accs = []
	C_list = 2.0**np.arange(-1,6)
	for C in C_list:
		clf = SVM(C=C)

		acc = clf.cross_val(X_ckn, y)
		accs.append(acc)
	print accs
	i = np.argmax(accs)
	C = C_list[i]
	print C

	### train and prediction
	if prediction:
		clf = SVM(C=C)
		clf.fit(X_ckn, y)
		if save_svm_model:
			save(clf, 'model/clf_'+model.layer_type)
		del X_ckn
		print("Prediction ...")
		X_test = load_data(split='test', reshape_images=True)
		X_ckn_te = model.out_maps(X_test)
		del X_test
		print X_ckn_te.shape

		X_ckn_te = X_ckn_te/norm
		y_pr = clf.predict(X_ckn_te)

		save_pred(y_pr)

def combine_model(models, train=True, save_model=False, prediction=False, save_svm_model=False):
	X, y = load_data(split='train', reshape_images=True)

	norms = []
	X_ckn_tr = []
	for i, model in enumerate(models):
		if train:
			tic = time.time()
			model.train(X)
			toc = time.time()
			print("ckn training time: %.2f s" % (toc - tic))
			if save_model:
				save(model, 'model/'+model.layer_type+'_'+str(model.n_layers))
		else:
			model = load('model/'+model.layer_type+'_'+str(model.n_layers))
			models[i] = model 
		X_ckn = model.out_maps(X)
		X_ckn = X_ckn - X_ckn.mean(axis=1, keepdims=1)
		norm = np.mean(np.sqrt(np.sum(np.square(X_ckn), axis=1)))
		X_ckn = X_ckn/norm
		norms.append(norm)
		X_ckn_tr.append(X_ckn)

	X_ckn_tr = np.hstack(X_ckn_tr)
	del X 
	del X_ckn

	print X_ckn_tr.shape

	### cross validation
	print('Cross validation ...')
	accs = []
	C_list = 2.0**np.arange(-2,4)
	for C in C_list:
		clf = SVM(C=C)

		acc = clf.cross_val(X_ckn_tr, y)
		accs.append(acc)
	print accs
	i = np.argmax(accs)
	C = C_list[i]
	print C


	### train and prediction
	if prediction:
		clf = SVM(C=C)
		clf.fit(X_ckn_tr, y)
		del X_ckn_tr
		if save_svm_model:
			save(clf, 'model/combine_clf_'+model.layer_type)
		print("Prediction ...")
		X_test = load_data(split='test', reshape_images=True)
		# X_ckn_te = model.out_maps(X_test)
		X_ckn_te = []
		for i, model in enumerate(models):
			X_ckn = model.out_maps(X_test)
			X_ckn = X_ckn/norms[i]
			X_ckn_te.append(X_ckn)

		X_ckn_te = np.hstack(X_ckn_te)
		del X_test
		del X_ckn 
		print X_ckn_te.shape

		y_pr = clf.predict(X_ckn_te)

		save_pred(y_pr)




if __name__ == "__main__":
	np.random.seed(0)
	model1 = CKN([1, 2], [2, 4], [12, 200], 'gradient')
	model2 = CKN([2, 2], [2, 4], [100, 200], 'patch')
	model3 = CKN([3], [10], [200], 'patch')
	# model4 = CKN([3, 2], [2, 4], [100, 200], 'shape')
	model = [model1, model2, model3]
	# single_model(model)
	combine_model(model, train=1, save_model=False, prediction=False)
