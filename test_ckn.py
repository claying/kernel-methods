from ckn import CKN, save, load 
from data_helper import load_data, save_pred
import numpy as np
import time
# from sklearn.model_selection import train_test_split

np.random.seed(0)
X, y = load_data(split='train', reshape_images=True)


model = CKN([1, 2], [2, 4], [12, 800], 'gradient')
tic = time.time()
model.train(X)
toc = time.time()
save(model, 'model/grad_800')
# model = load('model/grad_200')
# X_ckn = model.out_maps(X)

del X 
print X_ckn.shape

from svm import SVM
X_ckn = X_ckn - X_ckn.mean(axis=1, keepdims=1)
norm = np.mean(np.sqrt(np.sum(np.square(X_ckn), axis=1)))
X_ckn = X_ckn/norm

accs = []
for C in [256, 512, 1024]:
	clf = SVM(C=C)

	acc = clf.cross_val(X_ckn, y)
	accs.append(acc)
print accs
print("ckn training time: %.2f" % (toc - tic))
i = np.argmax(accs)
C = accs[i]
print C

clf = SVM(C=128)
clf.fit(X_ckn, y)
# save(clf, 'model/clf_grad_200')
del X_ckn
print("Prediction ...")
X_test = load_data(split='test', reshape_images=True)
X_ckn_te = model.out_maps(X_test)
del X_test
print X_ckn_te.shape

X_ckn_te = X_ckn_te/norm
y_pr = clf.predict(X_ckn_te)

save_pred(y_pr)

