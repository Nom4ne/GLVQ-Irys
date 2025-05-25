# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:11:47 2022
Generalized Learning Vector Quantization
@author: 
    https://proceedings.neurips.cc/paper_files/paper/1995/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf
    https://mrnuggelz.github.io/sklearn-lvq/glvq.html
    adopted by RZ
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
import hickle as hkl
# Install sklearn-lvq if not already installed:
# pip install sklearn-lvq
from sklearn_lvq import GlvqModel

x,y_t,x_norm,x_n_s,y_t_s = hkl.load('iris.hkl')
y_t -= 1
x=x.T
y_t = np.squeeze(y_t)

# Create and fit the LVQ classifier
lvq = GlvqModel()
help(lvq)
lvq.fit(x, y_t)

# Make predictions
y_pred = lvq.predict(x)

e = y_t - y_pred
PK = sum(abs(e)<0.5)/e.shape[0] * 100

# Evaluate the performance
print("\nPK = %5d" % PK)


# Example of StratifiedKFold using
data = x
target = y_t

CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
print(skfold)
PK_vec = np.zeros(CVN)

for i, (train, test) in enumerate(skfold.split(data, target), start=0):
    x_train, x_test = data[train], data[test]
    y_train, y_test = target[train], target[test]
    #print(x_train, x_test,y_train, y_test)
    result = lvq.predict(x_test)

    n_test_samples = test.size
    PK_vec[i] = np.sum(result == y_test) / n_test_samples
    
    print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i], n_test_samples))

PK = np.mean(PK_vec)
print("PK {}".format(PK))