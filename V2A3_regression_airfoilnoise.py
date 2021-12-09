#!/usr/bin/env python
# V2A3_regression_airfoilnoise.py
# Programmgeruest zu Versuch 2, Aufgabe 3
# to log outputs start with: python V2A3_regression_airfoilnoise.py >V2A3_regression_airfoilnoise.log

import numpy as np
import pandas as pd

from V2A2_Regression import *


# ***** MAIN PROGRAM ********
# (I) Hyper-Parameters
S=4;               # S-fold cross-validation
lmbda=1;           # regularization parameter (lambda>0 avoids also singularities)
K=12;               # K for K-Nearest Neighbors
flagKLinReg = 1;   # if flag==1 and K>=D then do a linear regression of the KNNs to make prediction
deg=5;             # degree of basis function polynomials 
flagSTD=1;         # if >0 then standardize data before training (i.e., scale X to mean value 0 and standard deviation 1)
N_pred=1;          # number of predictions on the training set for testing
x_test_1 = [1250, 11, 0.2, 69.2, 0.0051]
x_test_2 = [1305, 8, 0.1, 57.7, 0.0048]

# (II) Load data 
fname='../DATA/AirfoilSelfNoise/airfoil_self_noise.xls'
airfoil_data = pd.read_excel(fname,0); # load data as pandas data frame 
T = airfoil_data.values[:,5]           # target values = noise load (= column 5 of data table)
print("T:", T)
X = airfoil_data.values[:,:5]          # feature vectors (= column 0-4 of data table)
N,D=X.shape                            # size and dimensionality of data set
idx_perm = np.random.permutation(N)    # get random permutation for selection of test vectors 
print("Data set ",fname," has size N=", N, " and dimensionality D=",D)
print("X=",X)
print("T=",T)
print("x_test_1=",x_test_1)
print("x_test_2=",x_test_2)
print("number of basis functions M=", len(phi_polynomial(X[1],deg)))

phi=lambda x: phi_polynomial(x,2)
# (III) Do least-squares regression with regularization 
print("\n#### Least Squares Regression with regularization lambda=", lmbda, " ####")
lsr = LSRRegressifier(lmbda,phi)  # REPLACE dummy code: Create and fit Least-Squares Regressifier using polynomial basis function of degree deg and flagSTD for standardization of data  
lsr.fit(X,T)
print("lsr.W_LSR=",lsr.W_LSR)   # REPLACE dummy code: print weight vector for least squares regression)

print("III.1) Some predictions on the training data:")
for i in range(N_pred): 
    n=idx_perm[i]
    print("Prediction for X[",n,"]=",X[n]," is y=",lsr.predict(X[n]),", whereas true value is T[",n,"]=",T[n])   # REPLACE dummy code: compute prediction for X[n]
print("III.2) Some predictions for new test vectors:")
print("Prediction for x_test_1 is y=", lsr.predict(x_test_1))   # REPLACE dummy code: compute prediction for x_test_1
print("Prediction for x_test_2 is y=", lsr.predict(x_test_2))   # REPLACE dummy code: compute prediction for x_test_2
print("III.3) S=",S,"fold Cross Validation:")
err_abs,err_rel = lsr.crossvalidate(S,X,T)                   # REPLACE dummy code: do cross validation!! 
print("absolute errors (E,sd,min,max)=", err_abs, "\nrelative errors (E,sd,min,max)=", err_rel)

# (IV) Do KNN regression  
print("\n#### KNN regression with flagKLinReg=", flagKLinReg, " ####")
knnr = KNNRegressifier(K)
knnr.fit(X,T)                                  # REPLACE dummy code: Create and fit KNNRegressifier
print("IV.1) Some predictions on the training data:")
for i in range(N_pred): 
    n=idx_perm[i]
    print("Prediction for X[",n,"]=",X[n]," is y=",knnr.predict(X[n]),", whereas true value is T[",n,"]=",T[n])  # REPLACE dummy code: compute prediction for X[n]
print("IV.2) Some predictions for new test vectors:")
print("Prediction for x_test_1 is y=", knnr.predict(x_test_1))   # REPLACE dummy code: compute prediction for x_test_1
print("Prediction for x_test_2 is y=", knnr.predict(x_test_2))   # REPLACE dummy code: compute prediction for x_test_2
print("IV.3) S=",S,"fold Cross Validation:")
err_abs,err_rel = knnr.crossvalidate(S,X,T)                   # REPLACE dummy code: do cross validation!! 
print("absolute errors (E,sd,min,max)=", err_abs, "\nrelative errors (E,sd,min,max)=", err_rel)



# (V) Do KNN regression with scikit learn
print("\n(V) Do KNN regression with scikit learn")
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(np.round(X),np.round(T))
KNeighborsClassifier()
y_pred_1_KNN = neigh.predict(np.array([x_test_1]))
y_pred_2_KNN = neigh.predict(np.array([x_test_2]))
print("KNN regression with scikit learn for x_test_1 is: ",y_pred_1_KNN)
print("KNN regression with scikit learn for x_test_1 is: ",y_pred_2_KNN)

# (VI) Do LSR regression with scikit learn
print("\n(VI) Do LSR regression with scikit learn")
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, T)
reg.score(X, T)
y_pred_1_LSR = reg.predict(np.array([x_test_1]))
y_pred_2_LSR = reg.predict(np.array([x_test_2]))
print("LSR regression with scikit learn for x_test_1 is: ",y_pred_1_LSR)
print("LSR regression with scikit learn for x_test_2 is: ",y_pred_2_LSR)

# (VII) Do LSR regression with scikit learn
print("\n(VII) Do SVM with scikit learn")
from sklearn.svm import SVC
clf = SVC()
clf.fit(np.round(X),np.round(T))
SVC(C=1.0)
y_pred_1_SVM = clf.predict(np.array([x_test_1]))
y_pred_2_SVM = clf.predict(np.array([x_test_2]))
print("SVM regression with scikit learn for x_test_1 is: ",y_pred_1_SVM)
print("SVM regression with scikit learn for x_test_2 is: ",y_pred_2_SVM)

# (VIII) Do MLP regression with scikit learn
print("\n(VII) Do MLP with scikit learn")
from sklearn.neural_network import MLPClassifier
clfMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
clfMLP.fit(np.round(X),np.round(T))
y_pred_1_MLP = clfMLP.predict(np.array([x_test_1]))
y_pred_2_MLP = clfMLP.predict(np.array([x_test_2]))
print("MLP regression with scikit learn for x_test_1 is: ",y_pred_1_MLP)
print("MLP regression with scikit learn for x_test_2 is: ",y_pred_2_MLP)