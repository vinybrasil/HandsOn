# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:17:57 2020

@author: vinic
"""

import pandas as pd
from matplotlib import pyplot as plt
import joblib
import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier

mnist = joblib.load("mnist.z")
X, Y = mnist[0], mnist[1]

Y = Y.astype(np.int8)

#criando o conjunto de teste e embaralhando o conjunto de treino:

X_TREINO, X_TESTE, Y_TREINO, Y_TESTE = X[:60000], X[60000:], Y[:60000], Y[60000:]
shuffle_index = np.random.permutation(60000)
X_TREINO, Y_TREINO = X_TREINO[shuffle_index], Y_TREINO[shuffle_index]

#=====================CLASSIFICADOR MULTICLASSE===========================


#treinando um one vs onde com o stochastic gradient
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42)).fit(X_TREINO, Y_TREINO)
print(ovo_clf.predict([X[36000]]))
print(len(ovo_clf.estimators_))

#treinando um random forest classifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_TREINO, Y_TREINO)
print(forest_clf.predict_proba([X[36000]]))