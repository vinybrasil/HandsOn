# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:05:16 2020

@author: N13M4ND
"""

#classificação: predizer classes

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

'''
# da primeira vez faz isso: 

from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, return_X_y=True)
import joblib
joblib.dump(mnist, "mnist.z")
'''

mnist = joblib.load("mnist.z")
X, Y = mnist[0], mnist[1]

print(X)
print(Y)

Y = Y.astype(np.int8)
'''
DIGITO = X[36000]
print(DIGITO.shape)

DIGITO_IMAGEM = DIGITO.reshape(28,28)
plt.imshow(DIGITO_IMAGEM, cmap=matplotlib.cm.binary, 
           interpolation='nearest')
plt.axis('off')
plt.show()
print(Y[36000], "\n\n\nnoice.")
'''

#criando o conjunto de teste e embaralhando o conjunto de treino:

X_TREINO, X_TESTE, Y_TREINO, Y_TESTE = X[:60000], X[60000:], Y[:60000], Y[60000:]
shuffle_index = np.random.permutation(60000)
X_TREINO, Y_TREINO = X_TREINO[shuffle_index], Y_TREINO[shuffle_index]

#========treinando um classificador binário: é 0 ou não é?======================

Y_TREINO_0 = (Y_TREINO == 0) #dá True pra 9 e False pro resto
Y_TESTE_0 = (Y_TESTE == 0)

#usar o stochastic gradient descent

#da primeira vez:

#sgd_clf = SGDClassifier()
#modelo = sgd_clf.fit(X_TREINO, Y_TREINO_0)
#joblib.dump(modelo, "modelo.z")

modelo = joblib.load("modelo.z")

DIGITO = X_TREINO[1]
DIGITO_IMAGEM = DIGITO.reshape(28,28)
plt.imshow(DIGITO_IMAGEM, cmap=matplotlib.cm.binary, 
           interpolation='nearest')
plt.axis('off')
plt.show()
print(Y_TREINO[1])
print(modelo.predict([DIGITO])) #TEY 

#performarnce measures: evalidar um classificador é mais dificil que um regressor

#===========Cross validation: measuring accuracy==============================

#construindo um cross validador, dado que o cross_val_score é mto limitado

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_TREINO, Y_TREINO_0):
    clone_clf = clone(modelo)
    X_TREINO_FOLDS = X_TREINO[train_index]
    Y_TREINO_FOLDS = Y_TREINO_0[train_index]
    X_TESTE_FOLDS = X_TREINO[test_index]
    Y_TESTE_FOLDS = Y_TREINO_0[test_index]
    
    clone_clf.fit(X_TREINO_FOLDS, Y_TREINO_FOLDS)
    y_pred = clone_clf.predict(X_TREINO_FOLDS)
    n_correct = sum(y_pred == Y_TREINO_FOLDS)
    print(n_correct/(len(y_pred)))
    
print(cross_val_score(modelo, X_TREINO, Y_TREINO_0, cv=3, scoring="accuracy"))

#o modelo tem boa acurácia, mas um modelo que diz que tudo é "não-zero"
#tem 90% de acertO, logo acurácia é um lixo (pq 90% do dataset é "não-zero")

class Never0Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

nuncazero = Never0Classifier()
print(cross_val_score(nuncazero, X_TREINO, Y_TREINO_0, cv=3, scoring="accuracy"))

#==================Confusion Matrix===========================================

#precisamos dos valores pra construir ela. O cross_val_predict dá os valores 
#preditos com um kfold de 3. Cada linha é a classe certa e cada coluna
#representa a classe predita

Y_TREINO_PRED = cross_val_predict(modelo, X_TREINO, Y_TREINO_0, cv=3)

print("Confusion matrix")
print(confusion_matrix(Y_TREINO_0, Y_TREINO_PRED))
print("53499 = True negative\n578 = False positive\n247 = False positive  5676 = True Positive\n")
print("    0  |  1  ")
print("0 | TN |  FP ")
print("1 | FN |  TP ")
print('\nPrecisao = TP/(TP + FP), para o teste: ', precision_score(Y_TREINO_FOLDS, y_pred))
print('Recall = TP/(TP + FN), para o teste: ', recall_score(Y_TREINO_0, Y_TREINO_PRED))
print("A f1 é a média harmonica das duas: ", f1_score(Y_TREINO_FOLDS, y_pred))

