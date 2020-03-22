# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:07:07 2020

@author: vinic
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

housing = pd.read_csv("data/housing.csv")
print(housing.info()) #diz se é se tem null e o tipo
print(housing['ocean_proximity'].value_counts) #quanto de cada variavel categorica tem
print(housing.describe())

'''
#housing.hist(bins=50, figsize=(20,15)) #plotando histogramas
#plt.show()

#criando o test set:
def createtestset(data, razao):
    misturarIndices = np.random.permutation(len(data))
    tamanhoTeste = int(len(data) * razao)  
    indicesTeste = misturarIndices[:tamanhoTeste]
    indicesTreino = misturarIndices[tamanhoTeste:]
    return data.iloc[indicesTreino], data.iloc[indicesTeste]

conjuntoTreino, conjuntoTeste = createtestset(housing, 0.2)
print(len(conjuntoTreino), "treino + ", len(conjuntoTeste), " teste")
#cada vez que ele roda, gera um novo dataset de treino
#solução: larga essa merda e usa o scikitlearn
'''

#criando o test set: (o 42 é pra sempre ter o msm trainset)
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#arrumando pra que o dataset fique estratificado(as amostras sejam representativas):
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):    
    strat_train_set = housing.loc[train_index]    
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):    
    set.drop(["income_cat"], axis=1, inplace=True)


