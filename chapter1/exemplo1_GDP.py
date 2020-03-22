# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:56:16 2020

@author: vinic
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier


ocde = pd.read_csv("datasets/oecd.csv", thousands=',')
gdp = pd.read_csv("datasets/gdp.csv", thousands=',', delimiter='\t',
                  encoding='latin1', na_values='n/a')


def prepare_country_stats(oecd, gdp):
    oecd = oecd[oecd["INEQUALITY"]=="TOT"]
    oecd = oecd.pivot(index="Country", columns="Indicator", values="Value")
    gdp.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd, right=gdp,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[keep_indices]

country_stats = prepare_country_stats(ocde, gdp)

#concatenar duas array
X = np.c_[country_stats["GDP per capita"]]
Y = np.c_[country_stats["Life satisfaction"]]

print(X)
print(Y)

country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
plt.show()

lm = sklearn.linear_model.LinearRegression()
lm.fit(X,Y)
xnova =[[22587]]
print(lm.predict(xnova))

#ou o knn
'''
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,Y)
print(knn.predict(xnova))
'''