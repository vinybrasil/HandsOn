# -*- coding: utf-8 -*-
#cap. 2 do Hands On Machine Learning with python
"""
Created on Sun Mar 22 11:07:07 2020

@author: N13M4ND
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



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
#solução: larga essa merda e usa o scikitlearn'''

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

housing = strat_train_set.copy()
#o alpha serve pra ver a densidade
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
#plt.show()

#plotando um color map do tipo jet, onde o raio(s) é a densidade da população
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='população',
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar='True')
plt.legend()
#plt.show()

#quanto mais perto do oceano, mais caro (dá pra usar um algoritmo de cluster)

#vendo as correlações: (quanto maior a renda, maior o preço médio)

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

#outra forma de ver as correlações:

atributos = ['median_house_value', 'median_income', 'total_rooms',
             'housing_median_age']
scatter_matrix(housing[atributos], figsize=(12,8))

#criando novas variáveis que podem ajudar a entender o problema:
housing['room_per_household'] = housing['total_rooms']/housing['households'] #quarto por pessoa
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']   
housing['population_per_household']=housing['population']/housing['households']

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

#==============data cleaning==========================

#tirando a variavel que vai ser predita:
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

#trocando os valores pela mediana

imputer = SimpleImputer(strategy='median')

#tirar a categorica pq só dá pra usar em numerica

housing_num = housing.drop('ocean_proximity', axis=1)

#fazendo o imputer saber quais sao as medianas:

imputer.fit(housing_num)

#os valores aplicados ficam guargados na variavel imputer.statistics_

#aplicando:

print(imputer.statistics_)
X = imputer.transform(housing_num)

#como x é array, deve-se criar um novo dataframe

housing_tr = pd.DataFrame(X, columns=housing_num.columns)


#lidando com as variaveis categoricas: 
'''
#LabelEncoder:
    
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity'] 
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(encoder.classes_) #as classes ficam aqui

#[0 1 2 3], mas nao necessariamente é essa a ordem, tipo 3<1

#One Hot Encoder:

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot.toarray()

'''
#O LabelBinarizer faz os dois ao msm tempo:
    
encoder = LabelBinarizer()
housing_cat = housing['ocean_proximity'] 
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

#dá uma dense np array, se quiser uma sparse matrix usa soarse_output=True no LabelBinarizer
#dá pra criar um tbm, mas tá no livro


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): 
        # no *args or **kargs        
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):        
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]        
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:            
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


#======================feature scaling====================================

#duas formas: min-max scaling(normalization), onde fica um valor
#entre 0 e 1 (ou o que vc quiser) e ativado com o MinMaxScaler
#e standarzation, que subtrai a média e divide pelo desvio padrao, 
#sendo menos influenciado pelos outliers, sendo chamado pelo StandardScaler


num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()), 
                 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


#antes, fazer o custom transformer:
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)
#um pipeline pra lidar com numericas e categoricas ao msm tempo pode
#ser escrito dessa forma:


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),   
    ])                                        
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ])
print(housing)
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

#===========selecionando e treinando o modelo============================

lm = LinearRegression()
predito = lm.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

data_preparada = full_pipeline.transform(some_data)


print('Predito:\t ', predito.predict(data_preparada))
print('Labels corretos:\t ', list(some_labels))

yhat = predito.predict(housing_prepared)
print(r2_score(housing_labels, yhat)) 

#underfitting, logo deve-se tentar modelos mais complexos

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
yhattree = tree_reg.predict(housing_prepared) 
print(r2_score(housing_labels, yhattree)) 

#overfitting, logo devese usar a cross validation com o kfold pra treinar 
#ele, usando 10 folds nesse caso
#(sklearn.metrics.SCORERS.keys()) pra ver os scoring que pode

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring='r2', cv=10)

def display_scores(scores):
    print('Scores:', scores)
    print('Média dos scores:', scores.mean())
    print('Desvio padrão dos score', scores.std())
    
print(display_scores(scores)) #foi pessimo, tava overfittando msm

#tentar o crossvalidation no modelo linear: 
scores = cross_val_score(predito, housing_prepared, housing_labels,
                         scoring='r2', cv=10)

print(display_scores(scores))

#melhor, mas vamos usar o randomforest que vai ser melhor:

forest_reg = RandomForestRegressor ()
forest_reg.fit(housing_prepared, housing_labels)
yhatrandom = forest_reg.predict(housing_prepared)
print(r2_score(housing_labels, yhatrandom))

scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring='r2', cv=10)

print(display_scores(scores))

#salvando modelos

#from sklearn.externals import joblib
#joblib.dump(my_model, "my_model.pkl")
#carregar ele:
#my_model_loaded = joblib.load("my_model.pkl")

#=-=-=-=-=-=-=-=-=fine tuning the model=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#usar o  gridsearch pra achar a melhor combinação de hiperparametros:

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]}, 
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2, 3, 4]},
    ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='r2')
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(mean_score, params)
    
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
finalr2 = r2_score(y_test, final_predictions)
print(finalr2)