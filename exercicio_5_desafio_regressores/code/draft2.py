# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:40:11 2016

@description: This file contains the solution to the fifth exercise list in
              Machine Learning subject at UNICAMP, this work is about the use  
              of classifiers to get the best precision. This is a challenge.
              This files has better quality code than the fisrt draft.
@author: Juan Sebastián Beleño Díaz
@email: jsbeleno@gmail.com
"""

# Loading the libraries
import math
import numpy as np
import pandas as pd

from scipy.stats import randint as sp_randint
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

# ------------------------------- Getting Data -------------------------------

# Defining the URIs with raw data
url_train_data = 'http://www.ic.unicamp.br/%7Ewainer/cursos/2s2016/ml/train.csv'
url_test_data = 'http://www.ic.unicamp.br/%7Ewainer/cursos/2s2016/ml/test.csv'

# Reading the files with the raw data
df_train = pd.read_csv(url_train_data, header = None, delimiter = ",")
df_test = pd.read_csv(url_test_data, header = None, delimiter = ",")

# ---------------------------- Pre-processing the data -----------------------

# Creating a label encoders to handle categorical data
categorical_attributes = [4,5,6,7,8,9,11,12,15,16,17,20,22,28,29,30]
general_le = []
invert_index_le = 0

train_params = df_train.iloc[:, 1:33]
train_values = np.ravel(df_train.iloc[:, 1:2])

df_train_with_numbers = df_train

for i in categorical_attributes:
    general_le.append(preprocessing.LabelEncoder())
    df_train_with_numbers[i] = general_le[invert_index_le].fit_transform(df_train_with_numbers[i])
    invert_index_le = invert_index_le + 1
    
train_params_with_numbers = df_train_with_numbers.iloc[:, 1:33]
    

# Number of columns and rows in the train data
n_columns = df_train.shape[1] # 33
n_rows = df_train.shape[0]    # 9000


# ---------------------------- Parameters ------------------------------------

# Number of splits for internal and external cross validation
n_internal_folds = 3
n_external_folds = 3

# Random Forest
depth_rf = 3 # log2(sqrt(33)) = 2,52219705968
max_features_limit = 8 # 5.74456264654(sqrt(33)) + 2,52219705968

param_rf = {"max_depth": [depth_rf, None],
            "max_features": sp_randint(1, max_features_limit),
            "min_samples_split": sp_randint(1, max_features_limit),
            "min_samples_leaf": sp_randint(1, max_features_limit),
            "bootstrap": [True, False],
            "criterion": ["mae"]}

# WARNING: I work with an i5 with 4 cores 3.3.GHz, please adjust this parameter
# to the number of cores your processor have
n_jobs = 1
pre_dispatch = 2 # 2 * n_jobs

# Number of random iterations
n_iter_search = 3

# ---------------------- Random Classification Models ------------------------

def rf_precision(train_params, test_params, train_values, test_values):
    
    regressor = RandomForestRegressor()
    random_search = RandomizedSearchCV(regressor, 
                                       param_distributions=param_rf,
                                       n_iter=n_iter_search,
                                       n_jobs = n_jobs,
                                       pre_dispatch = pre_dispatch,
                                       cv = n_internal_folds,
                                       refit = False)
    random_search.fit(train_params, train_values)
    
    best_hyperparams = random_search.best_params_
    print('max_depth:', best_hyperparams['max_depth'])
    print('max_features:', best_hyperparams['max_features'])
    print('min_samples_split:', best_hyperparams['min_samples_split'])
    print('min_samples_leaf:', best_hyperparams['min_samples_leaf'])
    print('bootstrap:', best_hyperparams['bootstrap'])
    print('criterion:', best_hyperparams['criterion'])
    
    best_model = RandomForestRegressor(max_depth = best_hyperparams['max_depth'],
                                       max_features = best_hyperparams['max_features'],
                                       min_samples_split= best_hyperparams['min_samples_split'],
                                       min_samples_leaf = best_hyperparams['min_samples_leaf'],
                                       bootstrap = best_hyperparams['bootstrap'],
                                       criterion = best_hyperparams['criterion'])
    best_model.fit(train_params, train_values)                                       
                                       
    model_predictions = best_model.predict(test_params)
    mae = mean_absolute_error(test_values, model_predictions)
    
    print("MAE RF:", mae)
    random_search = nil
    print("---------------------------------------------")
    
# ------------------------ Here Goes the Magic -------------------------------

# Define the external K-Fold Stratified
external_skf = StratifiedKFold(n_splits = n_external_folds)
external_skf.get_n_splits(train_params, train_values)

# Iterate over external data
for external_train_index, external_test_index in external_skf.split(train_params, train_values):
    
    # Split the external training set and the external test set
    external_params_train = train_params.iloc[external_train_index, :]
    external_params_train_with_numbers = train_params_with_numbers.iloc[external_train_index, :]
    external_classes_train = train_values[external_train_index] 
    external_params_test = train_params.iloc[external_test_index, :]
    external_params_test_with_numbers = train_params_with_numbers.iloc[external_test_index, :]
    external_classes_test = train_values[external_test_index]
    
    # Random Forest Regressor
    rf_precision(external_params_train_with_numbers, 
                 external_params_test_with_numbers, 
                 external_classes_train, 
                 external_classes_test)
