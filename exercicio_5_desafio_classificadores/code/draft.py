# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:01:24 2016

@description: This file contains the solution to the fifth exercise list in
              Machine Learning subject at UNICAMP, this work is about the use  
              of classifiers to get the best precision. This is a challenge.
@author: Juan Sebastián Beleño Díaz
@email: jsbeleno@gmail.com
"""

# Loading the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from scipy.stats import expon


# ------------------------------- Getting Data -------------------------------

# Defining the URIs with raw data
url_train_data = 'http://www.ic.unicamp.br/%7Ewainer/cursos/2s2016/ml/train.csv'
url_test_data = 'http://www.ic.unicamp.br/%7Ewainer/cursos/2s2016/ml/test.csv'

# Reading the files with the raw data
df_train = pd.read_csv(url_train_data, header = None, delimiter = ",")
df_test = pd.read_csv(url_test_data, header = None, delimiter = ",")


# ---------------------------- Pre-processing the data -----------------------

# Number of columns and rows in the train data
n_columns = df_train.shape[1] # 33
n_rows = df_train.shape[0]    # 9000

# df_train.head(5)
# df_train.iloc[0:5,0:10]
# df_train.describe() - Summary
# df_train.iloc[:,0:10].describe() - partial summary


# Creating a label encoders to handle categorical data
categorical_attributes = [4,5,6,7,8,9,11,12,15,16,17,20,22,28,29,30]
general_le = []
invert_index_le = 0

for i in categorical_attributes:
    general_le.append(preprocessing.LabelEncoder())
    df_train[i] = general_le[invert_index_le].fit_transform(df_train[i])
    invert_index_le = invert_index_le + 1
    
train_params = df_train.iloc[:, 1:33]
train_classes = np.ravel(df_train.iloc[:, 1:2])

# plt.hist(train_classes)
# plt.show() the dataset is balanced

# I decided to avoid scaling and PCA data because I'm handling categorical data
# I did not find columns with the same value
    
# ---------------------------- Parameters ------------------------------------
n_iter_search = 20

n_internal_folds = 5
n_external_folds = 3

# WARNING: I work with an i5 with 4 cores 3.3.GHz, please adjust this parameter
# to the number of cores your processor have
n_jobs = 4
              
# ---------------------- Random Classification Models ------------------------

# I'm going to use an SVM to iterate and find the best hyperparameters
# I'm just to use a simple 3-Fold Cross-Validation    
def get_precision_svm(train_params, test_params, train_classes, test_classes):
    
    svm_params = {'C': expon(scale=100), 
                  'gamma': expon(scale=.1),
                  'kernel': ['rbf']}
    
    
    svm_model = SVC()

    # Parallelizing to get more speed
    clf_svm = RandomizedSearchCV(svm_model, 
                                 param_distributions = svm_params,
                                 n_iter = n_iter_search, 
                                 cv = n_internal_folds, 
                                 n_jobs = n_jobs)
    clf_svm.fit(train_params, train_classes)

    # Getting the best hyperparameters
    svm_best_hyperparams = clf_svm.best_params_
    print('C:', svm_best_hyperparams['C'])
    print('kernel:', svm_best_hyperparams['kernel'])
    print('gamma:', svm_best_hyperparams['gamma'])
    
    
    # Create the best SVM model
    svm_tuned = SVC(C = svm_best_hyperparams['C'], 
                    kernel = svm_best_hyperparams['kernel'], 
                    gamma = svm_best_hyperparams['gamma'])
    svm_tuned.fit(train_params, train_classes)

    # Getting the model precision
    svm_tuned_score = svm_tuned.score(test_params, test_classes)
    print('precision:', svm_tuned_score)

    return svm_tuned_score
    
# Just playing with Random Forest to find the best set of hyperparameters
def get_precision_rf(train_params, test_params, train_classes, test_classes):
    
    # Based on [1] we can find a good number of trees in the range of 63< nt <129.
    # Btw the work in [2] uses less than 9< nt < 201
    rf_parameters = {'n_estimators':range(9, 201),
                     'max_depth': [None],
                     "criterion": ["gini"]}
                                
    rf_model = RandomForestClassifier()
     # Parallelizing to get more speed
    clf_rf = RandomizedSearchCV( rf_model, 
                                 param_distributions = rf_parameters,
                                 n_iter = n_iter_search, 
                                 cv = n_internal_folds, 
                                 n_jobs = n_jobs)
    clf_rf.fit(train_params, train_classes)

    # Getting the best hyperparameters
    rf_best_hyperparams = clf_rf.best_params_
    print('n_estimators:', rf_best_hyperparams['n_estimators'])
    print('max_depth:', rf_best_hyperparams['max_depth'])
    print('criterion:', rf_best_hyperparams['criterion'])
    
    # Create the best Random Forest model
    rf_tuned = RandomForestClassifier(n_estimators = rf_best_hyperparams['n_estimators'],
                                      max_depth = rf_best_hyperparams['max_depth'],
                                      criterion = rf_best_hyperparams['criterion'])
    rf_tuned.fit(train_params, train_classes)

    # Getting the model precision
    rf_tuned_score = rf_tuned.score(test_params, test_classes)
    print('precision:', rf_tuned_score)

    return rf_tuned_score
    

# ------------------------ Here Goes the Magic -------------------------------

# Define the external K-Fold Stratified
external_skf = StratifiedKFold(n_splits = n_external_folds)
external_skf.get_n_splits(train_params, train_classes)

# Iterate over external data
for external_train_index, external_test_index in external_skf.split(train_params, train_classes):
    
    # Split the external training set and the external test set
    external_params_train = train_params.iloc[external_train_index, :] 
    external_classes_train = train_classes[external_train_index] 
    external_params_test = train_params.iloc[external_test_index, :]
    external_classes_test = train_classes[external_test_index]
    
    # Getting the precision of SVM with kernel RBF using a 3-Fold internal CV
    #svm_score = get_precision_svm(external_params_train, 
    #                              external_params_test, 
    #                              external_classes_train, 
    #                              external_classes_test)
    
    # Getting the precision of Random Forest with random hyperparameters
    rf_score = get_precision_rf(external_params_train, 
                                 external_params_test, 
                                 external_classes_train, 
                                 external_classes_test)

    
# [1] Oshiro, Thais Mayumi, Pedro Santoro Perez, and José Augusto Baranauskas. 
#    "How many trees in a random forest?." International Workshop on Machine 
#    Learning and Data Mining in Pattern Recognition. Springer Berlin Heidelberg, 2012.
    
# [2] Latinne, Patrice, Olivier Debeir, and Christine Decaestecker. 
#    "Limiting the number of trees in random forests." 
#    International Workshop on Multiple Classifier Systems. Springer Berlin Heidelberg, 2001.