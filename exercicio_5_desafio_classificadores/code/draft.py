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

from sklearn import preprocessing
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

# I decided to avoid scaling and PCA data because I'm handling categorical data
# I did not find columns with the same value
    
# ---------------------------- Parameters ------------------------------------
n_iter_search = 20

n_internal_folds = 3
n_external_folds = 3

n_jobs = 4
              
# ----------------------  Classification Models ------------------------------

# I'm going to use an SVM to iterate and find the best hyperparameters
# I'm just to use a simple 3-Fold Cross-Validation    
def get_precision_svm(train_params, test_params, train_classes, test_classes):
    
    svm_params = {'C': expon(scale=100), 
                  'gamma': expon(scale=.1),
                  'kernel': ['rbf'], 
                  'class_weight':['balanced', None]}
    
    
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
    print(svm_best_hyperparams)
    
    # Create the best SVM model
    svm_tuned = SVC(C = svm_best_hyperparams['C'], 
                    kernel = svm_best_hyperparams['kernel'], 
                    gamma = svm_best_hyperparams['gamma'],
                    class_weight = svm_best_hyperparams['class_weight'])
    svm_tuned.fit(train_params, train_classes)

    # Getting the model precision
    svm_tuned_score = svm_tuned.score(test_params, test_classes)

    return svm_tuned_score
    

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
    svm_score = get_precision_svm(external_params_train, 
                                  external_params_test, 
                                  external_classes_train, 
                                  external_classes_test)
    
    print(svm_score)

    
