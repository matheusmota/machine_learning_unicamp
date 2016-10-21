# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:51:31 2016

@description: This file contains the solution to the third exercise list in
              Machine Learning subject at UNICAMP, this work is about the use  
              of SVM, kNN, Neural Nets, Random Forest and Gradient
              Boosting Machine.
@author: Juan Sebastián Beleño Díaz
@email: jsbeleno@gmail.com
"""

# Loading the libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ------------------------ Getting and Cleaning Data -------------------------

# Defining the URIs with raw data
url_parameters = 'https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data'
url_results = 'https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data'

# Reading the files with the raw data
df_parameters = pd.read_csv(url_parameters, header = 0, delimiter = " ")
df_results = pd.read_csv(url_results, header = 0, delimiter = " ")

# Getting classes from result
df_classes = df_results.iloc[:, 0:1]
df_classes = np.ravel(df_classes)


# ------------------------- Parameter declaration ----------------------------

# Number of columns and rows in the raw data
n_columns = df_parameters.shape[1]
n_rows = df_parameters.shape[0]

# Precision mean for all models
knn_precision = 0 
svm_precision = 0
neural_net_precision = 0
random_forest_precision = 0
gbm_precision = 0

# Folds variables
n_external_folds = 5
n_internal_folds = 3

# 80% of variance in the PCA
variance_percentage_pca = 0.8
n_components_pca = 0

# k values for kNN
knn_parameters = {'n_neighbors':[1, 5, 11, 15, 21, 25]}

# parameters for SVM
svm_parameters = {'kernel':('rbf'), 'C':[2**(-5), 2**(0), 2**(5), 2**(10)], 'gamma':[2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)]}

# Number of neurons in the hidden layer for Neural nets
neural_nets_parameters = {'hidden_layer_sizes':[10, 20, 30, 40]}

# Random Forest parameters
random_forest_parameters = {'max_features':[10, 15, 20, 25], 'n_estimators':[100, 200, 300, 400]}

# Parameters for Gradient Boosting Machine
gbm_parameters ={'learning_rate':[0.1, 0.05], 'max_depth':[5], 'n_estimators':[30, 70, 100]}


# -------------------------- Here goes the magic -----------------------------

# Define the external K-Fold Stratified
external_skf = StratifiedKFold(n_splits = n_external_folds)
external_skf.get_n_splits(df_parameters, df_classes)

# Iterate over external data
for external_train_index, external_test_index in external_skf.split(df_parameters, df_classes):
    
    # Split the external training set and the external test set
    external_params_train = df_parameters.iloc[external_train_index, :] 
    external_classes_train = df_classes[external_train_index] 
    external_params_test = df_parameters.iloc[external_test_index, :]
    external_classes_test = df_classes[external_test_index]
    
    # *********************** Imputation of data ****************************
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(external_params_train)

    # Appling the imputation    
    imp_external_params_train = imp.transform(external_params_train)
    imp_external_params_test = imp.transform(external_params_test)
    
    # Scaling the data
    scaler = StandardScaler().fit(imp_external_params_train)
    scaled_external_params_train = scaler.transform(imp_external_params_train)
    scaled_external_params_test = scaler.transform(imp_external_params_test)
    
    # Cleaning NaN for bad scaling
    # clean_external_params_train = np.nan_to_num(scaled_external_params_train)
    # clean_external_params_test = np.nan_to_num(scaled_external_params_test)
    
    # ************** Defining kNN with 80% of the variance ******************    
    
    # Applying the PCA keeping the variance over 80%
    pca = PCA(n_components = variance_percentage_pca)
    pca.fit(scaled_external_params_train)
    external_params_reduced_train = pca.transform(scaled_external_params_train)
    external_params_reduced_test = pca.transform(scaled_external_params_test)
        
    # GridSearch over the kNN parameters using a 3 KFold
    # The cv parameter is for Cross-validation
    # We find the hyperparameters here
    knn_external = KNeighborsClassifier()
    clf_knn_external = GridSearchCV(knn_external, knn_parameters, cv=n_internal_folds)
    clf_knn_external.fit(external_params_reduced_train, external_classes_train)
    
    # Getting the best hyperparameters
    knn_best_hyperparams = clf_knn_external.best_params_
    
    # Create the best kNN model
    knn_tuned = KNeighborsClassifier(n_neighbors=knn_best_hyperparams['n_neighbors'])
    knn_tuned.fit(external_params_reduced_train, external_classes_train)
    knn_tuned_score = knn_tuned.score(external_params_reduced_test, external_classes_test)
    
    # Stacking the precision
    knn_precision = knn_precision + knn_tuned_score
 
knn_precision = knn_precision/n_external_folds
print('Accuracy kNN: ', knn_precision)       
        