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

from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# ------------------------ Getting and Cleaning Data -------------------------

# Defining the URIs with raw data'
url_parameters = 'https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data'
url_results = 'https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data'

# Reading the files with the raw data
df_parameters = pd.read_csv(url_parameters, header = 0, delimiter = " ")
df_results = pd.read_csv(url_results, header = 0, delimiter = " ")

# Getting classes from result
df_classes = df_results.iloc[:, 0:1]


# ------------------------- Parameter declaration ----------------------------

# Number of columns and rows in the raw data
n_columns = df_parameters.shape[1]
n_rows = df_parameters.shape[0]

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


# ----------------------------- Imputing data --------------------------------



# ----------------- Principal Component Analysis( ~ Gambiarra) ---------------

# Applying the PCA
pca = PCA(n_components = variance_percentage_pca)
pca.fit(df_parameters)