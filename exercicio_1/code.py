# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:42:41 2016

@description: This file contains the solution to the first exercise list in
              Machine Learning subject at UNICAMP, the python boilerplate was
              taken from http://stackoverflow.com/a/30813195
@author: Juan Sebastián Beleño Díaz
@email: jsbeleno@gmail.com
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# input_file = "./Documents/machine_learning_unicamp/exercicio_1/data.csv"



# -------------------- When data is available in internet --------------------

df = pd.read_csv('http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/data1.csv')

# If the professor deletes the file, still there is copy in GitHub
# df = pd.read_csv('https://raw.githubusercontent.com/jbeleno/machine_learning_unicamp/master/exercicio_1/data.csv')

# saving data in local 
# df.to_csv(input_file, sep=',', encoding='utf-8')



# ---------------------- When data is available in local----------------------

# comma delimited is the default
# df = pd.read_csv(input_file, header = 0)

# for space delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = " ")

# for tab delimited use:
# df = pd.read_csv(input_file, header = 0, delimiter = "\t")



# Getting the dataframe sizes

# In numpy
# df.shape[0] is the number of rows in the dataframe
# df.shape[1] is the number of columns in the dataframe

# In pandas
ncolumns = len(df.columns) # 167 columns



# ------------------ Getting the PCA in the training set ---------------------

ncolumns_without_class = ncolumns - 1 # 166 columns
ntraining_rows = 200
ntest_rows = 276

# Removing the column 'clase' 
df_without_class = df.iloc[:, 0:ncolumns_without_class]

# Getting the training set from the first 200 lines
df_training_set = df_without_class[0:ntraining_rows]

# Applying the PCA
pca = PCA(n_components= ncolumns_without_class)
pca.fit(df_training_set)
# >>> PCA(copy=True, n_components=166, whiten=False)

# Getting the cumulative variance
variance_acum = pca.explained_variance_ratio_.cumsum()

# Finding the number of components to keep the variance over 80%
ncomp = 0
var_max = 0.8

for i in range(0, ncolumns_without_class):
    if(variance_acum[i] >= var_max):
        ncomp = i + 1 # For this training data set ncomp = 12
        break
    
# Applying the dimensionality reduction based on the variance
pca = PCA(n_components= ncomp)
pca.fit(df_training_set)
df_training_set_reduced = pca.transform(df_training_set) # Array != Data Frame



# ---------------- Logistic Regression over the training set -----------------

# Getting the 'clase' result for the training data
results_training_set = df.iloc[0:ntraining_rows,ncolumns_without_class:ncolumns]
results_training_set = np.ravel(results_training_set) # convert column vector to vector

# setting up the regression models with and without PCA
model_with_pca = LogisticRegression().fit(df_training_set_reduced, results_training_set)
model_without_pca = LogisticRegression().fit(df_training_set, results_training_set)

# Getting the test data frame
df_test_set = df.iloc[ntraining_rows: (ntest_rows + ntraining_rows), 0:ncolumns_without_class]

results_test_set = df.iloc[ntraining_rows: (ntest_rows + ntraining_rows),ncolumns_without_class:ncolumns]
results_test_set = np.ravel(results_test_set) # convert column vector to vector

# Getting the model accuracy in the test data
pca = PCA(n_components= ncomp)
pca.fit(df_training_set)
df_test_set_reduced = pca.transform(df_test_set)

model_with_pca.score(df_test_set_reduced, results_test_set) # 0.80072
model_without_pca.score(df_test_set, results_test_set) # 0.79710
