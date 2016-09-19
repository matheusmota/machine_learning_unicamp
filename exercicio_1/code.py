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

# input_file = "./Documents/machine_learning_unicamp/exercicio_1/data.csv"



# -------------------- When data is available in internet --------------------

df = pd.read_csv('http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/data1.csv')

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
ncolumns = len(df.columns)



# ------------------ Getting the PCA in the training set ---------------------

ncolumns_without_class = ncolumns -1

# Removing the column 'clase' 
df_without_class = df.iloc[:, 0:ncolumns_without_class]

# Getting the training set from the first 200 lines
dt_training_set = df_without_class[0:200]

# Applying the PCA
pca = PCA(n_components= ncolumns_without_class)
pca.fit(dt_training_set)
# >>> PCA(copy=True, n_components=166, whiten=False)

# Getting the cumulative variance
variance_acum = pca.explained_variance_ratio_.cumsum()

# Finding the number of components to keep the variance over 80%
ncomp = 0
var_max = 0.8

for i in range(0, ncolumns_without_class):
    if(variance_acum[i] >= var_max):
        ncomp = i + 1 # For this data set ncomp = 12
        break
    
# Applying the dimensionality reduction based on the variance
pca = PCA(n_components= ncomp)
pca.fit(dt_training_set)
dt_training_set_reduced = pca.transform(dt_training_set)
