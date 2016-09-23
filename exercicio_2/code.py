# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:51:31 2016

@description: This file contains the solution to the second exercise list in
              Machine Learning subject at UNICAMP, this work is about the use  
              of k-folds, SVM, and hiper-parameters discovery using grid search
@author: Juan Sebastián Beleño Díaz
@email: jsbeleno@gmail.com
"""

# Loading the libraries
import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC

# ------------------------ Getting and Cleaning Data -------------------------

# Reading the csv file with the raw data
df = pd.read_csv('http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/data1.csv')



# ------------------------- Preparing the Data -------------------------------

ncolumns = df.shape[1] # 167 columns
ncolumns_without_class = ncolumns - 1 # 166 columns

# Removing the column 'clase' from the dataset
df_without_class = df.iloc[:, 0:ncolumns_without_class]

# Getting the column 'clase' from the dataset
df_class = df.iloc[:,ncolumns_without_class:ncolumns]
df_class = np.ravel(results_test_set) # convert a column vector to vector



# ------------------------- 