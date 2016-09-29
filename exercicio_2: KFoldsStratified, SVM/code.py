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
df_params = df.iloc[:, 0:ncolumns_without_class]

# Getting the column 'clase' from the dataset
df_result = df.iloc[:,ncolumns_without_class:ncolumns]
df_result = np.ravel(df_result) # convert a column vector to vector



# ------------------------- K-Fold Stratified ---------------------------------

# Declare important variables to use later
n_external_folds = 5
n_internal_folds = 3

gamma_values_set = [2**-15, 2**-10, 2**-5, 2**0, 2**5]
c_values_set = [2**-5, 2**-2, 2**0, 2**2, 2**5]
optimal_gamma = 0
optimal_c = 0

final_accuracy = 0

# Define the external K-Fold Stratified
external_skf = StratifiedKFold(n_splits = n_external_folds)

# Iterate over several folds to find a good accuracy in the SVM 
for external_train_index, external_test_index in external_skf.split(df_without_class, df_class):
    
    # Declare external variables
    fold_accuracy = 0
    
    # Split the external training set and the external test set
    external_params_train = df_without_class[external_train_index] 
    external_results_train = df_class[external_train_index] 
    external_params_test = df_without_class[external_test_index] 
    external_results_test = df_class[external_test_index]
    
    # Define the internal K-Fold Stratified 
    internal_skf = StratifiedKFold(n_splits = n_internal_folds)
    
    # Iterate over several internal folds
    for internal_train_index, internal_test_index in internal_skf.split(external_params_train, external_results_train):
        
        # Declare internal variables
        internal_accuracy = 0
        
        # Split the internal training set and the internal test set 
        internal_params_train = external_params_train[internal_train_index]
        internal_results_train = external_results_train[internal_train_index]
        internal_params_test = external_params_train[internal_test_index]
        internal_results_test = external_results_train[internal_test_index]
        
        # Iterate over gamma and C values to get best results in internal folds
        for gamma_value in grid_values_set:
            for c_value in c_values_set:
                
                # Set up the internal classifier
                internal_classifier = SVC(C = c_value, kernel = 'rbf', gamma = gamma_value)
                internal_classifier.fit(internal_params_train, internal_results_train)
                
                # Getting the accuracy of the internal classifier for experimental
                # values for gamma and C= 1/alpha
                temporal_accuracy = internal_classifier.score(internal_params_test, internal_results_test)
        
                # Looking for the best combination of gamma and C for 
                # internal folds
                if(temporal_accuracy > internal_accuracy):
                    internal_accuracy = temporal_accuracy
                    
                    # TO DO: Change the logic to accept better hiperparameters
                    optimal_gamma = gamma_value
                    optimal_c = c_value
        
        # Compare and update the fold accuracy
        if(internal_accuracy > fold_accuracy):
            fold_accuracy = internal_accuracy
    
    # Perform a sum over the fold accuracy
    final_accuracy = final_accuracy + fold_accuracy
                
# Divide the final_accuracy over the number of folds to get the mean accuracy
final_accuracy = final_accuracy/n_external_folds