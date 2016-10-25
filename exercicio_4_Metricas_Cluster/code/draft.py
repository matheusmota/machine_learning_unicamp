# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:41:08 2016

@description: This file contains the solution to the fourth exercise list in
              Machine Learning subject at UNICAMP, this work is about the use  
              of k-means and cluster metrics.
@author: Juan Sebastián Beleño Díaz
@email: jsbeleno@gmail.com
"""

# Loading the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics

# ------------------------------- Getting Data -------------------------------

# Defining the URIs with raw data
url_parameters = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/cluster-data.csv'
url_classes = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/cluster-data-class.csv'

# Reading the files with the raw data
df_parameters = pd.read_csv(url_parameters, header = 0, delimiter = ",")
df_classes = pd.read_csv(url_classes, header = 0, delimiter = ",")

# Number of columns and rows in the raw data
n_columns = df_parameters.shape[1]
n_rows = df_parameters.shape[0]

# --------------------------- Parameter's Definition -------------------------

# k values to iterate until get a good k
k_parameters = range(2, 11)

# Number of restarts
n_restarts = 5

best_internal_score = 0 # This is for Calinski Harabaz score
best_internal_k = 2

best_external_score = -1 # This is the minimum value for adjusted rand score
best_external_k = 2

# Matrix for external and internal metrics
# First line is for k values
# Second line is for internal metrics
# Third line is for external metrics
matrix_plot = [[0]*9 for i in range(3)]

# --------------------------- Here goes the magic ----------------------------

for k in k_parameters:
    
    # Declaring the model with k clusters
    k_means_model = KMeans(k, n_init = n_restarts)
    k_means_model.fit(df_parameters)
    
    
    # Internal metric
    predicted_labels = k_means_model.labels_
    internal_score = metrics.calinski_harabaz_score(df_parameters, predicted_labels)
    
    #print('Internal score: ', k,' - ',internal_score)
    if internal_score > best_internal_score:
        best_internal_score = internal_score
        best_internal_k = k
        
    
    # External metric
    true_labels = np.ravel(df_classes)
    external_score = metrics.adjusted_rand_score(true_labels, predicted_labels)
    #print('External score: ',k,' - ',external_score) 
    if external_score > best_external_score:
        best_external_score = external_score
        best_external_k = k
        
    
    # Filling the matrix with metrics
    matrix_plot[0][k-2] = k
    matrix_plot[1][k-2] = internal_score
    matrix_plot[2][k-2] = external_score
        

# Showing the k values for different metrics
print('K for internal metric (Calinski Harabaz Index): ', best_internal_k)
print('K for external metric (Adjusted Rand): ', best_external_k)

# Plotting charts with internal and external metrics
fig_width = 512
fig_height = 380
plt.figure(figsize=(fig_width, fig_height))
plt.subplot(121)
plt.plot(matrix_plot[0], matrix_plot[1], '#ff5722', marker='o', linestyle='-')
plt.xlabel('Number of clusters')
plt.ylabel('Calinski Harabaz Index')

plt.subplot(122)
plt.plot(matrix_plot[0], matrix_plot[2], '#009688', marker='o', linestyle='-')
plt.xlabel('Number of clusters')
plt.ylabel('Adjusted Rand Index')
plt.show()