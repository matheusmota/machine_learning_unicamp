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

# ------------------------ Getting and Cleaning Data -------------------------

# Reading the file with the raw data
url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data'
df = pd.read_csv(url_data, header = 0, delimiter = " ")