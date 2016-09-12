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



# put the original column names in a python list
original_headers = list(df.columns.values)
