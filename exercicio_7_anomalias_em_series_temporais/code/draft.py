# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:01:19 2016

@description: This file contains the solution to the seventh exercise list in
              Machine Learning subject at UNICAMP, this work is about dircord(anomalies)
              detection in temporal series.
@author: Juan Sebastián Beleño Díaz
@email: jsbeleno@gmail.com
"""

# Loading the libraries
import numpy as np
import pandas as pd


# ------------------------------- Getting Data -------------------------------

url_serie_1 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie1.csv'
url_serie_2 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie2.csv'
url_serie_3 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie3.csv'
url_serie_4 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie4.csv'
url_serie_5 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie5.csv'

df_serie_1 = pd.read_csv(url_serie_1, header= True)
df_serie_2 = pd.read_csv(url_serie_2, header= True)
df_serie_3 = pd.read_csv(url_serie_3, header= True)
df_serie_4 = pd.read_csv(url_serie_4, header= True)
df_serie_5 = pd.read_csv(url_serie_5, header= True)