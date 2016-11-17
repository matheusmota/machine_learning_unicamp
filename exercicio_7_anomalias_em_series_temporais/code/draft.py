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
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ------------------------------- Getting Data -------------------------------

url_serie_1 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie1.csv'
url_serie_2 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie2.csv'
url_serie_3 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie3.csv'
url_serie_4 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie4.csv'
url_serie_5 = 'http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie5.csv'

df_serie_1 = pd.read_csv(url_serie_1, header= 0)
df_serie_2 = pd.read_csv(url_serie_2, header= 0)
df_serie_3 = pd.read_csv(url_serie_3, header= 0)
df_serie_4 = pd.read_csv(url_serie_4, header= 0)
df_serie_5 = pd.read_csv(url_serie_5, header= 0)

# ------------------------- Pre-processing the data --------------------------

ts1 = np.ravel(df_serie_1['value'])
ts2 = np.ravel(df_serie_2['value'])
ts3 = np.ravel(df_serie_3['value'])
ts4 = np.ravel(df_serie_4['value'])
ts5 = np.ravel(df_serie_5['value'])

# ------------------------------ Algorithm -------- --------------------------

# Creates the descriptor (mean, standard deviation)
# ts: a vector of Time Series
# N: the length of the subsequence considered
def get_descriptor(ts, N):
    
    mean = []
    std = []
    M = len(ts)
    
    for i in range(0, M - N + 1 ):
        mean.append(np.mean(ts[i:i + N]))
        std.append(np.std(ts[i:i + N]))        
        
    return[mean, std]

# Matches each descriptor against others to find similarities based on
# pertecentage, i.e. for each descriptor we assume that there is a similarity 
# if the mean and stardard deviation does not change more than a percentage
# compared with other descriptor. Finally, we count the number of similar 
# descriptors and those with less similarities are anomalies
def match_descriptors(mean, std, p_limit = 2):
    
    K = len(mean)
    match_vector = []
    
    for i in range(0, K):
        counter = 0
        for j in range(0, K):
            p_mean = 0
            p_std = 0
            
            # I'm assming a gaussian distribution here
            # p_limit = 2 => 95% of confidence
            if(mean[j] <= (mean[i] + std[i] * p_limit) and mean[j] >= (mean[i] - std[i] * p_limit)):
                counter = counter + 1
            
                
        match_vector.append(counter)
    
    return match_vector


def find_N(ts):
    
    mean = np.mean(ts)
    subsequence_arr = []
    subsequence_length = 0

    for value in ts:
        if value > mean:
            subsequence_length = subsequence_length + 1
        elif subsequence_length > 0:
            subsequence_arr.append(subsequence_length)
            subsequence_length = 0
            
    return np.amax(subsequence_arr)
    


            

ts = ts5
wavelength = math.floor(find_N(ts))

print(wavelength)

descriptor = get_descriptor(ts, wavelength)
mean = descriptor[0]
std = descriptor[1]

match_descriptor = match_descriptors(mean, std)

plt.plot(match_descriptor)
plt.ylabel('Descritor')
plt.show()
