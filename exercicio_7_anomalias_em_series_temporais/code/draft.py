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
        q75, q25 = np.percentile(ts[i:i + N], [75 ,25])
        iqr = q75 - q25
        x = ts[i:i + N]
        data_clean = np.where(np.logical_and(x>=q25, x<=q75))
        std.append(np.std(data_clean))        
        
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
            if(mean[j] <= (mean[i] + std[i] * p_limit) and mean[j] >= (mean[i] - std[i] * p_limit)):
                counter = counter + 1
            
                
        match_vector.append(counter)
    
    return match_vector


            
#descriptor = get_descriptor(ts4, 2)
#mean = descriptor[0]
#std = descriptor[1]

#match_descriptor = match_descriptors(mean, std)
ts = ts1
fft_simetric_vector = np.fft.fft(ts)
L = len(fft_simetric_vector)
fft_vector = fft_simetric_vector[2:L/2] # Gambiarra
magnitude = [math.sqrt(x.real**2 + x.imag**2) for x in fft_vector]
index_max_fft = np.argmax(magnitude)
sample_rate = 1 # Hz
max_freq = (index_max_fft + 2)*sample_rate/len(ts)
wavelength = math.floor(sample_rate/max_freq)

print('Maximum Frequency:', max_freq)
print('Wavelength:', wavelength)

descriptor = get_descriptor(ts, wavelength)
mean = descriptor[0]
std = descriptor[1]

#match_descriptor = match_descriptors(mean, std)

plt.plot(std)
plt.ylabel('Descritor')
plt.show()

# Juan you should try with the median and count how many points are above the
# median before falling. You should do a mean of the number of points in 
# each subsequence and select it as N. Maybe using the standard deviation to 
# increase it.