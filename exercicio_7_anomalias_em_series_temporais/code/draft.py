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

# This function finds the size of the subsequence in the time series,
# counting the number of points in sequence above the mean, and also counting
# the number of points in sequence below the mean. We calculate the max 
# sequence for points above and below the mean and based on that we 
# choose the max sequence above or the substraction between max sequence
# below and max sequence above.
def find_N(ts):
    
    mean = np.mean(ts)
    subseq_up_arr = []
    subseq_down_arr = []
    subseq_up_length = 0
    subseq_down_length = 0

    for value in ts:
        if value > mean:
            subseq_up_length = subseq_up_length + 1
            if subseq_down_length > 0:
                subseq_down_arr.append(subseq_down_length)
                subseq_down_length = 0
        else:
            subseq_down_length = subseq_down_length + 1
            if subseq_up_length > 0:
                subseq_up_arr.append(subseq_up_length)
                subseq_up_length = 0
    
    subseq_up_max = np.amax(subseq_up_arr)
    subseq_down_max = np.amax(subseq_down_arr)
    subseq_relationship = subseq_down_max / subseq_up_max
    wavelength = 0
    
    if(subseq_relationship > 3  and subseq_relationship < 5):
        wavelength = subseq_down_max - subseq_up_max
    else:
        wavelength = subseq_up_max
        
    
    return wavelength
    
# Takes the time serie and find a wavelength to execute an anomaly detection
# Finally, it plots the time serie and the anomaly
def show_anomaly(ts):
    
    ts_length = len(ts)
    wavelength = math.floor(find_N(ts))

    descriptor = get_descriptor(ts, wavelength)
    mean = descriptor[0]
    std = descriptor[1]

    match_descriptor = match_descriptors(mean, std)
    anomaly_index = np.argmin(match_descriptor)
    
    # This is hacky just to make visible the anomaly in the pĺot
    # because in some case the anomaly size is so little it can't be
    # plotted
    if(wavelength < 10):
        wavelength = 10
    

    plt.plot(range(0,ts_length), ts, '#ff5722', 
             range(anomaly_index,(anomaly_index + wavelength)),
             ts[range(anomaly_index,(anomaly_index + wavelength))],
             '#009688')
    plt.ylabel('Index')
    plt.ylabel('Value')
    plt.show()
    
# ------------------------- Testing the Algorithm ----------------------------

show_anomaly(ts1)
show_anomaly(ts2)
show_anomaly(ts3)
show_anomaly(ts4)
show_anomaly(ts5)