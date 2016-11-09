# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:33:33 2016

@description: This file contains the solution to the sixth exercise list in
              Machine Learning subject at UNICAMP, this work is about text
              processing.
@author: Juan Sebastián Beleño Díaz
@email: jsbeleno@gmail.com
"""

# Loading the libraries
import numpy as np
import pandas as pd
import os
import zipfile
import urllib.request


# ------------------------------- Getting Data -------------------------------

url_zip_data = 'http://www.ic.unicamp.br/%7Ewainer/cursos/2s2016/ml/ex6/file-sk.zip'
filepath_zip = '../assets/file-sk.zip'
dirpath_zip = '../assets'
# dirpath_zip = '../assets/file-sk'

# Upload the zip file
urllib.request.urlretrieve(url_zip_data, filepath_zip)

# Creating the directory where I'll put the uncompressed files
# if not os.path.exists(dirpath_zip):
#    os.makedirs(dirpath_zip)

# Uncompress the zip files in a directory
zip_ref = zipfile.ZipFile(filepath_zip, 'r')
zip_ref.extractall(dirpath_zip)
zip_ref.close()