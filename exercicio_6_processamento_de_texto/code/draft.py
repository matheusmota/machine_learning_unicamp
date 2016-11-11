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

# URLs with data
url_zip_data = 'http://www.ic.unicamp.br/%7Ewainer/cursos/2s2016/ml/ex6/file-sk.zip'
url_categories = 'http://www.ic.unicamp.br/%7Ewainer/cursos/2s2016/ml/ex6/category.tab'

# Local path for different files and directories
filepath_zip = '../assets/file-sk.zip'
dirpath_zip = '../assets'
directories = {'Apps': '../assets/filesk/Apps/',
               'Enterprise': '../assets/filesk/Enterprise/',
               'Gadgets': '../assets/filesk/Gadgets/',
               'Social': '../assets/filesk/Social/',
               'Startups': '../assets/filesk/Startups/'}
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

# Extracting the categories for each file
df_categories = pd.read_csv(url_categories, 
                            header= None, 
                            names= ['file', 'category'],
                            skiprows = 0,
                            delimiter = " " )

# ---------------------------- Pre-processing the data -----------------------

punctuation = ['.', ',', ';', ':', " ' ", " 's ", '?', 
               '"', '”', '“', '’s', '—', '/'
               '(', ')', '[', ']', '1', '2', '3', 
               '4', '5', '6', '7', '8', '9', '0',
               '$', '%', '–', '•', '~']
               
stop_words = [" a ", " about ", " above ", " above ", " across ", " after ", " afterwards ",
              " again ", " against ", " all ", " almost ", " alone ", " along ", " already ",
              " also ","although ","always ","am ","among ", " amongst ", " amoungst ", 
              " amount ",  " an ", " and ", " another ", " any ","anyhow ","anyone ",
              " anything ","anyway ", " anywhere ", " are ", " around ", " as ",  " at ", 
              " back ","be ","became ", " because ","become ","becomes ", " becoming ", 
              " been ", " before ", " beforehand ", " behind ", " being ", " below ", 
              " beside ", " besides ", " between ", " beyond ", " bill ", " both ", 
              " bottom ","but ", " by ", " call ", " can ", " cannot ", " cant ", " co ", 
              " con ", " could ", " couldnt ", " cry ", " de ", " describe ", " detail ", 
              " do ", " done ", " down ", " due ", " during ", " each ", " eg ", " eight ", 
              " either ", " eleven ","else ", " elsewhere ", " empty ", " enough ", " etc ", 
              " even ", " ever ", " every ", " everyone ", " everything ", " everywhere ", 
              " except ", " few ", " fifteen ", " fify ", " fill ", " find ", " fire ", 
              " first ", " five ", " for ", " former ", " formerly ", " forty ", " found ", 
              " four ", " from ", " front ", " full ", " further ", " get ", " give ", " go ", 
              " had ", " has ", " hasnt ", " have ", " he ", " hence ", " her ", " here ", 
              " hereafter ", " hereby ", " herein ", " hereupon ", " hers ", " herself ", 
              " him ", " himself ", " his ", " how ", " however ", " hundred ", " ie ", " if ", 
              " in ", " inc ", " indeed ", " interest ", " into ", " is ", " it ", " its ", 
              " itself ", " keep ", " last ", " latter ", " latterly ", " least ", " less ", 
              " ltd ", " made ", " many ", " may ", " me ", " meanwhile ", " might ", " mill ", 
              " mine ", " more ", " moreover ", " most ", " mostly ", " move ", " much ", 
              " must ", " my ", " myself ", " name ", " namely ", " neither ", " never ", 
              " nevertheless ", " next ", " nine ", " no ", " nobody ", " none ", " noone ", 
              " nor ", " not ", " nothing ", " now ", " nowhere ", " of ", " off ", " often ", 
              " on ", " once ", " one ", " only ", " onto ", " or ", " other ", " others ", 
              " otherwise ", " our ", " ours ", " ourselves ", " out ", " over ", " own ",
              " part ", " per ", " perhaps ", " please ", " put ", " rather ", " re ", " same ", 
              " see ", " seem ", " seemed ", " seeming ", " seems ", " serious ", " several ", 
              " she ", " should ", " show ", " side ", " since ", " sincere ", " six ", " sixty ", 
              " so ", " some ", " somehow ", " someone ", " something ", " sometime ", 
              " sometimes ", " somewhere ", " still ", " such ", " system ", " take ", " ten ", 
              " than ", " that ", " the ", " their ", " them ", " themselves ", " then ", 
              " thence ", " there ", " thereafter ", " thereby ", " therefore ", " therein ", 
              " thereupon ", " these ", " they ", " thickv ", " thin ", " third ", " this ", 
              " those ", " though ", " three ", " through ", " throughout ", " thru ", 
              " thus ", " to ", " together ", " too ", " top ", " toward ", " towards ", 
              " twelve ", " twenty ", " two ", " un ", " under ", " until ", " up ", " upon ", 
              " us ", " very ", " via ", " was ", " we ", " well ", " were ", " what ", 
              " whatever ", " when ", " whence ", " whenever ", " where ", " whereafter ", 
              " whereas ", " whereby ", " wherein ", " whereupon ", " wherever ", " whether ", 
              " which ", " while ", " whither ", " who ", " whoever ", " whole ", " whom ", 
              " whose ", " why ", " will ", " with ", " within ", " without ", " would ", 
              " yet ", " you ", " your ", " yours ", " yourself ", " yourselves ", " the "]

# Iterate over the directories
for directory in directories:
    # for filename in os.listdir(os.getcwd()):
    for filename in np.ravel(df_categories['file'].loc[df_categories['category'] == directory]):

        # Setting up the relative path for the file
        filename = directories[directory] + filename + '.txt'
        
        # Open the file and read the content
        with open(filename, "r") as inputFile:
            content = inputFile.read()
            
        # Open the file in writing mode
        with open(filename, "w") as outputFile:
            
            # Transform the content to lowercase
            content = content.lower()
            
            # Remove punctuation
            for char in punctuation:
                content = content.replace(char, '')
                
            # special puntuation
            content = content.replace('’re', ' are')
            content = content.replace('n’t', ' not')
            content = content.replace('s’', 's')
            content = content.replace('-', ' ')
            
            # Remove stop words
            for stop_word in stop_words:
                content = content.replace(stop_word, ' ')
                
            # Steming
            
            # write the preprocessed content
            outputFile.write(content)
        