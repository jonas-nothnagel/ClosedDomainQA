#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:56:48 2020

@author: jonas 

@title: clean_dataset

@descriptions: set of functions that enable splitting and cleaning.
"""

#%%
import pandas as pd
import numpy as np
import string
from itertools import chain
from textwrap3 import wrap
import re

def split_at_length(dataframe, column, length, PIMS_ID = True):
    wrapped = []
    for i in dataframe[column]:
        wrapped.append(wrap(i, length))

    dataframe = dataframe.assign(wrapped=wrapped)
    dataframe['wrapped'] = dataframe['wrapped'].apply(lambda x: '; '.join(map(str, x)))

    if PIMS_ID == True:
        splitted = pd.concat([pd.Series(row['PIMS_ID'], row['wrapped'].split("; "), )              
                            for _, row in dataframe.iterrows()]).reset_index()
        splitted = splitted.rename(columns={"index": "text", 0: "PIMS_ID"})

    else:
        splitted = []   

    
    
    return dataframe, splitted

def basic(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # Text Lowercase
    s = s.lower() 
    # Remove punctuation
    translator = str.maketrans(' ', ' ', string.punctuation) 
    s = s.translate(translator)
    # Remove URLs
    s = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', s, flags=re.MULTILINE)
    s = re.sub(r"http\S+", " ", s)
    # Remove new line characters
    s = re.sub('\n', ' ', s) 
  
    # Remove distracting single quotes
    s = re.sub("\'", " ", s) 
    # Remove all remaining numbers and non alphanumeric characters
    s = re.sub(r'\d+', ' ', s) 
    s = re.sub(r'\W+', ' ', s)

    # define custom words to replace:
    #s = re.sub(r'strengthenedstakeholder', 'strengthened stakeholder', s)
    
    return s.strip()

def remove_linebreaks(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # Remove new line characters
    s = re.sub('\n', ' ', s) 
    
    return s.strip()