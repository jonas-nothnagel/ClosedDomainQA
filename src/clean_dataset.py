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


def split_at_length(dataframe, column, length):
    wrapped = []
    for i in dataframe[column]:
        wrapped.append(wrap(i, length))

    dataframe = dataframe.assign(wrapped=wrapped)
    dataframe['wrapped'] = dataframe['wrapped'].apply(lambda x: ', '.join(map(str, x)))

    splitted = pd.concat([pd.Series(row['PIMS_ID'], row['wrapped'].split(", "), )              
                        for _, row in dataframe.iterrows()]).reset_index()

    splitted = splitted.rename(columns={"index": "text", 0: "PIMS_ID"})
    
    return dataframe, splitted