# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:05:37 2024

@author: LENOVO
"""
import pandas as pd
df = pd.read_csv('./KNNAlgorithmDataset.csv')
df.drop(columns = ['Unnamed: 32'],inplace = True)
