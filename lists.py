#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:33:34 2019

@author: dressa94
"""

import pandas as pd

ALL_NAMES = {'PESSOA': ['parentesco', 'profissão', 'pronome'], 
             'ORGANIZAÇÃO': ['estabelecimento'],
             'LOCAL' : ['geomorfologia','logradouro'],
             'ACONTECIMENTO' : ['evento'],
             'OBRA' : ['obra'],
             'COISA' : ['objeto','coisa']}

def generate_list(path):
    dic = {}
    
    df = pd.read_excel(path)
    for column in df.columns:
        dic[column] = df[column].values
        
    return dic
    
def get_classifiers(classes=None):
            
    classifiers = {}

    if classes == None:
        classes = list(ALL_NAMES)
    
    for class_name in classes:
        for subclass in ALL_NAMES[class_name]:
            name = "|".join([class_name, subclass])
            classifiers[name] = generate_list("lists/%s.xlsx"%(subclass))
    
    return classifiers
        


