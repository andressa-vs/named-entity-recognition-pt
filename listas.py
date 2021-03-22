#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:33:34 2019

@author: dressa94
"""

import pandas as pd

ALL_NAMES = ('parentesco', 'profissao', 'pronome', 'estabelecimento', 
         'geomorfologia','logradouro','evento','obra','objeto','coisa')


def get_classifiers(lists_names=None):
    
    def generate_list(path):
        dic = {}
        
        df = pd.read_excel(path)
        for column in df.columns:
            dic[column] = df[column].values
            
        return dic
        
    classifiers = {}

    if lists_names == None:
        lists_names = ALL_NAMES
    
    for list_name in lists_names:
        
        if list_name in ('parentesco','profissao','pronome'):
            name = 'pessoa|%s'%(list_name)
        elif list_name in ('logradouro','geomorfologia'):
            name = 'local|%s'%(list_name)
        elif list_name in ('estabelecimento'):
            name = 'organização|%s'%(list_name)
        elif list_name in ('evento'):
            name = 'acontecimento|%s'%(list_name)
        elif list_name in ('objeto', 'coisa'):
            name = 'coisa|%s'%(list_name)
        elif list_name in ('obra'):
            name = 'obra|%s'%(list_name)
        
        classifiers[name] = generate_list("listas/%s.xlsx"%(list_name))
    
    return classifiers
        


