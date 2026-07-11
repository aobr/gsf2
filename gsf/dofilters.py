# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2018--, Aura Obreja and the GalacticStructureFinder (gsf) contributors.

import numpy as np

def apply_filters(dataframe,filters):    
    propertiesList = list(dataframe.keys())
    filter_index = list(np.arange(1,len(filters)))
    for i in filter_index:
        if filters[i]['type']=='BandPass':
            keep = np.where((dataframe[filters[i]['property']]>filters[i]['LowLimit']) & (dataframe[filters[i]['property']]<filters[i]['UpLimit']))
        if filters[i]['type']=='LowPass':
            keep = np.where(dataframe[filters[i]['property']]<filters[i]['Limit'])
        if filters[i]['type']=='HighPass':
            keep = np.where(dataframe[filters[i]['property']]>filters[i]['Limit'])
        print('Applying filter: ',filters[i])
        for prop in propertiesList: 
            try:
                dataframe[prop] = dataframe[prop][keep]
            except:
                print('Property %s is not a valid particle array.'%prop)
                pass
    return
