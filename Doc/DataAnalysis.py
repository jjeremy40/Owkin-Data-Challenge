#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:27:25 2020

@author: jeremyjaspar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
import tensorflow as tf
import tensorflow.keras.layers as tkl

#%% TIPS:

archive = np.load('x_train/images/patient_002.npz')
scan = archive['scan']
mask = archive['mask']
#scan.shape equals mask.shape

for i in range(92):
    plt.imshow(scan[:,:,i])
    plt.pause(0.5)
    
for i in range(92):
    plt.imshow(mask[:,:,i])
    plt.pause(0.5)

#%%
train_output = pd.read_csv('x_train/output.csv', index_col=0)
p0 = train_output.loc[202]
print(p0.Event) #0 or 1
print(p0.SurvivalTime) #time to event or time to last known alive in days


#%%
train_base = np.empty((300, 92, 92))
basepath = 'x_train/images/'
i = 0
for file in sorted(os.listdir(basepath)):
    scan = np.load(os.path.join(basepath,file))['mask']
    train_base[i,:,:] = scan.sum(axis=2)/92
    i += 1

for i in range(92):
    plt.imshow(train_base[i,:,:])
    plt.pause(0.5)



