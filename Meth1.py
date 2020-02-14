#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Methode using a CNN neural network and use as features the 
sum (normalized) of all the masks for eachs patient. So each patient will 
be characterized by 1 image with size 92x92x1. We use all the examples from x_train 
for the training et all the examples from x_test for the testing the algorithme (we don't deparate 
dataset public and dataset private)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
import tensorflow as tf
import tensorflow.keras.layers as tkl
import os

#%% Data Prep :

#train_base :
train_base = np.empty((300, 92, 92))
basepath = 'x_train/images/'
i = 0
for file in sorted(os.listdir(basepath)):
    scan = np.load(os.path.join(basepath,file))['mask']
    train_base[i,:,:] = scan.sum(axis=2)/92
    i += 1

    
#test_base :
test_base = np.empty((125, 92, 92))
basepath = 'x_test/images/'
i = 0
for file in sorted(os.listdir(basepath)):
    scan = np.load(os.path.join(basepath,file))['mask']
    test_base[i,:,:] = scan.sum(axis=2)/92
    i += 1
    

#train_label :
train_output = pd.read_csv('x_train/output.csv', index_col=0).reset_index().values
train_label = train_output[:, 1]
id_patient_train = train_output[:, 0]
event_train = train_output[:,2]

#test_label : Performances
test_output_real = pd.read_csv('x_test/output.csv', index_col=0)
buff = test_output_real.reset_index().values
id_patient = buff[:, 0]
event = buff[:,2]


#For the CNN:
train_base=train_base.reshape(len(train_base),train_base.shape[1], train_base.shape[2], 1)
test_base=test_base.reshape(len(test_base),test_base.shape[1], test_base.shape[2], 1)


#%% Model Training :

model = tf.keras.models.Sequential()
model.add(tkl.Conv2D(10, (20, 20), activation='relu', input_shape=(92, 92, 1)))
model.add(tkl.MaxPool2D(pool_size=(4, 4)))
model.add(tkl.Flatten(data_format='channels_last'))
model.add(tkl.Dense(100, activation='relu'))
model.add(tkl.Dense(20, activation='relu'))
model.add(tkl.Dense(1, activation='relu'))

model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


#%% Learning :
model.fit(train_base, train_label, batch_size=30, epochs=200)

#%%Performance :

test = model.predict(test_base)
test_output = pd.DataFrame(data = test[:,0], index = id_patient, columns=['SurvivalTime'], dtype=np.int64)
test_output.insert(1,'Event',event)
test_output.index.name = 'PatientID'
print(cindex(test_output_real, test_output))

#%%Performance :

test = model.predict(train_base)
test_output = pd.DataFrame(data = test[:,0], index = id_patient_train, columns=['SurvivalTime'], dtype=np.int64)
test_output.insert(1,'Event',event_train)
test_output.index.name = 'PatientID'
print(cindex(train_output, test_output))

#%%
train_output = pd.read_csv('x_train/output.csv', index_col=0)
test_output_real = pd.read_csv('x_test/output.csv', index_col=0)

print(cindex(train_output, train_output))
print(cindex(test_output_real, test_output_real))



