#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:54:05 2020

@author: jeremyjaspar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from metrics import *
import tensorflow as tf
import tensorflow.keras.layers as tkl

#%% Data Prep:

#train_base:
train_buff_clinical = pd.read_csv('x_train/features/clinical_data.csv',index_col=0)
train_buff_radiomics = pd.read_csv('x_train/features/radiomics.csv',index_col=0)
train_buff_output = pd.read_csv('x_train/output.csv').values

train_clinical = train_buff_clinical[['Nstage', 'Tstage', 'age']].values.astype(float)
train_radiomics = train_buff_radiomics.values[2:,:].astype(float)
train_base = np.empty((300,56))
train_base[:,0:53] = train_radiomics
train_base[:,53:56] = train_clinical

#train_label :
train_label = train_buff_output[:, 1]

#test_base :
test_buff_clinical = pd.read_csv('x_train/features/clinical_data.csv',index_col=0)
test_buff_radiomics = pd.read_csv('x_train/features/radiomics.csv',index_col=0)
test_buff_output = pd.read_csv('x_train/output.csv').reset_index().values

test_clinical = (test_buff_clinical.loc[test_buff_clinical['SourceDataset']=='l1',:])[['Nstage', 'Tstage', 'age']].values.astype(float)
test_radiomics = (test_buff_radiomics.iloc[2:,:]).values.astype(float)
test_radiomics= test_radiomics[test_buff_clinical['SourceDataset']=='l1',:]
test_base = np.empty((199,56))
test_base[:,0:53] = test_radiomics
test_base[:,53:56] = test_clinical

#test_label :
test_buff_clinical = pd.read_csv('x_test/features/clinical_data.csv',index_col=0)
id_patient_test_public = (test_buff_clinical.loc[test_buff_clinical['SourceDataset']=='l1',:].reset_index().values)[:,0].astype(int)
event_public = np.ones((199,1))


#%% Model :
model = tf.keras.models.Sequential()
model.add(tkl.Dense(100, input_shape=(56,), activation='relu'))
model.add(tkl.Dense(100, activation='relu'))
model.add(tkl.Dense(1, activation='relu')) 

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])


#%% Learning :
model.fit(train_base, train_label, batch_size=30, epochs=100)

#%%

test = model.predict(test_base)

#%%
test_output_public = pd.DataFrame(data = test[:,0], index = id_patient_public, columns=['SurvivalTime'], dtype=np.int64)
test_output_public.insert(1,'Event',event_public)
test_output_public.index.name = 'PatientID'

print(cindex(test_output_real_public, test_output_public))




