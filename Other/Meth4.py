#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""MLP :
Train : concatenate radiomics and clinical data (Nstage, Tstage, age) for patient Event=1 (feature 1x57, 162 patients)
Test : concatenate radiomucs and clinical data (Nstage, Tstage, age) for patient Event=1 (feature 1x57, 162 patients)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from metrics import *
import tensorflow as tf
import tensorflow.keras.layers as tkl
import os

#%% Data Prep:

#train_base:
train_buff_radiomics = pd.read_csv('x_train/features/radiomics.csv',index_col=0)
train_buff_clinical = pd.read_csv('x_train/features/clinical_data.csv',index_col=0)

train_base_clinical = train_buff_clinical[['Nstage','Tstage','age']].values
train_buff_radiomics = train_buff_radiomics.values
train_base_radiomics = train_buff_radiomics[2:,:].astype(np.float64)

#test_base for Dataset l1 public and Dataset l2 private :
test_buff_radiomics = pd.read_csv('x_test/features/radiomics.csv',index_col=0)
test_buff_clinical = pd.read_csv('x_test/features/clinical_data.csv',index_col=0)

test_base_clinical_public = test_buff_clinical.loc[test_buff_clinical['SourceDataset']=='l1',:][['Nstage','Tstage','age']].values
test_base_clinical_private = test_buff_clinical.loc[test_buff_clinical['SourceDataset']=='l2',:][['Nstage','Tstage','age']].values

test_buff_radiomics = test_buff_radiomics.values
test_buff_radiomics = test_buff_radiomics[2:,:]
test_base_radiomics_public = test_buff_radiomics[test_buff_clinical['SourceDataset']=='l1',:].astype(np.float64)
test_base_radiomics_private = test_buff_radiomics[test_buff_clinical['SourceDataset']=='l2',:].astype(np.float64)


#train_label :
train_output = pd.read_csv('x_train/output.csv', index_col=0).reset_index().values
train_label = train_output[:, 1]

#test_label : Performances
test_output_real = pd.read_csv('x_test/output.csv', index_col=0)
test_output_real_public = test_output_real.loc[test_buff_clinical['SourceDataset']=='l1',:]
test_output_real_private = test_output_real.loc[test_buff_clinical['SourceDataset']=='l2',:]
buff = test_output_real.reset_index().values
id_patient_public = buff[test_buff_clinical['SourceDataset']=='l1', 0]
id_patient_private = buff[test_buff_clinical['SourceDataset']=='l2', 0]
event_public = buff[test_buff_clinical['SourceDataset']=='l1',2]
event_private = buff[test_buff_clinical['SourceDataset']=='l2',2]


del train_buff_clinical, train_buff_radiomics
del train_output, buff, test_buff_clinical, test_buff_radiomics, test_output_real


#%% Model :
model = tf.keras.models.Sequential()
model.add(tkl.Dense(100, input_shape=(53,), activation='relu'))
model.add(tkl.Dense(100, activation='relu'))
model.add(tkl.Dense(1, activation='relu')) 

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])


#%% Learning :
model.fit(train_base_radiomics, train_label, batch_size=30, epochs=150)


#%%Performance :
#on public test:
test = model.predict(test_base_radiomics_public)
test_output_public = pd.DataFrame(data = test[:,0], index = id_patient_public, columns=['SurvivalTime'], dtype=np.int64)
test_output_public.insert(1,'Event',event_public)
test_output_public.index.name = 'PatientID'

print(cindex(test_output_real_public, test_output_public))

#on private test:
test = model.predict(test_base_radiomics_private)
test_output_private = pd.DataFrame(data = test[:,0], index = id_patient_private, columns=['SurvivalTime'], dtype=np.int64)
test_output_private.insert(1,'Event',event_private)
test_output_private.index.name = 'PatientID'

print(cindex(test_output_real_private, test_output_private))


