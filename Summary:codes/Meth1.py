#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CNN: LeNet-5
Train : 3D mask for patient Event=1 and Public dataset (features 92x92x92, 128 patients)
Test_public : 3D mask for patient Public dataset (features 92x92x92, 85 patients)
Test_private : 3D mask for patient Private dataset (features 92x92x92, 40 patients)
result C-index : train public event=1 : 0.404 on public and 0.512 on private
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
import tensorflow as tf
import tensorflow.keras.layers as tkl
import os

#%% Data Prep : 

#Train :
train_buff_clinical = pd.read_csv('x_train/features/clinical_data.csv',index_col=0)
train_buff_output = pd.read_csv('x_train/output.csv', index_col=0)

id_patient_train_public = train_buff_clinical.loc[train_buff_clinical['SourceDataset']=='l1',:]
id_patient_train_public = (id_patient_train_public.loc[train_buff_output['Event']==1,:].reset_index().values)[:,0].astype(int)

basepath = 'x_train/images/patient_'
train_base_public = np.empty((128, 92, 92, 92))

for i in range(128):
    if (id_patient_train_public[i]-10) < 0:
        scan = np.load(basepath + '00' + str(id_patient_train_public[i]) + '.npz')['mask']
        train_base_public[i,:,:,:] = scan
    elif (id_patient_train_public[i]-100) < 0:
        scan = np.load(basepath + '0' + str(id_patient_train_public[i]) + '.npz')['mask']
        train_base_public[i,:,:,:] = scan
    else :
        scan = np.load(basepath + str(id_patient_train_public[i]) + '.npz')['mask']
        train_base_public[i,:,:,:] = scan

train_label_public = train_buff_output.loc[train_buff_clinical['SourceDataset']=='l1',:]
train_label_public = (train_label_public.loc[train_buff_output['Event']==1,:].reset_index().values)[:,1].astype(float)

del train_buff_output, basepath, i, scan, id_patient_train_public, train_buff_clinical

# Test :
test_buff_clinical = pd.read_csv('x_test/features/clinical_data.csv',index_col=0)
id_patient_test_public = (test_buff_clinical.loc[test_buff_clinical['SourceDataset']=='l1',:].reset_index().values)[:,0].astype(int)
id_patient_test_private = (test_buff_clinical.loc[test_buff_clinical['SourceDataset']=='l2',:].reset_index().values)[:,0].astype(int)

basepath = 'x_test/images/patient_'
test_base_public = np.empty((85, 92, 92, 92))
test_base_private = np.empty((40, 92, 92, 92))

for i in range(85):
    if (id_patient_test_public[i]-10) < 0:
        scan = np.load(basepath + '00' + str(id_patient_test_public[i]) + '.npz')['mask']
        test_base_public[i,:,:,:] = scan
    elif (id_patient_test_public[i]-100) < 0:
        scan = np.load(basepath + '0' + str(id_patient_test_public[i]) + '.npz')['mask']
        test_base_public[i,:,:,:] = scan
    else :
        scan = np.load(basepath + str(id_patient_test_public[i]) + '.npz')['mask']
        test_base_public[i,:,:,:] = scan

for i in range(40):
    if (id_patient_test_private[i]-10) < 0:
        scan = np.load(basepath + '00' + str(id_patient_test_private[i]) + '.npz')['mask']
        test_base_private[i,:,:,:] = scan
    elif (id_patient_test_private[i]-100) < 0:
        scan = np.load(basepath + '0' + str(id_patient_test_private[i]) + '.npz')['mask']
        test_base_private[i,:,:,:] = scan
    else :
        scan = np.load(basepath + str(id_patient_test_private[i]) + '.npz')['mask']
        test_base_private[i,:,:,:] = scan
    
test_buff_output = pd.read_csv('x_test/output.csv', index_col=0)
test_label_public = (test_buff_output.loc[test_buff_clinical['SourceDataset']=='l1',:].reset_index().values)[:,1].astype(float)
test_label_private = (test_buff_output.loc[test_buff_clinical['SourceDataset']=='l2',:].reset_index().values)[:,1].astype(float)

del basepath, i, scan, test_buff_clinical

    
#%% Model LNet-5 :

model = tf.keras.models.Sequential()
model.add(tkl.ZeroPadding2D(2, input_shape=(92, 92, 92)))
model.add(tkl.Conv2D(6, (5, 5), activation='relu'))
model.add(tkl.MaxPool2D(pool_size=(2, 2)))
model.add(tkl.Conv2D(16, (5, 5), activation='relu'))
model.add(tkl.MaxPool2D(pool_size=(2, 2)))
model.add(tkl.Flatten(data_format='channels_last'))
model.add(tkl.Dense(120, activation='relu'))
model.add(tkl.Dense(84, activation='relu'))
model.add(tkl.Dense(1, activation='relu'))

model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])


#%% Training :
history = model.fit(train_base_public, train_label_public, batch_size=30, epochs=100)

#%% History Plot :
plt.plot(history.history['loss'])
plt.plot(history.history['mse'])
plt.show()


#%% Performance :
test_buff_clinical = pd.read_csv('x_test/features/clinical_data.csv',index_col=0)
test_buff_output_public = test_buff_output.loc[test_buff_clinical['SourceDataset']=='l1',:]
test_buff_output_private = test_buff_output.loc[test_buff_clinical['SourceDataset']=='l2',:]
buff = test_buff_output.reset_index().values
event_public = buff[test_buff_clinical['SourceDataset']=='l1',2]
event_private = buff[test_buff_clinical['SourceDataset']=='l2',2]

del test_buff_clinical, buff, test_buff_output

#%% On public test:
test_public = model.predict(test_base_public)
test_output_public = pd.DataFrame(data = test_public[:,0], index = id_patient_test_public, columns=['SurvivalTime'], dtype=np.int64)
test_output_public.insert(1,'Event', event_public)
test_output_public.index.name = 'PatientID'

print(cindex(test_buff_output_public, test_output_public))

#%% On private test:
test_private = model.predict(test_base_private)
test_output_private = pd.DataFrame(data = test_private[:,0], index = id_patient_test_private, columns=['SurvivalTime'], dtype=np.int64)
test_output_private.insert(1,'Event',event_private)
test_output_private.index.name = 'PatientID'

print(cindex(test_buff_output_private, test_output_private))





