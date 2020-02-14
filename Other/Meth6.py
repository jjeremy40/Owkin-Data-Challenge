#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""CNN : 
Train : sum of masks on the axis=2 then normalized for patient Event=1 (features 92x92, 162 patients)
Test_public : sum of masks on the axis=2 for patient Public dataset (features 92x92, 85 patients)
Test_private : sum of masks on the axis=2 for patient Private dataset (features 92x92, 40 patients)
result C-index = 0.47 on public and 0.49 on private
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
train_buff_output = pd.read_csv('x_train/output.csv',index_col=0)
id_patient_train = (train_buff_output.loc[train_buff_output['Event']==1,:].reset_index().values)[:,0].astype(int)

basepath = 'x_train/images/patient_'
train_base = np.empty((162, 92, 92))

#loading images in the same order than PatientID:
for i in range(162):
    if (id_patient_train[i]-10) < 0:
        scan = np.load(basepath + '00' + str(id_patient_train[i]) + '.npz')['mask']
        train_base[i,:,:] = scan.sum(axis=2)/92
    elif (id_patient_train[i]-100) < 0:
        scan = np.load(basepath + '0' + str(id_patient_train[i]) + '.npz')['mask']
        train_base[i,:,:] = scan.sum(axis=2)/92
    else :
        scan = np.load(basepath + str(id_patient_train[i]) + '.npz')['mask']
        train_base[i,:,:] = scan.sum(axis=2)/92

train_label = (train_buff_output.loc[train_buff_output['Event']==1,:].reset_index().values)[:,1].astype(float)

del train_buff_output, basepath, i, scan, id_patient_train, 

# Test :
test_buff_clinical = pd.read_csv('x_test/features/clinical_data.csv',index_col=0)
id_patient_test_public = (test_buff_clinical.loc[test_buff_clinical['SourceDataset']=='l1',:].reset_index().values)[:,0].astype(int)
id_patient_test_private = (test_buff_clinical.loc[test_buff_clinical['SourceDataset']=='l2',:].reset_index().values)[:,0].astype(int)

basepath = 'x_test/images/patient_'
test_base_public = np.empty((85, 92, 92))
test_base_private = np.empty((40, 92, 92))

for i in range(85):
    if (id_patient_test_public[i]-10) < 0:
        scan = np.load(basepath + '00' + str(id_patient_test_public[i]) + '.npz')['mask']
        test_base_public[i,:,:] = scan.sum(axis=2)/92
    elif (id_patient_test_public[i]-100) < 0:
        scan = np.load(basepath + '0' + str(id_patient_test_public[i]) + '.npz')['mask']
        test_base_public[i,:,:] = scan.sum(axis=2)/92
    else :
        scan = np.load(basepath + str(id_patient_test_public[i]) + '.npz')['mask']
        test_base_public[i,:,:] = scan.sum(axis=2)/92

for i in range(40):
    if (id_patient_test_private[i]-10) < 0:
        scan = np.load(basepath + '00' + str(id_patient_test_private[i]) + '.npz')['mask']
        test_base_private[i,:,:] = scan.sum(axis=2)/92
    elif (id_patient_test_private[i]-100) < 0:
        scan = np.load(basepath + '0' + str(id_patient_test_private[i]) + '.npz')['mask']
        test_base_private[i,:,:] = scan.sum(axis=2)/92
    else :
        scan = np.load(basepath + str(id_patient_test_private[i]) + '.npz')['mask']
        test_base_private[i,:,:] = scan.sum(axis=2)/92
    

del basepath, i, scan, test_buff_clinical

#for CNN :
train_base = train_base.reshape(len(train_base),train_base.shape[1], train_base.shape[2], 1)
test_base_public = test_base_public.reshape(len(test_base_public), test_base_public.shape[1], test_base_public.shape[2], 1)
test_base_private = test_base_private.reshape(len(test_base_private),test_base_private.shape[1], test_base_private.shape[2], 1)
    

#%% Model:

model = tf.keras.models.Sequential()
model.add(tkl.Conv2D(100, (10, 10), activation='relu', input_shape=(92, 92, 1)))
model.add(tkl.MaxPool2D(pool_size=(4, 4)))
model.add(tkl.Flatten(data_format='channels_last'))
model.add(tkl.Dense(100, activation='relu'))
model.add(tkl.Dense(20, activation='relu'))
model.add(tkl.Dense(1, activation='relu'))

model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

#%% Training :
history = model.fit(train_base, train_label, batch_size=30, epochs=50)

#%% History Plot :
# plt.plot(history.history['loss'])
# plt.plot(history.history['mse'])
# plt.show()

#%% History Plot :
plt.plot(history.history['loss'])
plt.plot(history.history['mse'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#%% Performance :
test_buff_clinical = pd.read_csv('x_test/features/clinical_data.csv',index_col=0)
test_buff_output = pd.read_csv('x_test/output.csv', index_col=0)
test_buff_output_public = test_buff_output.loc[test_buff_clinical['SourceDataset']=='l1',:]
test_buff_output_private = test_buff_output.loc[test_buff_clinical['SourceDataset']=='l2',:]
buff = test_buff_output.reset_index().values
event_public = buff[test_buff_clinical['SourceDataset']=='l1',2]
event_private = buff[test_buff_clinical['SourceDataset']=='l2',2]

del test_buff_clinical, buff, test_buff_output

#%% On public test:
test_public = model.predict(test_base_public)
test_output_public = pd.DataFrame(data = np.abs(test_public[:,0]), index = id_patient_test_public, columns=['SurvivalTime'], dtype=np.int64)
test_output_public.insert(1,'Event', event_public)
test_output_public.index.name = 'PatientID'

print(cindex(test_buff_output_public, test_output_public))

#%% On private test:
test_private = model.predict(test_base_private)
test_output_private = pd.DataFrame(data = test_private[:,0], index = id_patient_test_private, columns=['SurvivalTime'], dtype=np.int64)
test_output_private.insert(1,'Event',event_private)
test_output_private.index.name = 'PatientID'

print(cindex(test_buff_output_private, test_output_private))



