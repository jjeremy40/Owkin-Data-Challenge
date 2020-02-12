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

pd.options.display.max_rows = 10

#%% TIPS:

archive = np.load('x_test/images/patient_000.npz')
scan = archive['scan']
mask = archive['mask']
#scan.shape equals mask.shape

#%% TIPS:
train_output = pd.read_csv('x_train/output.csv', index_col=0)
p0 = train_output.loc[202]
print(p0.Event) #0 or 1
print(p0.SurvivalTime) #time to event or time to last known alive in days

for i in range(92):
    plt.figure()
    plt.imshow(scan[:,:,i])
    plt.figure()
    plt.imshow(mask[:,:,i])
    plt.pause(0.5)


#%% Pandas :

#chargement fichier: 
train_output = pd.read_csv('x_train/output.csv', index_col=0)
#dimensions :
print(train_output.shape)
#afficher premier lignes du jeu :
print(train_output.head())
#afficher derniere lignes du jeu :
print(train_output.tail())
#enumeration des colonnes :
print(train_output.columns)
#accès à une colonne :
print(train_output['Event'])
print(train_output.Event)
#accès à plusieurs colonnes :
print(train_output[['SurvivalTime', 'Event']])
#comptage des valeurs:
print(train_output['Event'].value_counts())
#certaines valeurs d'une colonne :
print(train_output['Event'][0:3])
#accès valeur située en (0,0) (vue matricielle):
print(train_output.iloc[0,0])
#valeur derniere ligne, premiere colonne (shape[0] = nb ligne, shape[1] = nb colonne)
print(train_output.iloc[train_output.shape[0]-1,0])
#5 premieres lignes et colonne 0, 1 et 4 :
print(train_output/iloc[0:5, [0,2,4]])
#liste individues avec Event 1 :
print(train_output.loc[train_outpu['Event']==1, :])
#scission des données selon Event:
g = train_output.groupby('Event')
#création d'une série de 0 de la même longueur :
code = pd.Series(np.zeros(train_output.shape[0]))



#%% Data prep :
train_output = pd.read_csv('x_train/output.csv', index_col=0)
train_label = train_output['SurvivalTime']





#%% MLP :

MLP_model = tf.keras.models.Sequential()
MLP_model.add(tkl.Dense(100, activation='relu', input_shape=()))






#%% Convolutional network :











