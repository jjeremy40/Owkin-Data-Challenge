#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Same as Meth1 :
Methode using a CNN neural network and use as features the 
sum (normalized) of all the masks for eachs patient. So each patient will 
be characterized by 1 image with size 92x92x1. 
We separate the dataset public et private for the training and the testing.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
import tensorflow as tf
import tensorflow.keras.layers as tkl
import os

#%% Data Prep : 