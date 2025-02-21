# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:53:57 2021

@author: sandy
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D,Flatten, Input
#%%
def cnn(embed_size, recurrent_units, classes):
    input_layer = Input(shape = (1,embed_size))
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(input_layer)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = Dense(10, activation="relu")(x)
    x = Dense(classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

def fcn(embed_size, recurrent_units, classes):
    input_layer = Input(shape = (1,embed_size))
    x = Dense(recurrent_units, activation='relu')(input_layer)
    x=Dropout(0.2)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model