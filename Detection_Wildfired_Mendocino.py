# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 20:45:45 2021

@author:  M.I. Sandra Paola Hernandez Lopez 
Universidad Politecnica de Juventino Rosas
Maestria en ingenieria Sistemas Inteligentes
Proyect: Wildfires detection from multispectral image through tensorial techniques and artificial intelligence
"""

import numpy as np
from numpy import load
import function
from scipy.stats import entropy
import matplotlib.pyplot as plt
from math import log2
from tensorly.decomposition import non_negative_tucker
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy.io
import scipy.io as sio
from sklearn.svm import SVC
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D,Flatten, Input
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import keras
from keras.regularizers import l2
import imbalance_metrics as im
from sklearn.metrics import confusion_matrix, classification_report
import analysis_data as ad
import Redes_neuronales as Rn
import pandas as pd
import time
from matplotlib import pyplot
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from fitter import Fitter, get_common_distributions, get_distributions
import numpy as np
#%%
dataset = load('tensor_Wildfire_left4.npy') #Se coloca las imagenes en tensor


ground_truth = scipy.io.loadmat('grounth_fire_left5.mat') # Ground truth de las imagenes con incendio
gt = ground_truth['csv'] #Se selecciona el archivo csv que es el archivo en tensor

height = dataset.shape[0]# se coloca en poscion el acho
width = dataset.shape[1]# se coloca en poscion el largo
bands = dataset.shape[2]# se coloca en poscion las bandas
pixels= height*width# se coloca el total de pixeles
num_bins = 2**16 # Escala de las imagenes lansat 
num_class=4 #numero de clases que cuenta la imagen 
#%% NTD 
bands=11 #se coloca el numero de banda con los que se va a trabajar 
tucker_rank = [height, width, bands] # se manda hablar la funcion de tucker_rank que es ancho, largo y el numero de bandas para formar el tensor
g_ntd, _  = non_negative_tucker(np.float32(dataset), rank = tucker_rank, init='svd', tol=10e-5)# se tendra el nuevo tensor con las matrices de proyeccion 
#%% Data analysis
# #Entropy
#entropy_band = ad.entropy_2(dataset, pixels, bands, num_bins) # Entropia de las imagenes originales 
Entropy_NTD_band =ad.entropy_2(g_ntd, pixels, bands, num_bins)# Entropia del nuevo tensor 

# #Mutual information
#mutual_infor_dataset =ad.mutual_inf(dataset, bands) # Se manda hablar la libreria de mutual information , donde se estara colocando el tensor con el numero de bandas.
mutual_infor_g_ntd =ad. mutual_inf(g_ntd, bands)

# #Box and whiskers
#box_whiskers_dataset = ad.box_whiskers(dataset, pixels, bands) # para el diagrama de cajas y bigotes se necesita la base de datos, los pixeles y la banda.
box_whiskers_dataset = ad.box_whiskers(g_ntd, pixels, bands)
#%%RF
##Para realizar Random Forest a nuestra base de datos se particiono en dos base de datos, esta particion se realizo sin aleatoridad.
## para la particion de datos se realizo mediante el programa base de datos.
#inicio = time.time()
x_train = pd.read_csv('dataset_train.csv',delimiter=',', header=None)# teniendo los datos de entrenamiento en un archivo .csv se manda hablar y se cambia el formato para realizar RF
x_test = pd.read_csv('dataset_test.csv',delimiter=',', header=None)# teniendo los datos de testing en un archivo .csv se manda hablar y se cambia el formato para realizar RF
#x_train = pd.read_csv('g_ntd11_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd11_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd9_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd9_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd8_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd8_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd7_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd7_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd6_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd6_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd4_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd4_test.csv',delimiter=',', header=None)

y_train = pd.read_csv('gt_train.csv',delimiter=',', header=None)
y_test = pd.read_csv('gt_test.csv',delimiter=',', header=None)

x_train= x_train.iloc[:,:].values
x_test= x_test.iloc[:,:].values
y_train= y_train.iloc[:,:].values
y_test= y_test.iloc[:,:].values

inicio = time.time()
clf=RandomForestClassifier(n_estimators=20)
#Train the model using the training sets y_pred=clf.predict(X_test)
RandomF= clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
#y_test =  y_test.argmax(axis=1)
y_pred = y_pred.argmax(axis=1)

Cm = confusion_matrix(y_test,y_pred)
print(Cm)
# Performance evaluation
multiclass_metrics = im.report(y_test, y_pred)   
print(classification_report(y_test,y_pred))

fin = time.time()
print(fin-inicio)
#%%
inicio = time.time()
clf=RandomForestClassifier(n_estimators=20)# Se coloca el numero de arboles con los que se estaran trabajando
#Train the model using the training sets y_pred=clf.predict(X_test)
RandomF= clf.fit(x_train,y_train)#Se colocan los arreglos de entrenamiento
# y_pred=clf.predict(x_test)#x_train
# y_test = y_test.argmax(axis=1)
# y_pred = y_pred.argmax(axis=1)

y_pred1=clf.predict(x_train)
y_test1=y_train.argmax(axis=1)
y_pred1=y_pred1.argmax(axis=1)

# Cm = confusion_matrix(y_test,y_pred)
# print(Cm)
Cm = confusion_matrix(y_test1,y_pred1)
print(Cm)

# Performance evaluation
# multiclass_metrics = im.report(y_test, y_pred)   
# print(classification_report(y_test,y_pred))

multiclass_metrics = im.report(y_test1, y_pred1)   
print(classification_report(y_test1,y_pred1))


fin = time.time()
print(fin-inicio)


#%%

#model = ExtraTreesClassifier()
#model = ExtraTreesClassifier(criterion='gini', n_estimators=100, random_state=0)
model= ExtraTreesClassifier(criterion='entropy', n_estimators=100, random_state=0)
model.fit(x_train,y_train)
print(model.feature_importances_)#entropy
#list(model.feature_importances_)

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
#%%KNN
inicio = time.time()
x_train = pd.read_csv('dataset_train.csv',delimiter=',', header=None)
x_test = pd.read_csv('dataset_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd11_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd11_test.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd9_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd8_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd8_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd7_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd7_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd6_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd6_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd4_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd4_test.csv',delimiter=',', header=None)


y_train = pd.read_csv('gt_train.csv',delimiter=',', header=None)
y_test = pd.read_csv('gt_test.csv',delimiter=',', header=None)


x_train= x_train.iloc[:,:].values
x_test= x_test.iloc[:,:].values
y_train= y_train.iloc[:,:].values
y_test= y_test.iloc[:,:].values

#knn = OneVsRestClassifier(KNeighborsClassifier())
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)
#knn.predict(x_test[0].reshape(1,-1))
# y_pred = knn.predict(x_test)

#y_test =  y_test.argmax(axis=1)
#y_pred = y_pred.argmax(axis=1)

#Cm = confusion_matrix(y_test,y_pred)
#print(Cm)

# Performance evaluation
# multiclass_metrics = im.report(y_test, y_pred)   
# print(classification_report(y_test,y_pred))
knn.predict(x_train[0].reshape(1,-1))
# y_pred = knn.predict(x_test)
y_pred1=knn.predict(x_train)
y_test1=y_train.argmax(axis=1)
y_pred1=y_pred1.argmax(axis=1)

Cm = confusion_matrix(y_test1,y_pred1)
print(Cm)
multiclass_metrics = im.report(y_test1, y_pred1)   
print(classification_report(y_test1,y_pred1))

fin = time.time()
print(fin-inicio)
#%%CNN
bands=11
num_class=4
inicio = time.time()
#x_train = pd.read_csv('dataset_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('dataset_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd11_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd11_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd9_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd9_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd8_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd8_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd7_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd7_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd6_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd6_test.csv',delimiter=',', header=None)
x_train = pd.read_csv('g_ntd4_train.csv',delimiter=',', header=None)
x_test = pd.read_csv('g_ntd4_test.csv',delimiter=',', header=None)
y_train = pd.read_csv('gt_train.csv',delimiter=',', header=None)
y_test = pd.read_csv('gt_test.csv',delimiter=',', header=None)

x_train= x_train.iloc[:,:].values
x_test= x_test.iloc[:,:].values
y_train= y_train.iloc[:,:].values
y_test= y_test.iloc[:,:].values

inicio = time.time()
x_train =np.reshape(x_train,(-1,1,bands))
x_test =np.reshape(x_test,(-1,1,bands))
y_train =np.reshape(y_train,(-1,1,num_class))
y_test =np.reshape(y_test,(-1,1,num_class))

model=Rn.cnn(bands,8,num_class)
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=15, batch_size=50)

# Performance evaluation
# y_pred = model.predict(x_test)

# y_test =np.reshape(y_test,(-1,num_class))
# y_pred =np.reshape(y_pred,(-1,num_class))
# y_test = y_test.argmax(axis=1)
# y_pred = y_pred.argmax(axis=1)
y_pred = model.predict(x_train)

y_test =np.reshape(y_train,(-1,num_class))
y_pred =np.reshape(y_pred,(-1,num_class))
y_test = y_train.argmax(axis=1)
y_pred = y_pred.argmax(axis=1)

Cm = confusion_matrix(y_test,y_pred)
print(Cm)

# Performance evaluation
multiclass_metrics = im.report(y_test, y_pred)   
print(classification_report(y_test,y_pred))

fin = time.time()
print(fin-inicio)
#%% SVM
inicio = time.time()

#x_train = pd.read_csv('datasetSVM11_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('datasetSVM11_test.csv',delimiter=',', header=None)
# x_train = pd.read_csv('datasetSVM9_train.csv',delimiter=',', header=None)
# x_test = pd.read_csv('datasetSVM9_test.csv',delimiter=',', header=None)
# x_train = pd.read_csv('datasetSVM8_train.csv',delimiter=',', header=None)
# x_test = pd.read_csv('datasetSVM8_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('datasetSVM7_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('datasetSVM7_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('datasetSVM6_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('datasetSVM6_test.csv',delimiter=',', header=None)
x_train = pd.read_csv('datasetSVM4_train.csv',delimiter=',', header=None)
x_test = pd.read_csv('datasetSVM4_test.csv',delimiter=',', header=None)

y_train = pd.read_csv('gt_trainSVM.csv',delimiter=',', header=None)
y_test = pd.read_csv('gt_testSVM.csv',delimiter=',', header=None)

x_train= x_train.iloc[:,:].values
x_test= x_test.iloc[:,:].values
y_train= y_train.iloc[:,:].values
y_test= y_test.iloc[:,:].values


scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

#inicio = time.time()
svclassifier = SVC(kernel='linear')
svm_classifier = svclassifier.fit(x_train, y_train)
#y_pred= svclassifier.predict(x_test)
y_pred1= svclassifier.predict(x_train)
#y_test =  y_test.argmax(axis=1)
#y_pred = y_pred.argmax(axis=1)

# Cm = confusion_matrix(y_test,y_pred)
# print(Cm)

# # Performance evaluation
# multiclass_metrics = im.report(y_test, y_pred)   
# print(classification_report(y_test,y_pred))
Cm = confusion_matrix(y_test1,y_pred1)
print(Cm)
multiclass_metrics = im.report(y_test1, y_pred1)   
print(classification_report(y_test1,y_pred1))

fin = time.time()
print(fin-inicio)
#%%FCN
bands=4
num_class=4
inicio = time.time()
#x_train = pd.read_csv('dataset_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('dataset_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd11_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd11_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd9_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd9_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd8_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd8_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd7_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd7_test.csv',delimiter=',', header=None)
#x_train = pd.read_csv('g_ntd6_train.csv',delimiter=',', header=None)
#x_test = pd.read_csv('g_ntd6_test.csv',delimiter=',', header=None)
x_train = pd.read_csv('g_ntd4_train.csv',delimiter=',', header=None)
x_test = pd.read_csv('g_ntd4_test.csv',delimiter=',', header=None)
y_train = pd.read_csv('gt_train.csv',delimiter=',', header=None)
y_test = pd.read_csv('gt_test.csv',delimiter=',', header=None)



x_train= x_train.iloc[:,:].values
x_test= x_test.iloc[:,:].values
y_train= y_train.iloc[:,:].values
y_test= y_test.iloc[:,:].values

inicio = time.time()
x_train =np.reshape(x_train,(-1,1,bands))
x_test =np.reshape(x_test,(-1,1,bands))
y_train =np.reshape(y_train,(-1,1,num_class))
y_test =np.reshape(y_test,(-1,1,num_class))


model=Rn.fcn(bands,8,num_class)

model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

# Performance evaluation
# y_pred = model.predict(x_test)

# y_test =np.reshape(y_test,(-1,num_class))
# y_pred =np.reshape(y_pred,(-1,num_class))
# y_test = y_test.argmax(axis=1)
# y_pred = y_pred.argmax(axis=1)
y_pred1= model.predict(x_train)

y_test1 =np.reshape(y_train,(-1,num_class))
y_pred1=np.reshape(y_pred1,(-1,num_class))
y_test1 = y_test1.argmax(axis=1)
y_pred1= y_pred1.argmax(axis=1)

Cm = confusion_matrix(y_test1,y_pred1)
print(Cm)

# Performance evaluation

multiclass_metrics = im.report(y_test1, y_pred1)   
print(classification_report(y_test1,y_pred1))

fin = time.time()
print(fin-inicio)
