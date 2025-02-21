# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:11:20 2021

@author: sandy
"""
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from math import log2
import matplotlib.pyplot as plt
#%%
def entropy_2(dataset, pixels, bands, num_bins):
    vectorBand = np.zeros((pixels,bands))
    entropy_of_each_band = []
    for band_to_analyze in range(0, bands):
        vectorBand[:,band_to_analyze] = dataset[:,:,band_to_analyze].flatten()
        maxim = int(max(vectorBand[:,band_to_analyze])) + 1
        minin = int(min(vectorBand[:,band_to_analyze])) - 1
        class_width = (maxim - minin)/num_bins
        vectorBand[:,band_to_analyze]= vectorBand[:,band_to_analyze] - minin 
    
        h, e = np.histogram(vectorBand[:,band_to_analyze], num_bins)

        total=0
        probs = np.zeros((num_bins, 2))

        for k in range(num_bins):
            probs[k,0] = e[k]
            probs[k,1] = h[k]/pixels
         
        entropy= 0.0
        for i in range(0, num_bins):
            if(probs[i, 1] != 0):
                entropy = entropy - (probs[i, 1] * log2(probs[i, 1]))
        entropy_of_each_band.append(entropy)
    x = np.arange(1,bands+1)
    plt.figure()
    plt.bar(x, entropy_of_each_band, align='center', alpha=0.5)
    plt.title('Entropy')
    plt.ylabel('Entropy')
    plt.xlabel('Bands')
    plt.show()
    return entropy_of_each_band

def mutual_inf(dataset, bands):
    Mutual_information = np.zeros((bands,bands))
    for i in range(bands):
        for j in range(bands):
            Mutual_information[i, j] = mutual_info_score(np.float16(dataset[:,:,i]).reshape(-1), np.float16(dataset[:,:,j]).reshape(-1))
    plt.imshow(Mutual_information, cmap= 'jet', vmin= np.min(Mutual_information), vmax= np.max(Mutual_information))
    plt.colorbar()
    plt.show()
    return Mutual_information

def box_whiskers(dataset,pixels,bands):
    boxes_img = np.reshape(dataset, (pixels, bands))
    plt.figure()
    plt.boxplot(boxes_img, showfliers=False)
    plt.show()
    