'''
Integrantes:
Amanda Fernandes de Oliveira
Guilherme Gomes Nogueira Chaves
'''

import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

def __plotHistograms__(image_raw, image_eq, cdf_orig, cdf_eq):
    plt.subplot(1,2,1)
    plt.plot(cdf_orig, color = 'b')
    plt.hist(image_raw.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('CDF','Histograma'), loc = 'upper left')
    plt.title("Histograma imagem original")

    plt.subplot(1,2,2)
    plt.plot(cdf_eq, color = 'b')
    plt.hist(image_eq.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('CDF','Histograma'), loc = 'upper left')
    plt.title("Histograma imagem equalizada")

def equalize(image_raw, type):
    if len(image_raw.shape) > 2: # Se tiver 3 canais
       gray2D = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    else:
       gray2D = image_raw

    if type == 'numpy':
        hist, bins = np.histogram(image_raw.flatten(),256,[0,256]) #Imagem fica em uma dimensão

        cdf = hist.cumsum() # Soma cumulativa sobre um eixo
        cdf_normalized = cdf * hist.max() / cdf.max()

        cdf_m = np.ma.masked_equal(cdf,0) # Aplica uma mascara no vetor quando a condição for satisfeita
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8') # Fill onde a mascara foi aplicada

        image2 = cdf[image_raw]

        hist2, bins2 = np.histogram(image2.flatten(),256,[0,256])

        cdf2 = hist2.cumsum()
        cdf_normalized2 = cdf2 * hist.max() / cdf2.max()

        __plotHistograms__(image_raw, image2, cdf_normalized, cdf_normalized2)
        return image2


    elif type == 'opencv':
        equ = cv2.equalizeHist(gray2D)
        return equ

    elif type == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray2D)
        return cl1
