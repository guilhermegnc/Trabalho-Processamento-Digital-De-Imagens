'''
Integrantes:
Amanda Fernandes de Oliveira
Guilherme Gomes Nogueira Chaves
'''

import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def quantize(image_raw, levels):
    if len(image_raw.shape) > 2: # Se tiver 3 canais
        gray2D = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY) # Grayscale 2 canais
        gray_raw = cv2.cvtColor(gray2D, cv2.COLOR_GRAY2BGR) # 3 canais
    else:
        gray_raw = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2BGR) # 3 canais
    
    if (levels < 0) and (levels > 256): # Tons de cinza
        print("Quantidade de tons precisa ser positiva e maior que 255")

    image = np.array(gray_raw, dtype=np.float64) / 255
    h, w, d = image.shape
    image_array = np.reshape(image, (h * w, d))

    plt.subplot(1, 2, 1)
    plt.clf
    plt.title('Histograma imagem original')
    hist = np.histogram(image_raw, bins=np.arange(0,256))
    plt.plot(hist[1][:-1], hist[0], lw = 2)

    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=levels).fit(image_array_sample)
    labels = kmeans.predict(image_array)

    image_out = np.zeros((h, w, d))
    label_idx = 0
    for i in range(h):
        for j in range(w):
            image_out[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            label_idx += 1

    image_out = np.array(image_out * 255, dtype=np.uint8)

    plt.subplot(1, 2, 2)
    plt.clf
    plt.title('Histograma imagem quantizada')
    hist2 = np.histogram(image_out, bins=np.arange(0,256))
    plt.plot(hist2[1][:-1], hist2[0], lw = 2)


    return image_out
    
