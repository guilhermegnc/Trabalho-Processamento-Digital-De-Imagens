'''
Integrantes:
Amanda Fernandes de Oliveira
Guilherme Gomes Nogueira Chaves
'''

import time
import numpy as np
from tkinter import *
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import moments_hu, shannon_entropy

def exibirClassificacao(root, message):
     matrizClassificacaoJanela = Toplevel(root) 
     matrizClassificacaoJanela.title("Classificação da Imagem")
     matrizClassificacaoJanela.geometry('500x200')
     label = Label(matrizClassificacaoJanela, text = message, font=("Arial", 12)).pack()

def Classifier(root, levels, modelos, image):
    if levels == 16:
        data = np.array((image/16), 'int')
    else:
        data = np.array((image/8), 'int')

    g = greycomatrix(data, [1, 2, 4, 8, 16], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=levels, normed=True, symmetric=True)

    predict = [-1,-1,-1,-1,-1]

    start = time.time()

    if modelos[0] != None:
        contraste = greycoprops(g, 'contrast')
        contraste = [sum(i) for i in contraste]
        predict[0] = modelos[0].predict(np.array(contraste).reshape(1,-1))[0]
    if modelos[1] != None:
        homogeneidade = greycoprops(g, 'homogeneity')
        homogeneidade = [sum(i) for i in homogeneidade]
        predict[1] = modelos[1].predict(np.array(homogeneidade).reshape(1, -1))[0]
    if modelos[2] != None:
        energia = greycoprops(g, 'energy')
        energia = [sum(i) for i in energia]
        predict[2] = modelos[2].predict(np.array(energia).reshape(1, -1))[0]
    if modelos[3] != None:
        entropia = shannon_entropy(data)
        predict[3] = modelos[3].predict(np.array(entropia).reshape(1,-1))[0]
    if modelos[4] != None:
        hu = moments_hu(data)
        predict[4] = modelos[4].predict(np.array(hu).reshape(1, -1))[0]

    end = time.time()
    executionTime = end-start

    message = ''

    for index, i in enumerate(predict):
        if i != -1:
            if index == 0:
                message += "Utilizando a característica contraste a textura prevista foi a: {}ª\n".format(predict[0]+1)
            elif index == 1:
                message += "Utilizando a característica homogeneidade a textura prevista foi a: {}ª\n".format(predict[0]+1)
            elif index == 2:
                message += "Utilizando a característica energia a textura prevista foi a: {}ª\n".format(predict[0]+1)
            elif index == 3:
                message += "Utilizando a característica entropia a textura prevista foi a: {}ª\n".format(predict[0]+1)
            elif index == 4:
                message += "Utilizando a característica hu a textura prevista foi a: {}ª\n".format(predict[0]+1)

    message += '\n Tempo de execução: {}s'.format(executionTime)
    exibirClassificacao(root, message)
