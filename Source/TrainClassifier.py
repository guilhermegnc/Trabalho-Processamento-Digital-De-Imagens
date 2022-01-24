'''
Integrantes:
Amanda Fernandes de Oliveira
Guilherme Gomes Nogueira Chaves
'''

from tkinter import *
from tkinter import filedialog
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import moments_hu, shannon_entropy
import cv2
import numpy as np
from PIL import ImageTk, Image
import os
import sys
import random
import time
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Treinamento():
    def __init__(self, levels, opcaoContraste, opcaoEnergia,opcaoHomogeneidade, opcaoEntropia, opcaoHu):    
        
        self.imagensTreinamento = [] 
        self.features = []
        self.labels = []
        self.caracteristicaContraste = []
        self.caracteristicaHomogeneidade = []
        self.caracteristicaEnergia = []
        self.caracteristicaEntropia = []
        self.caracteristicaHu = []
        self.y_true = []

        self.matrizConfusao = None
        self.modelosTreinados = []

        if levels == 1:
            self.levels = 16
            self.div = 16
        else:
            self.levels = 32
            self.div = 8

        self.opcaoContraste = opcaoContraste
        self.opcaoEnergia = opcaoEnergia
        self.opcaoHomogeneidade = opcaoHomogeneidade
        self.opcaoEntropia = opcaoEntropia
        self.opcaoHu = opcaoHu

        self.leArquivos()


    def leArquivos(self):
        count = 0

        path = filedialog.askdirectory(title='Selecione o diretório com as texturas')
        # Se realmente tiver o diretório
        if type(path) is str and path != '':
            # Loop para cada item encontrado
            for item in os.listdir(path):
                self.imagensTreinamento.append(list())
                # Caso seja um subdiretorio
                if os.path.isfile(os.path.join(path, item)) is False:
                    for img in os.listdir(os.path.join(path, item)):
                        if img.endswith('.png'):
                            caminhoImagem = os.path.join(path, item) + '/' + img
                            tmp = (caminhoImagem, cv2.imread(caminhoImagem, cv2.IMREAD_GRAYSCALE))
                            self.imagensTreinamento[count].append(tmp)
                        
                    count += 1

            # Inicializa o treina do classificador de imagens
            self.treinamento()

        else:
            return

    def treinamento(self):

        contraste = []
        homogeneidade = []
        energia = []
        entropia = []
        hu = []

        svmContraste = None
        svmHomogeneidade = None
        svmEnergia = None
        svmEntropia = None
        svmHu = None

        start = time.time()

        for textura in self.imagensTreinamento: # Para cada array
            random.shuffle(textura) # Embaralha õ conteudo de textura
            for img in textura[:round(len(textura)*0.75)]: # 75% das imagens para o treinamento
                im = img[1]
                data = np.array((im/self.div), 'int') # Divide os valores de cinza da imagem
                g = greycomatrix(data, [1, 2, 4, 8, 16], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=self.levels, normed=True, symmetric=True) # Calcula a matrix de co-ocorrência do nível de cinza da imagem.
                
                if self.opcaoContraste.get():               
                    contraste = greycoprops(g, 'contrast') # Calcula o contraste da matrix de co-ocorrência de níveis de cinza
                    contraste = [sum(i) for i in contraste]
                    self.caracteristicaContraste.append(contraste)
                    
                if self.opcaoHomogeneidade.get():
                    homogeneidade = greycoprops(g, 'homogeneity') # Calcula a homogeneidade da matrix de co-ocorrência de níveis de cinza
                    homogeneidade = [sum(i) for i in homogeneidade]
                    self.caracteristicaHomogeneidade.append(homogeneidade)
                if self.opcaoEnergia.get():
                    energia = greycoprops(g, 'energy') # Calcula a energia da matrix de co-ocorrência de níveis de cinza
                    energia = [sum(i) for i in energia]
                    self.caracteristicaEnergia.append(energia)
                if self.opcaoEntropia.get():
                    entropia = shannon_entropy(data) # Calcula a entropia de Shannon da imagem
                    self.caracteristicaEntropia.append(entropia)
                if self.opcaoHu.get():
                    hu = moments_hu(data) # Calcula os movimentos de Hu da imagem
                    self.caracteristicaHu.append(hu)

                self.labels.append(self.imagensTreinamento.index(textura))
                
         
        # Treinamento da SVM
        if self.opcaoContraste.get():           
            svmContraste = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=42, max_iter=1000000))
            svmContraste.fit(self.caracteristicaContraste, self.labels)
        
        if self.opcaoHomogeneidade.get(): 
            svmHomogeneidade = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=9, max_iter=100000))
            svmHomogeneidade.fit(self.caracteristicaHomogeneidade, self.labels)
        
        if self.opcaoEnergia.get(): 
            svmEnergia = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=9, max_iter=100000))
            svmEnergia.fit(self.caracteristicaEnergia, self.labels)

        if self.opcaoEntropia.get(): 
            svmEntropia = LinearSVC(random_state=9, max_iter=100000)
            svmEntropia.fit(np.array(self.caracteristicaEntropia).reshape(-1, 1), self.labels)

        if self.opcaoHu.get(): 
            svmHu = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=9, max_iter=100000))
            svmHu.fit(self.caracteristicaHu, self.labels)

        X = [list()]*5
        y_pred1 = []
        y_pred2 = []
        y_pred3 = []
        y_pred4 = []
        y_pred5 = []
        y_correct = []

        # Teste dos modelos
        for textura in self.imagensTreinamento:
            for img in textura[round(len(textura)*0.75):]: # 25% das imagens
                im = img[1] 
                data = np.array((im/self.div), 'int') 
                g = greycomatrix(data, [1, 2, 4, 8, 16], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=self.levels, normed=True, symmetric=True) 
                
                if self.opcaoContraste.get():    
                    contraste = greycoprops(g, 'contrast') 
                    contraste = [sum(i) for i in contraste]
                    X[0].append(contraste)
                    y_pred1.append(svmContraste.predict(np.array(contraste).reshape(1,-1))[0])
                if self.opcaoHomogeneidade.get():
                    homogeneidade = greycoprops(g, 'homogeneity') 
                    homogeneidade = [sum(i) for i in homogeneidade]
                    X[1].append(homogeneidade)
                    y_pred2.append(svmHomogeneidade.predict(np.array(homogeneidade).reshape(1, -1))[0])
                if self.opcaoEnergia.get():
                    energia = greycoprops(g, 'energy')
                    energia = [sum(i) for i in energia]
                    X[2].append(energia)
                    y_pred3.append(svmEnergia.predict(np.array(energia).reshape(1, -1))[0])
                if self.opcaoEntropia.get():
                    entropia = shannon_entropy(data) 
                    X[3].append(entropia)
                    y_pred4.append(svmEntropia.predict(np.array(entropia).reshape(1,-1))[0])
                if self.opcaoHu.get():
                    hu = moments_hu(data) 
                    X[4].append(hu)
                    y_pred5.append(svmHu.predict(np.array(hu).reshape(1, -1))[0])

                y_correct.append(self.imagensTreinamento.index(textura))

        end = time.time()
        executionTime = end-start
                
        # Calculo da acuracia, erro, sensibilidade, especificidade e matriz de confusão
        if self.opcaoContraste.get(): 
            acuracia = "{:.2}".format(metrics.accuracy_score(y_correct, y_pred1))        
            cm = confusion_matrix(y_correct, y_pred1, labels=np.unique(y_correct))
            print(cm)
            self.matrizConfusao = cm
            self.y_true = y_correct
            erro = (cm[1,0]+cm[0,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
            sensibilidade = cm[0,0]/(cm[0,0]+cm[0,1])
            especificidade = cm[1,1]/(cm[1,0]+cm[1,1])
            caracteristica = "Contraste"

            self.plotConfusionMatrix(acuracia, erro, sensibilidade, especificidade, caracteristica, executionTime)

        if self.opcaoHomogeneidade.get():
            acuracia = "{:.2}".format(metrics.accuracy_score(y_correct, y_pred2))
            cm = confusion_matrix(y_correct, y_pred2, labels=np.unique(y_correct))
            self.matrizConfusao = cm
            self.y_true = y_correct
            erro = (cm[1,0]+cm[0,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
            sensibilidade = cm[0,0]/(cm[0,0]+cm[0,1])
            especificidade = cm[1,1]/(cm[1,0]+cm[1,1])
            caracteristica = "Homogeneidade"

            self.plotConfusionMatrix(acuracia, erro, sensibilidade, especificidade, caracteristica, executionTime)

        if self.opcaoEnergia.get():
            acuracia = "{:.2}".format(metrics.accuracy_score(y_correct, y_pred3))
            cm = confusion_matrix(y_correct, y_pred3, labels=np.unique(y_correct))
            self.matrizConfusao = cm
            self.y_true = y_correct
            erro = (cm[1,0]+cm[0,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
            sensibilidade = cm[0,0]/(cm[0,0]+cm[0,1])
            especificidade = cm[1,1]/(cm[1,0]+cm[1,1])
            caracteristica = "Energia"

            self.plotConfusionMatrix(acuracia, erro, sensibilidade, especificidade, caracteristica, executionTime)

        if self.opcaoEntropia.get():
            acuracia = "{:.2}".format(metrics.accuracy_score(y_correct, y_pred4))
            cm = confusion_matrix(y_correct, y_pred4, labels=np.unique(y_correct))
            self.matrizConfusao = cm
            self.y_true = y_correct
            erro = (cm[1,0]+cm[0,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
            sensibilidade = cm[0,0]/(cm[0,0]+cm[0,1])
            especificidade = cm[1,1]/(cm[1,0]+cm[1,1])
            caracteristica = "Entropia"

            self.plotConfusionMatrix(acuracia, erro, sensibilidade, especificidade, caracteristica, executionTime)

        if self.opcaoHu.get():
            acuracia = "{:.2}".format(metrics.accuracy_score(y_correct, y_pred5))
            cm = confusion_matrix(y_correct, y_pred5, labels=np.unique(y_correct))
            self.matrizConfusao = cm
            self.y_true = y_correct
            erro = (cm[1,0]+cm[0,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
            sensibilidade = cm[0,0]/(cm[0,0]+cm[0,1])
            especificidade = cm[1,1]/(cm[1,0]+cm[1,1])
            caracteristica = "Hu"

            self.plotConfusionMatrix(acuracia, erro, sensibilidade, especificidade, caracteristica, executionTime)
        
        # Armazena os modelos
        self.modelosTreinados.append(svmContraste)
        self.modelosTreinados.append(svmHomogeneidade)
        self.modelosTreinados.append(svmEnergia)
        self.modelosTreinados.append(svmEntropia)
        self.modelosTreinados.append(svmHu)
        

    def plotConfusionMatrix(self, acuracia, erro, sensibilidade, especificidade, caracteristica, tempoExecucao):
        cm = self.matrizConfusao
        somaCm = np.sum(cm, axis=1, keepdims=True)
        percCm = cm / somaCm.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        rows, cols = cm.shape
        for i in range(rows):
            for j in range(cols):
                perc = percCm[i, j]
                cmValues = cm[i, j]
                if i == j:
                    soma = somaCm[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (perc, cmValues, soma)
                elif cmValues == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (perc, cmValues)

        cm = pd.DataFrame(cm, index=np.unique(self.y_true), columns=np.unique(self.y_true))
        textures = np.array([i+1 for i in np.unique(self.y_true)])
        cm.index = textures
        cm.columns = textures
        cm.index.name = 'Textura Verdadeira'
        cm.columns.name = 'Textura Prevista'
        fig, (ax, ax2) = plt.subplots(2, figsize=(10,10))
        ax2.axis([0, 10, 0, 10])
        ax2.axis("off")
        ax2.text(0, 8, 'Característica: {}'.format(caracteristica))
        ax2.text(0, 7, 'Acurácia: {:.2f}%'.format(float(acuracia)*100))
        ax2.text(0, 6, 'Taxa de erro: {:.2f}%'.format(erro*100))
        ax2.text(0, 5, 'Sensibilidade: {:.4f}'.format(sensibilidade))
        ax2.text(0, 4, 'Especificidade: {:.4f}'.format(especificidade))
        ax2.text(0, 3, 'Tempo de execução: {:.6f}s'.format(tempoExecucao))
        sns.heatmap(cm, cmap= "viridis", annot=annot, fmt='', ax=ax)
        plt.show()
