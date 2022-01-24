'''
Integrantes:
Amanda Fernandes de Oliveira
Guilherme Gomes Nogueira Chaves
'''

from TrainClassifier import Treinamento
from ResizeRegion import resizeImage
from Zoom import MainWindow
from Equalization import equalize
from Quantization import quantize
from Save import _save_file_dialogs
from Classifier import Classifier
import easygui
import unicodedata
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import cv2 as cv2
import matplotlib.pyplot as plt
import os

def pathImage():
    global filepath, enteredImage, baseImage, hasBaseImage, app
    baseROI = messagebox.askquestion("", "A imagem é uma região de interesse?")
    uni_img = easygui.fileopenbox()
    try:
        filepath = unicodedata.normalize('NFKD', uni_img).encode('ascii','ignore')
    except:
        return
    filepath = filepath.decode('utf-8')
    hasBaseImage = False
    enteredImage = True
    if baseROI == 'yes':
        hasBaseImage = True
        baseImage = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)   
    app = MainWindow(group1, filepath)

def pathSave():
     global app
     verificaBaseImage(app)
     if not hasBaseImage:
        messagebox.showerror("Erro", "Abra uma região de interesse primeiro")
     else:
        path = _save_file_dialogs()
        if path != None:
            result=cv2.imwrite(path, baseImage)
            if result==True:
                messagebox.showinfo("Salvo", "Arquivo Salvo")
            else:
                messagebox.showerror("Erro", "Erro ao salvar")

def pathTexture():
    global optionVar, opcaoContraste, opcaoEnergia, opcaoHomogeneidade, opcaoEntropia, opcaoHu, modeloTreinado
    if (not opcaoContraste.get()) and (not opcaoEnergia.get()) and (not opcaoEntropia.get()) and (not opcaoHomogeneidade.get()) and (not opcaoHu.get()):
        messagebox.showerror("Erro", "Selecione uma característica pelo menos")
        return
    modelos = Treinamento(optionVar, opcaoContraste, opcaoEnergia, opcaoHomogeneidade, opcaoEntropia, opcaoHu)
    modeloTreinado = modelos.modelosTreinados

def pathResize():
    global app
    verificaBaseImage(app)
    if not hasBaseImage:
        messagebox.showerror("Erro", "Abra uma região de interesse primeiro")
    else:
        scale = easygui.enterbox("Qual a escala será aplicada na imagem (%)?")
        if scale != None:
            global baseImage
            imageResized = resizeImage(baseImage, int(scale))
            baseImage = imageResized
            result=cv2.imwrite('temp.png', baseImage)
            app = MainWindow(group1, 'temp.png')

def pathQuantize():
    global app
    verificaBaseImage(app)
    if not hasBaseImage:
        messagebox.showerror("Erro", "Abra uma região de interesse primeiro")
    else:
        levels = easygui.enterbox("Qual a quantidade de tons de cinza aplicada na imagem (tipo int)?")
        if levels != None:
            global baseImage
            imageQuantized = quantize(baseImage, int(levels))
            baseImage = imageQuantized
            result=cv2.imwrite('temp.png', baseImage)
            app = MainWindow(group1, 'temp.png')
            plt.show()
         
def pathEqualize():
    global app, optionVar2
    verificaBaseImage(app)
    if not hasBaseImage:
        messagebox.showerror("Erro", "Abra uma região de interesse primeiro")
    else:
        global baseImage
        if optionVar2.get() == "Numpy":
            imageEqualized = equalize(baseImage, 'numpy')
        elif optionVar2.get() == "OpenCV":
            imageEqualized = equalize(baseImage, 'opencv')
        else:
            imageEqualized = equalize(baseImage, 'clahe')

        baseImage = imageEqualized
        result=cv2.imwrite('temp.png', baseImage)
        app = MainWindow(group1, 'temp.png')
        plt.show()

def pathSair():
    master_window.destroy()
    temp = cv2.imread('temp.png')
    try:
        if temp.size != 0:
            os.remove('temp.png')
    except:   
        print('A imagem temporaria não foi apagada')

def pathClassify():
    if not enteredImage:
        messagebox.showerror("Erro", "Abra uma imagem primeiro")
        return
    if modeloTreinado == None:
        messagebox.showerror("Erro", "Treine o classificador primeiro")
        return
    if optionVar.get() == 1:
        levels = 16
    else:
        levels = 32

    if hasBaseImage:
        if len(baseImage.shape) > 2:
            gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
            Classifier(group1, levels, modeloTreinado, gray)
        else:
            Classifier(group1, levels, modeloTreinado, baseImage)
    else:
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        Classifier(group1, levels, modeloTreinado, image)


def verificaBaseImage(app):
    if app.canvas.verifica:
        global hasBaseImage, baseImage
        hasBaseImage = True
        baseImage = cv2.imread('temp.png')
        app.canvas.verifica = False


filepath = 'KMabWEXvea4P6QSXqDM6.png'
baseImage = cv2.imread('KMabWEXvea4P6QSXqDM6.png', cv2.IMREAD_UNCHANGED)
enteredImage = False
hasBaseImage = False
changeZoom = True
app = None
modeloTreinado = None

master_window = tk.Tk()
master_window.title("Menu")

menubar = Menu(master_window)

opcoesFile = Menu(menubar, tearoff=0)
opcoesFile.add_command(label='Abrir', command=pathImage)
opcoesFile.add_command(label="Salvar", command=pathSave)
opcoesFile.add_command(label="Sair", command=pathSair)
menubar.add_cascade(label="File", menu=opcoesFile)

opcaoEntropia = BooleanVar()
opcaoEntropia.set(False)
opcaoHomogeneidade = BooleanVar()
opcaoHomogeneidade.set(False)
opcaoEnergia = BooleanVar()
opcaoEnergia.set(False)
opcaoContraste = BooleanVar()
opcaoContraste.set(True)
opcaoHu =  BooleanVar()
opcaoHu.set(False)

optionVar = tk.IntVar(menubar)

niveisCinza = Menu(menubar, tearoff=0)
niveisCinza.add_radiobutton(label="16 Níveis", value=1, variable=optionVar)
niveisCinza.add_radiobutton(label="32 Níveis", value=2, variable=optionVar)
optionVar.set(2)
Caracteristicas = Menu(menubar, tearoff=0)
Caracteristicas.add_checkbutton(label='Contraste', variable=opcaoContraste, onvalue=True, offvalue=False)
Caracteristicas.add_checkbutton(label='Homogeneidade', variable=opcaoHomogeneidade, onvalue=True, offvalue=False)
Caracteristicas.add_checkbutton(label='Energia', variable=opcaoEnergia, onvalue=True, offvalue=False)
Caracteristicas.add_checkbutton(label='Entropia', variable=opcaoEntropia, onvalue=True, offvalue=False)
Caracteristicas.add_checkbutton(label='Momentos de Hu', variable=opcaoHu, onvalue=True, offvalue=False)
opcoesTreinamento = Menu(menubar, tearoff=0)
opcoesTreinamento.add_cascade(label='Selecionar Características', menu=Caracteristicas)
opcoesTreinamento.add_cascade(label='Selecionar Níveis de Cinza', menu=niveisCinza)
opcoesTreinamento.add_command(label="Treinar", command=pathTexture)
opcoesTexturas = Menu(menubar, tearoff=0)
opcoesTexturas.add_command(label="Classificar", command=pathClassify)
opcoesTexturas.add_cascade(label="Treinamento", menu=opcoesTreinamento)
menubar.add_cascade(label="Texturas", menu=opcoesTexturas)

optionVar2 = tk.StringVar(menubar)

opcoes = Menu(menubar, tearoff=0)
opcoes.add_command(label="Redimensionar", command=pathResize)
opcoes.add_command(label="Quantizar", command=pathQuantize)
opcoesEqualizar = Menu(menubar, tearoff=0)
opcoesEqualizar.add_radiobutton(label="Numpy", value="Numpy", variable=optionVar2, command=pathEqualize)
opcoesEqualizar.add_radiobutton(label="OpenCV", value="OpenCV", variable=optionVar2, command=pathEqualize)
opcoesEqualizar.add_radiobutton(label="Clahe", value="Clahe", variable=optionVar2, command=pathEqualize)
opcoes.add_cascade(label="Equalização", menu=opcoesEqualizar)
menubar.add_cascade(label="Opções", menu=opcoes)

master_window.config(menu=menubar)

# Frame Group1 ----------------------------------------------------
group1 = tk.LabelFrame(master_window, text="Imagem", padx=5, pady=5)
group1.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky='ewns')

master_window.columnconfigure(0, weight=1)
master_window.rowconfigure(0, weight=1)

group1.rowconfigure(0, weight=1)
group1.columnconfigure(0, weight=1)

# Cria o Canvas da imagem
app = MainWindow(group1, filepath)

master_window.mainloop()
