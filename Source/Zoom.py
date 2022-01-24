'''
Integrantes:
Amanda Fernandes de Oliveira
Guilherme Gomes Nogueira Chaves
'''

import math
import warnings
import tkinter as tk

from tkinter import ttk
from PIL import Image, ImageTk

class AutoScrollbar(ttk.Scrollbar):
    # Uma barra de scroll que se esconde se não for necessária
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Não é possível usar pack com o Widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Não é possível usar place com o Widget ' + self.__class__.__name__)

class CanvasImage:
    # Mostra a imagem e o zoom da imagem
    def __init__(self, root, path):
        # Inicializa o frame
        self.imscale = 1.0  # Escala prom zoom na imagem
        self.__delta = 1.3  # Magnitude do zoom
        self.__filter = Image.NORMAL
        self.__previous_state = 0  # Ultimo estado do teclado
        self.path = path
        self.root = root
        # Cria o frame na root
        self.__imframe = ttk.Frame(root)  # root
        # Scrollbar vertical e horizontal
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Cria o canvas e coloca a scrollbar nele
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # Espera o canvas ser criado
        hbar.configure(command=self.__scroll_x)  # Coloca a scrollbar
        vbar.configure(command=self.__scroll_y)
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # Canvas é redimensionado
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # Lembra da posição do cursor
        self.canvas.bind('<B1-Motion>',     self.__move_to)  # Move para a nova posição
        self.canvas.bind('<MouseWheel>', self.__wheel)  # Zoom
        self.canvas.bind('<Double-Button-1>', self.__Roi)
        self.canvas.bind('<KeyPress>', self.__keyPressEvent)

        self.__isRoi = False
        self.x = 0
        self.y = 0
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self.__rectangle = []
        self.verifica = False
        self.backup = None
        self.temp22 = []

        # Vê se a imagem é muito grande
        self.__huge = False 
        self.__huge_size = 14000  # Define o tamanho de uma imagem grande
        self.__band_width = 1024  # largura do tile
        Image.MAX_IMAGE_PIXELS = 1000000000  # não mostra o erro DecompressionBombError
        with warnings.catch_warnings():  # ignora DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)
        self.imwidth, self.imheight = self.__image.size  # pega o tamanho
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
           self.__image.tile[0][0] == 'raw':  # image tem que ser raw
            self.__huge = True
            self.__offset = self.__image.tile[0][2]  # Ofsset inicial
            self.__tile = [self.__image.tile[0][0],
                           [0, 0, self.imwidth, 0],  # cria um tile (retangulo)
                           self.__offset,
                           self.__image.tile[0][3]]  # lista de argumento pro decoder
        self.__min_side = min(self.imwidth, self.imheight)  # pega o menor lado
        # Cria uma piramide para a imagem
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Seta o coeficiente da piramide
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # Imagem atual da piramide
        self.__scale = self.imscale * self.__ratio  # Escala da imagem
        self.__reduction = 2  # Reduz o grau da imagem
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # Imagem no topo da piramide tem 512 pixels
            w /= self.__reduction  # divide pelo grau de redução
            h /= self.__reduction  # divide pelo grau de redução
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Coloca a imagem em um container e usa para definir as coordenadas da imagem
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()
        self.canvas.focus_set()  # Coloca o foco no canvas

    def smaller(self):
        # Redimensiona a imagem proporcionalemente e retorna a menor imagem
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2 # Igual a 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # taxa de compressão
            w = int(w2)  # comprimento
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # taxa de compressão
            w = int(w2)  # comprimento
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # taxa de compressão
            w = int(h2 * aspect_ratio1)  # comprimento
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)
        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # Comprimento do tile
            self.__tile[1][3] = band  # define o comprimento do tile
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # offset do tile (3 bits por pixel)
            self.__image.close()
            self.__image = Image.open(self.path)
            self.__image.size = (self.imwidth, band)  # define o tamanho do tile
            self.__image.tile = [self.__tile]  # atribui o tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # corta o tile
            image.paste(cropped.resize((w, int(band * k)+1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30*' ' + '\r', end='')  # Esconde o print
        return image

    def redraw_figures(self):
        # Função Dummy para redesenhar a figura nas classes filhas
        pass

    def grid(self, **kw):
        self.__imframe.grid(**kw)
        self.__imframe.grid(sticky='nswe')
        self.__imframe.rowconfigure(0, weight=1)
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        raise Exception('Não é possível usar o pack com este widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise Exception('Não é possível usar o place com este widget ' + self.__class__.__name__)

    def __scroll_x(self, *args, **kwargs):
        # Scroll horizontalmente e redesenha a imagem
        self.canvas.xview(*args)  # scroll horizontal
        self.__show_image()

    def __scroll_y(self, *args, **kwargs):
         # Scroll verticalmente e redesenha a imagem
        self.canvas.yview(*args)  # scroll vertical
        self.__show_image()

    def __show_image(self):
        # Mostra a imagem no canvas
        box_image = self.canvas.coords(self.container)  # pega a imagem
        box_canvas = (self.canvas.canvasx(0),  # paga a área visivel do canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # Converte para int
        # Pega a região de scroll
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Parte horizontal da imagem que é visivel
        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]
        # Parte vertical da imagem que é visivel
        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Converte a região do scroll em uma tupla e int
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # Define a região de scroll
        x1 = max(box_canvas[0] - box_image[0], 0)  # pega as coordenadas x1, x2, y1, y2
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # Mostra a imagem se ela está na area visivel
            if self.__huge and self.__curr_img < 0:  # Mostra imagem grande
                h = int((y2 - y1) / self.imscale)  # altura do tile
                self.__tile[1][3] = h  # define a altura do tile
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)
                self.__image.size = (self.imwidth, h)  # Define o tamanho do tile
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # mostra a imagem original
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # Corta a imagem atual da piramide
                                    (int(x1 / self.__scale), int(y1 / self.__scale),
                                     int(x2 / self.__scale), int(y2 / self.__scale)))

            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                                max(box_canvas[1], box_img_int[1]),
                                                anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # Coloca a imagem no background
            self.canvas.imagetk = imagetk  # Mantem uma referencia extra para prevenir o garbage-collector
            self.backup = image.resize((int(x2 - x1), int(y2 - y1)))

    def __move_from(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()

    def outside(self, x, y):
        # Verifica se o ponto (x, y) está fora da area da imagem
        bbox = self.canvas.coords(self.container)  # pega a area da imagem
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # Ponto esta dentro
        else:
            return True  # Ponto esta fora

    def __wheel(self, event):
        x = self.canvas.canvasx(event.x)  # Pega as coordenadas do evento
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # Só da zoom dentro da imagem
        scale = 1.0
        if event.delta == -120:  # Tira zoom
            if round(self.__min_side * self.imscale) < 30: return  # Image é menor que 30 pixels
            self.imscale /= self.__delta
            scale        /= self.__delta
        if event.delta == 120:  # Zoom
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel é maior que a area visivel
            self.imscale *= self.__delta
            scale        *= self.__delta
        # Pega a imagem apropriada da piramide
        k = self.imscale * self.__ratio  # Coeficiente temporario
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #

        self.temp22.append(scale)
        self.canvas.scale('all', x, y, scale, scale)  # re-escala todos os objetos
        # Redesenha algumas imagens antes de mostrar
        self.redraw_figures()  # metodo para classes filhas
        self.__show_image()
    
    def __Roi(self, event):
        x = self.canvas.canvasx(event.x)  # Pega as coordenadas do evento
        y = self.canvas.canvasy(event.y)

        box_image = self.canvas.coords(self.container)  # pega a imagem
        box_canvas = (self.canvas.canvasx(0),  # paga a área visivel do canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        x1 = max(box_canvas[0] - box_image[0], 0)  # pega as coordenadas x1, x2, y1, y2
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]

        scale = self.imscale * self.__ratio
        self.x1 = x - (64 * scale)
        self.y1 = y - (64 * scale)
        self.x2 = x + (64 * scale)
        self.y2 = y + (64 * scale)

        self.x = (x-box_image[0])/scale
        self.y = (y-box_image[1])/scale
        self.__rectangle.append(self.canvas.create_rectangle(self.x1, self.y1, self.x2, self.y2, outline="blue"))
        self.canvas.tag_raise(self.__rectangle)

    def __keyPressEvent(self, event):
        if event.char == 'r' or event.char == 'R':
            for i in self.__rectangle:
                self.canvas.delete(i)
        elif event.char == 'c' or event.char == 'C':
            for i in self.__rectangle:
                self.canvas.delete(i)

            scale = self.imscale * self.__ratio
            #Pegar o x, y do canvas pra resolver essa ultima parte
            image = self.crop((int(self.x-64), int(self.y-64), int(self.x+64), int(self.y+64)))
            image.save('temp.png')
            self.verifica = True
            self.destroy()
            canvas = MainWindow(self.root, 'temp.png')

    def crop(self, bbox):
        # Corta um retângulo da imagem e o retorna
        if self.__huge:  # Imagem é muito grande e não está totalmente na RAM
            band = bbox[3] - bbox[1]  # Comprimento do tile
            self.__tile[1][3] = band  # Define a altura do tile
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # Atribui o offset do tile
            self.__image.close()
            self.__image = Image.open(self.path) 
            self.__image.size = (self.imwidth, band)  # Coloca o size do tile
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # Imagem esta na RAM
            print('oi', self.backup)
            #self.__pyramid[0] = self.backup
            for i in self.__pyramid:
                print(i)
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        #Destructor
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # Fecha todas as imagens na pirâmide
        del self.__pyramid[:]  # Deleta a lista de pirâmides
        del self.__pyramid  # Deleta a variavel piramide
        self.canvas.destroy()
        self.__imframe.destroy()

class MainWindow(ttk.Frame):
    def __init__(self, mainframe, path):
        # Inializa o frame
        ttk.Frame.__init__(self, master=mainframe)
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.canvas = CanvasImage(self.master, path)
        self.canvas.grid(row=0, column=0)  # show widget

