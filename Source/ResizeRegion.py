'''
Integrantes:
Amanda Fernandes de Oliveira
Guilherme Gomes Nogueira Chaves
'''

import cv2 as cv2

def resizeImage(image, scale):
	#image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
	scale_percent = scale
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	print('Dimensões redimensionadas : ',resized.shape)
	return resized
	
