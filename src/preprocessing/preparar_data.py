import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def cargar_datos(carpeta_main, tamaño_img=100, color_gris=False):
    img_paths = []
    etiquetas_clases = sorted(os.listdir(carpeta_main))  # Ordenar para garantizar consistencia
    etiqueta_map = {nombre: i for i, nombre in enumerate(etiquetas_clases)}

    # carpetas = os.listdir(carpeta_main)
    
    for carpeta in etiquetas_clases:
        path_carpeta = os.path.join(carpeta_main, carpeta)
        imgs = os.listdir(path_carpeta)
        for img in imgs:
            img_path = os.path.join(path_carpeta, img)
            img_paths.append((img_path, carpeta))
    
    datos_entrenamiento = []
    for img_path, etiqueta in img_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (tamaño_img, tamaño_img))
        if color_gris:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(tamaño_img, tamaño_img, 1)  # Añadir dimensión de canal
        else:
            img = img / 255.0  # Normalizar imágenes en color
        etiqueta = etiqueta_map[etiqueta]
        datos_entrenamiento.append([img, etiqueta])

    # Verificar algunas etiquetas
    # for img, etiqueta in datos_entrenamiento[:10]:
    #     print(f'Imagen: {img}, Etiqueta: {etiqueta}')
    
    return datos_entrenamiento

def procesar_datos(datos_entrenamiento):
    X = []
    y = []
    
    for imagen, etiqueta in datos_entrenamiento:
        X.append(imagen)
        y.append(etiqueta)

    X = np.array(X).astype(np.float32) / 255  # Normalizar imágenes
    y = np.array(y)
    
    print(Counter(y))
    return X, y

def mostrar_ejemplos(datos_entrenamiento, num_ejemplos=5):
    indices = random.sample(range(len(datos_entrenamiento)), num_ejemplos)
    for i in indices:
        img, etiqueta = datos_entrenamiento[i]
        # Verificar la forma de la imagen
        print(f'Forma de la imagen {i}: {img.shape}')
        if img.shape[-1] == 1:  # Grayscale
            plt.imshow(img.squeeze(), cmap='gray', interpolation='nearest')
        else:  # Color
            plt.imshow(img)
        plt.title(f'Etiqueta: {etiqueta}')
        plt.show()
