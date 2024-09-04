import os
import random
import matplotlib.pyplot as plt # type: ignore
import matplotlib.image as mpimg # type: ignore
import cv2 # type: ignore

def mostrar_img_etiquetas(carpeta_main, cantidad=25, tamaño_img=100, color_gris=False):
    if not os.path.exists(carpeta_main):
        print(f"--> ERROR: La carpeta {carpeta_main}, no existe.")
        return
    
    img_paths = []
    carpetas = os.listdir(carpeta_main)
    for carpeta in carpetas:
        path_carpeta = os.path.join(carpeta_main, carpeta)
        imgs = os.listdir(path_carpeta)
        for img in imgs:
            img_paths.append((os.path.join(path_carpeta, img), carpeta))

        # Seleccion de imagenes de forma random
        img_select = random.sample(img_paths, min(cantidad, len(imgs)))

        plt.figure(figsize=(10, 10))
        for i, (img_path, etiqueta) in enumerate(img_select):
            img = mpimg.imread(img_path)
        
            # Redimensionar la imagen
            img = cv2.resize(img, (tamaño_img, tamaño_img))
            
            # Convertir a blanco y negro si se especifica
            if color_gris:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            if color_gris:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(f"Etiqueta: {etiqueta}")
            plt.axis('off')

        plt.show()

def main():
    # carpeta_dataset = '../data/Billete/'
    carpeta_dataset = '../data/'
    mostrar_img_etiquetas(carpeta_dataset, cantidad=25, tamaño_img=100, color_gris=False)

if __name__ == '__main__':
    main()
