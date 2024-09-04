import os

def cant_imagenes(carpeta):
    print(f"Revisando la carpeta: {os.path.abspath(carpeta)}")
    if not os.path.exists(carpeta):
        print(f"-->Error: La carpeta {carpeta} no existe.")
        return 0
    return len([nombre for nombre in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, nombre))])

def main():
    carpetas = {
    #    '10': '../data/Billete/10',
    #    '20': '../data/Billete/20', 
        
       'cats': '../data/cats',
       'dogs': '../data/dogs', 
    }

    for categoria, carpeta in carpetas.items():
        cant = cant_imagenes(carpeta)
        print(f"-> Cantidad de imagenes en {categoria}: {cant}")

if __name__ == '__main__':
    main()