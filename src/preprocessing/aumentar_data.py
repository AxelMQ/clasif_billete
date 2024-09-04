from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import matplotlib.pyplot as plt

def aumentar_datos(X, y):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=15,
        zoom_range=[0.7, 1.4],
        horizontal_flip=True,
        vertical_flip=True
    )
    datagen.fit(X)
    
    print(f'--> Visualizando datos aumentados...')
    # Visualizar las imágenes aumentadas
    plt.figure(figsize=(10, 10))
    for images, etiqueta in datagen.flow(X, y, batch_size=10):
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(images[i].reshape(224, 224), cmap="gray")
            plt.title(f'Etiqueta: {etiqueta[i]}')
        break
    plt.show()

    # Generador de datos para el entrenamiento
    return datagen.flow(X, y, batch_size=32)
