
import os
import tensorflow as tf
import subprocess
import logging
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.utils import custom_object_scope

# Configura las rutas
modelo_path = '../src/models/result/cnn_AD_pet.h5'
carpeta_salida = '../src/salida'
logging.basicConfig(filename='conversion.log', level=logging.INFO)

# Asegúrate de que la carpeta de salida exista
os.makedirs(carpeta_salida, exist_ok=True)

def save_model(model):
    """Guarda el modelo en la ruta especificada."""
    try:
        model.save(modelo_path)
        print(f'Modelo guardado en {modelo_path}')
    except Exception as e:
        print(f'Error al guardar el modelo: {e}')

def convert_to_tfjs(modelo_path, carpeta_salida):
    """Convierte el modelo guardado a TensorFlow.js."""
    if os.path.exists(modelo_path):
        try:
            # Cargar el modelo utilizando custom_object_scope para manejar la política de dtype personalizada
            # with custom_object_scope({'DTypePolicy': tf.keras.mixed_precision.Policy}):
            #     model = load_model(modelo_path)
            model = tf.keras.models.load_model(modelo_path) 
                
            if model is not None:
                try:
                    comando = [
                        'tensorflowjs_converter',
                        '--input_format=keras',
                        modelo_path,
                        carpeta_salida
                    ]
                    subprocess.run(comando, check=True)
                    print(f'Conversion successful! en: {carpeta_salida}')
                except subprocess.CalledProcessError as e:
                    print(f"Error durante la conversion a TensorFlow.js: {e}")
                    print(e.output)  # Imprime la salida completa del comando
                    print(e.stderr)  # Imprime los mensajes de error
                    # Guarda la salida en un archivo para análisis posterior
                    with open('conversion_error.log', 'w') as f:
                        f.write(str(e))
            else:
                print(f"Failed to load model from '{modelo_path}'.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
    else:
        print(f"El modelo '{modelo_path}' no existe.")

if __name__ == "__main__":
    print(f'Comenzado conversion ....')
    
    model_path = '../src/models/result/cnn_AD_pet.h5'
    output_dir = '../src/salida'

    convert_to_tfjs(model_path, output_dir)

    # convert_to_tfjs(args.model_path, args.output_dir)
