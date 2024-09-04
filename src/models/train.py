import os
from tensorflow.keras.callbacks import TensorBoard # type: ignore
import numpy as np
from src.models.model import modelo_denso, modelo_cnn, modelo_cnn2
from src.preprocessing.preparar_data import cargar_datos, procesar_datos, mostrar_ejemplos
import tensorboard
from src.preprocessing.aumentar_data import aumentar_datos;

def entrenar_modelo(modelo, X, y, nombre_log):
    log_dir = f'../src/logs/{nombre_log}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True) 
    tensorboard = TensorBoard(log_dir=log_dir)
    modelo.fit(X, y, batch_size=32,
                validation_split=0.15,
                epochs=100,
                callbacks=[tensorboard])

    # Guardar el modelo entrenado
    modelo_path = f'../src/models/result/{nombre_log}.h5'
    modelo.save(modelo_path)
    print(f'--> Modelo guardado en {modelo_path}')
    
def entrenar_modelo_generador(modelo, data_gen_entrenamiento, X_validacion, y_validacion, nombre_log):
    log_dir = f'../src/logs/{nombre_log}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True) 
    tensorboard = TensorBoard(log_dir=log_dir)
    
    modelo.fit(
        data_gen_entrenamiento,
        epochs=100,
        validation_data=(X_validacion, y_validacion),
        steps_per_epoch=int(np.ceil(len(data_gen_entrenamiento) / float(32))),
        validation_steps=int(np.ceil(len(X_validacion) / float(32))),
        callbacks=[tensorboard]
    )

    # Guardar el modelo entrenado
    modelo_path = f'../src/models/result/{nombre_log}.h5'
    modelo.save(modelo_path)
    print(f'--> Modelo guardado en {modelo_path}')

def iniciar_tensorboard(log_dir):
    if not os.path.exists(log_dir):
        raise ValueError(f'La carpeta de logs {log_dir} no existe.')
    
    # Inicia TensorBoard
    os.system(f'tensorboard --logdir={log_dir}')

def main():
    # carpeta_data = '../data/billete/'
    carpeta_data = '../data/'
    print(f"--> Cargando datos ...")
    datos_entrenamiento = cargar_datos(carpeta_data, tamaño_img=224, color_gris=True)
    print(f"--> Procesando datos...")
    X, y = procesar_datos(datos_entrenamiento)

    # Dividir los datos en entrenamiento y validación
    # Supongamos que divides el 85% de los datos para entrenamiento y el 15% para validación
    from sklearn.model_selection import train_test_split # type: ignore
    X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # Verificar los tamaños
    print(f'Tamaño de X: {X.shape}')
    print(f'Tamaño de y: {y.shape}')
    print(f'Tipo de X: {X.dtype}')
    print(f'Tipo de y: {y.dtype}')
    print(f'Primeros 5 valores de y: {y[:5]}')
    # print(f'X: {X}')
    # print(f'y: {y}')

    # Mostrar ejemplos
    print(f'--> Mostrando ejemplos...')
    mostrar_ejemplos(datos_entrenamiento, num_ejemplos=5)

    # Aplicar aumento de datos desde el script separado
    print(f'--> Aumentando Datos...')
    # data_gen_entrenamiento = aumentar_datos(X, y)

    # Aplicar aumento de datos
    data_gen_entrenamiento = aumentar_datos(X_entrenamiento, y_entrenamiento)

    model1 = 'denso_AD_pet'
    model2 =  'cnn_AD_pet'
    model3 =  'cnn2_AD_pet'

    # Crear y entrenar modelos con datos aumentados
    print(f'-->Creando y entrenando modelo DENSO con aumento de datos...')
    denso = modelo_denso(input_shape=(224, 224, 1))
    entrenar_modelo_generador(denso, data_gen_entrenamiento, X_validacion, y_validacion, model1)

    print(f'-->Creando y entrenando modelo CNN con aumento de datos...')
    cnn = modelo_cnn(input_shape=(224, 224, 1))
    entrenar_modelo_generador(cnn, data_gen_entrenamiento, X_validacion, y_validacion, model2)

    print(f'-->Creando y entrenando modelo CNN2 con aumento de datos...')
    cnn2 = modelo_cnn2(input_shape=(224, 224, 1))
    entrenar_modelo_generador(cnn2, data_gen_entrenamiento, X_validacion, y_validacion, model3)


    # print(f'--> Iniciando TensorBoard...')
    # iniciar_tensorboard('../src/logs')

if __name__ == '__main__':
    main()
