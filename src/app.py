from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Cargar el modelo
# src\models\result\perros-gatos-cnn-ad.h5
model = tf.keras.models.load_model('../src/models/result/cnn_AD_1.h5')  # Asegúrate de que la ruta sea correcta

# Paso 1: Definir la función de preprocesamiento
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    if len(image_array.shape) == 2:  # Imagen en escala de grises
        image_array = np.expand_dims(image_array, axis=-1)  # Convertir a forma (224, 224, 1)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Añadir la dimensión del batch
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Abre la imagen
        image = Image.open(file.stream).convert('L')  # Convertir a escala de grises
        # Preprocesa la imagen
        image_array = preprocess_image(image)
        # Realiza la predicción
        prediction = model.predict(image_array)
        # class_index = np.argmax(prediction)  # Asumiendo que es un problema de clasificación multiclase
        class_index = int(np.argmax(prediction))  # Convertir numpy.int64 a int
        
        # Mapear el índice a nombre de clase
        class_names = ['denominacion_10', 'denominacion_20']
        class_name = class_names[class_index]
        
        return jsonify({'class_index': class_index, 'class_name': class_name}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
