import sys
import os

# Añadir el directorio raíz al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ejecutar el script de entrenamiento
# from models.train import main
from src.models.train import entrenar_modelo,main
# from src.models.model import save_model

def main1():
    main()

if __name__ == '__main__':
    main1()
