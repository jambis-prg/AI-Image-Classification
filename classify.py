import argparse
import os
from PIL import Image
import tensorflow as tf
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Classify an image using a trained model.')
    parser.add_argument('--image', type=str, required = True, help = 'Path to the image file to classify')

    args = parser.parse_args()

    imagePath = args.image

    if not os.path.isfile(imagePath):
        print(f'Error: O arquivo de imagem "{imagePath}" não foi encontrado.')
        return
    
    # Processar a imagem
    try:
        image = Image.open(imagePath)
        print(f'Imagem carregada com sucesso: {imagePath}')

        image = image.resize((image.width, image.height))
    
        # Converter a imagem para um array NumPy
        image_array = np.array(image)
        
        # Normalizar os valores dos pixels para [0, 1] se necessário
        image_array = image_array / 255.0
        
        # Adicionar uma dimensão de lote (batch dimension)
        image_array = np.expand_dims(image_array, axis=0)

        directory = 'models'
        filePath = os.path.join(directory, 'train_model.h5')
        if not os.path.exists(directory) or not os.path.isfile(filePath):
            return

        model = tf.keras.models.load_model(filePath)
        result = model.predict(image_array)
        print(f'Resultado da classificação: {np.argmax(result, axis=1)[0]}')
    except Exception as e:
        print(f'Exceção lançada: {e}')

if __name__ == '__main__':
    main()