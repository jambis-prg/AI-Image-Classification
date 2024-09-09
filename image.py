import tensorflow as tf
import numpy as np
from PIL import Image  # Pillow é importado como PIL
import os

# Função para salvar a imagem
def save_image(image_array, filename):
    image = Image.fromarray(np.uint8(image_array))
    image.save(filename)

def main():
    # Carregar o conjunto de dados CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    n = 10
    # Selecionar a primeira imagem do conjunto de treinamento
    image = x_train[n]  # Escolha a imagem desejada (0 aqui representa a primeira imagem)
    label = y_train[n]
    # Salvar a imagem
    save_image(image, 'cifar10_sample_image.png')
    print('Imagem salva como cifar10_sample_image.png')
    print(f"Imagem baixada tem label: {label}")

if __name__ == '__main__':
    main()

