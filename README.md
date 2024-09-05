# AI-Image-Classification

Este projeto consiste em uma IA criada em Python capaz de reconhecer e classificar objetos em imagens usando redes neurais convolucionais (CNNs). O modelo foi treinado utilizando o dataset CIFAR-10 e foi desenvolvido utilizando o TensorFlow.

## Requisitos

- Python 3.x
- TensorFlow

## Instalação

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/AI-Image-Classification.git
cd AI-Image-Classification
```

2. Instale o tensorflow
```bash
pip install tensorflow
```

## Dataset
O modelo foi treinado usando o dataset CIFAR-10, que contém imagens de 32x32 no formato RGB e possui até 10 classificações com 6000 imagens por classe. O dataset é baixado pelo próprio tensorflow no treinamento do modelo

## Como rodar o código
1. Para treinar o modelo com o dataset, execute:
```bash
python train.py
```
2. Para testar o modelo com novas imagens, execute:
```bash
python classify.py --image caminho/para/sua/imagem.jpg
```
