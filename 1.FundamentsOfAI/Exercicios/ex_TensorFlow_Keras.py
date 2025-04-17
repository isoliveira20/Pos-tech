#Desenvolvido para código aberto para computação numérica e aprendizado de máquina
#Exemplo: criar e treinar rede neural simples

import tensorflow as tf
import numpy as np

# Exemplo de dados de entrada
# X: matriz com 100 amostras e 3 características
# y: vetor de saída com 100 valores
X = np.random.random((100, 3))
y = np.random.random((100, 1))

# Definindo o modelo usando Input
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(10, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o modelo
model.fit(X, y, epochs=5)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#Exemplo: criar um modelo de classificaão com Keras
# Exemplo de dados de entrada
# X: matriz com 100 amostras e 8 características
# y: vetor de saída com 100 valores binários (0 ou 1)
X = np.random.random((100, 8))
y = np.random.randint(2, size=(100, 1))

# Definindo o modelo
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
model.fit(X, y, epochs=150, batch_size=10)