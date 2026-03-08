"""
Neural Network Model Definitions: Dense/MLP, CNN, RNN (LSTM)
Dataset: Fashion-MNIST (28x28 grayscale images, 10 classes)
Techniques: ReLU, Sigmoid/Softmax, Adam optimizer
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CLASS_NAMES = [
    'Camiseta', 'Pantalón', 'Pullover', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolsa', 'Bota'
]


def build_dense_model(input_shape=(784,), num_classes=10):
    """Red Neuronal Densa / MLP con activaciones ReLU y Softmax."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu', name='dense_relu_1'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', name='dense_relu_2'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='sigmoid', name='dense_sigmoid'),
        layers.Dense(num_classes, activation='softmax', name='dense_output')
    ], name='Dense_MLP')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Red Neuronal Convolucional con Conv2D, ReLU, MaxPooling y Softmax."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_relu_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_relu_2'),
        layers.MaxPooling2D((2, 2), name='maxpool'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_relu_3'),
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='fc_relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax', name='cnn_output')
    ], name='CNN')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_rnn_model(input_shape=(28, 28), num_classes=10):
    """Red Neuronal Recurrente (LSTM) procesando la imagen fila por fila."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, activation='relu', name='lstm_relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', name='rnn_fc_relu'),
        layers.Dense(num_classes, activation='softmax', name='rnn_output')
    ], name='RNN_LSTM')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
