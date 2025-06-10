import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from keras import Model, Sequential, layers, regularizers, optimizers, callbacks
from colorama import Fore, Style

def initialize_LSTM_classifier(input_shape: tuple) -> Model:
    """
    Initialize the LSTM Neural Network with random weights
    """
    kernel_regularizer = regularizers.l2(0.005)
    model = Sequential()
    model.add(layers.LSTM(8, input_shape=input_shape, kernel_regularizer=kernel_regularizer))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    print("✅ Model initialized")
    return model

def initialize_LSTM_regressor(input_shape: tuple) -> Model:
    """
    Initialize the LSTM Neural Network with random weights
    """
    kernel_regularizer = regularizers.l2(0.005)
    model = Sequential()
    model.add(layers.LSTM(8, input_shape=input_shape, kernel_regularizer=kernel_regularizer))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='linear'))

    print("✅ Model initialized")
    return model



def compile_LSTM_classifier(model: Model, learning_rate=0.001) -> Model:
    """
    Compile the LSTM Neural Network
    """
    adam_opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=adam_opt, metrics=['accuracy', 'recall', 'precision'])

    print("✅ Model compiled")
    return model


def compile_LSTM_regressor(model: Model, learning_rate=0.001) -> Model:
    """
    Compile the LSTM Neural Network
    """
    adam_opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=adam_opt, metrics=['mae'])

    print("✅ Model compiled")
    return model


def fit_LSTM_classifier(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Tuple,
        batch_size=32,
        patience=10
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1)

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6)

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es, reduce_lr],
        verbose=1)

    print(f"✅ Classification model trained with val accuracy: {round(np.max(history.history['val_accuracy']), 2)}")
    return model, history


def fit_LSTM_regressor(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Tuple,
        batch_size=32,
        patience=10
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1)

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6)

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es, reduce_lr],
        verbose=1)

    print(f"✅ Regression model trained with val accuracy: {round(np.max(history.history['val_accuracy']), 2)}")
    return model, history
