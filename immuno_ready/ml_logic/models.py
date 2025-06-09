import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def simple_LSTM():

    model = Sequential()
    model.add(LSTM(
        8,
        input_shape=(X_train_matrix_pad.shape[1], X_train_matrix_pad.shape[2]),
        kernel_regularizer=l2(0.005)))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model
