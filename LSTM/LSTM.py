from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input

def LSTM_Model(X_train):
    # Defining LSTM model
    LSTM_MODEL = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(128, return_sequences=True), # first LSTM layer
        LSTM(64, return_sequences=True),  # second LSTM layer
        TimeDistributed(Dense(1))         # output 1 value per timestep
    ])

    # Compile model
    LSTM_MODEL.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return LSTM_MODEL
