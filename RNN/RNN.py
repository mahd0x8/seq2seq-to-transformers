import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, TimeDistributed, Input

def RNN(X_train):
    # Building the RNN model
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        
        # First RNN layer (return sequences to stack another RNN)
        SimpleRNN(128, return_sequences=True),
        Dropout(0.2),
        
        # Second RNN layer
        SimpleRNN(64, return_sequences=True),
        Dropout(0.2),
        
        # TimeDistributed Dense layer to output 1 value per timestep
        TimeDistributed(Dense(1))
    ])

    # Compiling model
    model.compile(optimizer='adam', loss='mse')
    
    return model
