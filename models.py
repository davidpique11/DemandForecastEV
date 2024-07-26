import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


def get_model(name, window_size, num_features):
    if name == 'LSTM':
        return get_LSTM(window_size, num_features)
    elif name == 'LSTM_stacked':
        return get_LSTM_stacked(window_size, num_features)
    elif name == 'LSTM_stacked_2':
        return get_LSTM_stacked_2(window_size, num_features)
    elif name == 'LSTM_stacked_3':
        return get_LSTM_stacked_3(window_size, num_features)
    elif name == 'LSTM_stacked_3_alt':
        return get_LSTM_stacked_3_alt(window_size, num_features)
    elif name == 'Bidirectional_LSTM':
        return get_Bidirectional_LSTM(window_size, num_features)
    elif name == 'Bidirectional_LSTM_alt':
        return get_Bidirectional_LSTM_alt(window_size, num_features)
    

def get_LSTM(window_size, num_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(1),
        ])
    return model

def get_LSTM_stacked(window_size, num_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(1),
        ])
    return model

def get_LSTM_stacked_2(window_size, num_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(1),
        ])
    return model

def get_LSTM_stacked_3(window_size, num_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(64, activation='relu'),  
        tf.keras.layers.Dense(1) 
    ])
    return model

def get_LSTM_stacked_3_alt(window_size, num_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  
    ])
    return model

def get_Bidirectional_LSTM(window_size, num_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

def get_Bidirectional_LSTM_alt(window_size, num_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

from keras.callbacks import EarlyStopping, ModelCheckpoint

def configure_callbacks(early_stopping = False,patience=10, checkpoint= False, model_name= None, window_size=None, batch_size=None): 
    callbacks = []
    if checkpoint:
        model_nm = model_name + f'_ws{window_size}_bs{batch_size}'
        model_name += f'_ws{window_size}_bs{batch_size}.keras'
        checkpoint_dir = f'Results/train_{model_nm}_results'
        checkpoint_filepath = os.path.join(checkpoint_dir, model_name)
        
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        callbacks.append(model_checkpoint_callback)

    if early_stopping:
        callbacks.append(EarlyStopping(patience=patience, monitor='val_loss'))
        
    return callbacks

def predict(model,generator,scaler):
    # Run inferences on dataset generator
    predictions = model.predict(generator)
    # Rescale predicted values
    predictions = scaler.inverse_transform(predictions)
    
    return predictions