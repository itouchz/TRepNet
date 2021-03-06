import numpy as np
import pandas as pd
import tensorflow as tf
import os
import warnings
import time

warnings.filterwarnings('ignore') 

from tensorflow import keras
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from datasets import load_data, random_benchmark, list_datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score

np.random.seed(7)
tf.random.set_seed(7)

from tensorflow.keras.layers import Conv1D, LSTM, GRU, Bidirectional, MaxPool1D, RepeatVector, Dense, Attention, Flatten, Dot
from tensorflow.keras.layers import BatchNormalization, Input, Activation, Dropout, Lambda, Multiply, Add, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

def get_output_dim(original_dim):
    if original_dim // 1.3 >= 512:
        return 512
    elif original_dim // 1.3 <= 128:
        return 128
    else:
        return int(original_dim // 1.3)

def TRepNet(n_steps, n_features, activation='elu'):
    codings_size = get_output_dim(n_steps * n_features)
    dilation_rates = [2**i for i in range(10)] * 1
    
    skips = []

    encoder_input = Input(shape=[n_steps, n_features])
    # Convolution
    conv = encoder_input
    for dilation_rate in dilation_rates:
        conv = keras.layers.GaussianNoise(0.01)(conv)
        conv = Conv1D(16, 1, activation=activation, padding='same')(conv)
        
        conv_filter = Conv1D(filters=128, kernel_size=3, padding='causal', activation=activation, dilation_rate=dilation_rate)(conv)
        conv_filter = Dropout(0.1)(conv_filter)
        
        conv_gate = Conv1D(filters=128, kernel_size=3, padding='causal', activation=activation, dilation_rate=dilation_rate)(conv)
        conv_gate = Dropout(0.1)(conv_gate)
        
        mul = Multiply()([Activation('tanh')(conv_filter), Activation('sigmoid')(conv_gate)])
        skip = Conv1D(16, 1, padding='same', activation=activation)(mul)
        
        conv = Add()([conv, skip])
        
        skips.append(skip)
    
    conv = Activation(activation)(Add()(skips))
    conv = Conv1D(16, 1, activation=activation, padding='same')(conv)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = Flatten()(conv)

    # CNN
    cnn = Conv1D(32, kernel_size=1, padding="SAME", activation=activation)(encoder_input)
    cnn = MaxPool1D(pool_size=2)(cnn)
    cnn = Conv1D(64, kernel_size=3, padding="SAME", activation=activation)(cnn)
    cnn = MaxPool1D(pool_size=2)(cnn)
    cnn = Conv1D(128, kernel_size=5, padding="SAME", activation=activation)(cnn)
    
    cnn = Conv1D(16, 1, activation=activation, padding='same')(cnn)
    cnn = MaxPool1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    
    z1 = Dense(codings_size, kernel_initializer='lecun_normal',  activation='selu')(conv)
    z3 = Dense(codings_size, kernel_initializer='lecun_normal',  activation='selu')(cnn)

    z = Add()([z1, z3])
    
    encoder_output = Dense(codings_size, activation='sigmoid')(z)
    encoder = Model(inputs=[encoder_input], outputs=[encoder_output])

    # Decoder
    decoder_input = Input(shape=[codings_size])
    noise_input = keras.layers.GaussianNoise(0.01)(decoder_input)
    dconv = keras.layers.Reshape([codings_size, 1, 1])(noise_input)
    dconv = Conv2DTranspose(filters=32, kernel_size=3, activation=activation)(dconv)
    dconv = Conv2DTranspose(filters=16, kernel_size=1, activation=activation)(dconv)
    dconv = Flatten()(dconv)
    x = Dense(n_steps * n_features)(dconv)
    decoder_output = keras.layers.Reshape([n_steps, n_features])(x)
    decoder = Model(inputs=[decoder_input], outputs=[decoder_output])

    return encoder, decoder