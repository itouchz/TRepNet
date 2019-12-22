#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore') 

from tensorflow import keras
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from Models import SAE, CNN_AE, LSTM_AE, GRU_AE, Bi_LSTM_AE, CNN_Bi_LSTM_AE, Causal_CNN_AE, Wavenet, Attention_Bi_LSTM_AE, Attention_CNN_Bi_LSTM_AE, Attention_Wavenet

np.random.seed(7)
tf.random.set_seed(7)


# In[2]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# In[3]:


svm_reg = SVR(gamma='scale')


# In[4]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
dataset_path = './datasets/regression/'


# In[5]:


def n_steps_reshape(X_train_full, y_train_full, n_steps=10, for_rnn=False):
    new_data = []
    new_label = []
    columns = X_train_full.columns
    for i in range(X_train_full.shape[0]-n_steps):
        new_instance = []
        train_data = X_train_full[i:i+n_steps]
        for c in columns:
            for v in train_data[c].values:
                new_instance.append(v)
#         for _, row in train_data.iterrows():
#             for c in columns:
#                 new_instance.append(row[c])
        new_label.append(y_train_full[i+n_steps])
        new_data.append(new_instance)

    scaler = RobustScaler()
    new_data = scaler.fit_transform(new_data)
    new_label = scaler.fit_transform(np.array(new_label).reshape(-1,1))

    if for_rnn:
        return np.array(new_data).reshape(len(new_data), n_steps, columns.shape[0]), new_label
    else:
        return np.array(new_data), new_label


# In[6]:


def LSTM_Model(n_steps, n_features):
    return keras.models.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=[n_steps, n_features]),
        keras.layers.LSTM(128),
        keras.layers.Dense(1, activation=keras.layers.LeakyReLU(alpha=0.5))
    ])


# In[7]:


results = []


# In[8]:


from TRepNet import TRepNet

def get_codings(X_train, n_steps, n_features):
#     X_train, X_test, n_steps = flatten_ts(train_x, test_x)
#     X_train, X_test = rnn_reshape(X_train, X_test, n_steps // n_features, n_features)
    encoder, decoder = fn(n_steps, n_features, activation='elu')
    model = keras.models.Sequential([encoder, decoder])
    
    model.compile(loss="mae", optimizer=keras.optimizers.Nadam(lr=0.001, clipnorm=1.), metrics=['mae'])
    history = model.fit(X_train, X_train, epochs=500, batch_size=16, validation_split=0.20, callbacks=[es], verbose=1, shuffle=False)

    # Codings
    return encoder.predict(X_train)


# In[9]:


es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)


# In[ ]:


for fn in [SAE, CNN_AE, LSTM_AE, GRU_AE, Bi_LSTM_AE, CNN_Bi_LSTM_AE, Causal_CNN_AE, Wavenet, TRepNet]:
    results = []

    print(fn.__name__)
    
    name = 'Solar Generation'
    solar_data = pd.read_csv(dataset_path + 'Solar/data.csv', quotechar='"').fillna(0)
    solar_data_X = solar_data.drop(columns=['SITE_NO', 'DATE', 'TIME'])
    solar_data_y = solar_data['GEN_ENERGY']
    X_train_full, y_train_full = n_steps_reshape(solar_data_X, solar_data_y, 10, for_rnn=True)
    X_train_full = get_codings(X_train_full, 10, X_train_full.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=7)
    svm_reg.fit(X_train, y_train)
    pred = svm_reg.predict(X_test)
    print(mse(y_test, pred))
    results.append({'dataset': name, 'MSE': mse(y_test, pred)})
    
    name = 'Beijing PM 2.5'
    beijing_data = pd.read_csv(dataset_path + 'Beijing-PM25.csv').dropna().drop(columns=['No', 'year']).reset_index(drop=True)
    beijing_data_X = pd.get_dummies(beijing_data, columns=['cbwd'])
    beijing_data_y = beijing_data['pm2.5']
    X_train_full, y_train_full = n_steps_reshape(beijing_data_X, beijing_data_y, 10, for_rnn=True)
    X_train_full = get_codings(X_train_full, 10, X_train_full.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=7)
    svm_reg.fit(X_train, y_train)
    pred = svm_reg.predict(X_test)
    print(mse(y_test, pred))
    results.append({'dataset': name, 'MSE': mse(y_test, pred)})
    
    name = 'Appliance Energy Prediction'
    energy_data = pd.read_csv(dataset_path + 'energydata_complete.csv')
    enery_data_X = energy_data.drop(columns=['date'])
    enery_data_y = energy_data['Appliances']
    X_train_full, y_train_full = n_steps_reshape(enery_data_X, enery_data_y, 10, for_rnn=True)
    X_train_full = get_codings(X_train_full, 10, X_train_full.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=7)
    svm_reg.fit(X_train, y_train)
    pred = svm_reg.predict(X_test)
    print(mse(y_test, pred))
    results.append({'dataset': name, 'MSE': mse(y_test, pred)})
    
    name = 'Parking Birmingham'
    parking_data = pd.read_csv(dataset_path + 'Parking Birmingham.csv')
    parking_data_X = parking_data.drop(columns=['SystemCodeNumber', 'LastUpdated'])
    parking_data_y = parking_data['Occupancy']
    X_train_full, y_train_full = n_steps_reshape(parking_data_X, parking_data_y, 10, for_rnn=True)
    X_train_full = get_codings(X_train_full, 10, X_train_full.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=7)
    svm_reg.fit(X_train, y_train)
    pred = svm_reg.predict(X_test)
    print(mse(y_test, pred))
    results.append({'dataset': name, 'MSE': mse(y_test, pred)})
    
    name = 'Daily Deemand Forecasting'
    demand_data = pd.read_csv(dataset_path + 'Daily_Demand_Forecasting_Orders.csv', sep=';')
    demand_data_X = demand_data
    demand_data_y = demand_data['Target']
    X_train_full, y_train_full = n_steps_reshape(demand_data_X, demand_data_y, 10, for_rnn=True)
    X_train_full = get_codings(X_train_full, 10, X_train_full.shape[2])
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=7)
    svm_reg.fit(X_train, y_train)
    pred = svm_reg.predict(X_test)
    print(mse(y_test, pred))
    results.append({'dataset': name, 'MSE': mse(y_test, pred)})
    
    pd.DataFrame(results).to_csv('./results/regression-'+ fn.__name__ +'-results.csv', index=False)
print('END')