import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import RobustScaler, Normalizer
from datasets import load_data, random_benchmark, list_datasets
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, RepeatVector, Dense, Attention, Input, Embedding, Dropout, BatchNormalization
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=7)
svm_clf = SVC(C=10, gamma='scale', random_state=7)

def RNN_AE(n_steps, n_features, activation):   
    recurrent_encoder = keras.models.Sequential([
        GRU(50, activation=activation, return_sequences=True, input_shape=[n_steps, n_features]),
        GRU(20, activation=activation),
        Dense(200, activation='relu')
    ])

    recurrent_decoder = keras.models.Sequential([
        RepeatVector(n_steps, input_shape=[200]),
        GRU(20, activation=activation, return_sequences=True),
        GRU(50, activation=activation, return_sequences=True),
        TimeDistributed(Dense(n_features, activation='selu'))
    ])

    return recurrent_encoder, recurrent_decoder

def BI_RNN_AE(n_steps, n_features, activation):   
    recurrent_encoder = keras.models.Sequential([
        Bidirectional(GRU(50, activation=activation, return_sequences=True, input_shape=[n_steps, n_features])),
        Bidirectional(GRU(20, activation=activation)),
        Dense(200, activation='relu')
    ])

    recurrent_decoder = keras.models.Sequential([
        RepeatVector(n_steps, input_shape=[200]),
        Bidirectional(GRU(20, activation=activation, return_sequences=True)),
        Bidirectional(GRU(50, activation=activation, return_sequences=True)),
        TimeDistributed(Dense(n_features, activation='selu'))
    ])

    return recurrent_encoder, recurrent_decoder

es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

rnn_results = []
birnn_results = []

def flatten_ts(df):
    new_list = []
    for _, row in df.iterrows():
        new_dict = {}
        for i in row.index:
            for j in range(len(row[i])):
                new_dict[i + '_' + str(j)] = row[i][j]
        new_list.append(new_dict)
    return pd.DataFrame(new_list)

def rnn_reshape(train, test):
    train, test = flatten_ts(train).values, flatten_ts(test).values
    #     impute NaN first!!

#     scaler = Normalizer()
#     scaler.fit(train)
#     train, test = scaler.transform(train), scaler.transform(test)
    return train.reshape(train.shape[0], train.shape[1], 1), test.reshape(test.shape[0], test.shape[1], 1)

def evaluate(data_name):
    print('Data: ', data_name)
    train_x, train_y, test_x, test_y = load_data(data_name)
    X_train, X_test = rnn_reshape(train_x, test_x)
    
    encoder, decoder = RNN_AE(X_train.shape[1], X_train.shape[2], activation='relu')
    model = keras.models.Sequential([encoder, decoder])
    model.compile(loss="mse", optimizer='adam', metrics=['mae'])
    history = model.fit(X_train, X_train, epochs=50, batch_size=24, validation_data=[X_test, X_test], callbacks=[es], verbose=2)
    
    # Codings
    codings_train = encoder.predict(X_train)
    codings_test = encoder.predict(X_test)
    
    # RF
    rf_clf.fit(flatten_ts(train_x), train_y)
    pred = rf_clf.predict(flatten_ts(test_x))
    rf1_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='macro')}
    print('RF1 -- ACC:', rf1_scores['accuracy'], 'F1:', rf1_scores['f1'])

    rf_clf.fit(codings_train, train_y)
    pred = rf_clf.predict(codings_test)
    rf2_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='macro')}
    print('RF2 -- ACC:', rf2_scores['accuracy'], 'F1:', rf2_scores['f1'])

    # SVM
    svm_clf.fit(flatten_ts(train_x), train_y)
    pred = svm_clf.predict(flatten_ts(test_x))
    svm1_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='macro')}
    print('SVM1 -- ACC:', svm1_scores['accuracy'], 'F1:', svm1_scores['f1'])

    svm_clf.fit(codings_train, train_y)
    pred = svm_clf.predict(codings_test)
    svm2_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='macro')}
    print('SVM2 -- ACC:', svm2_scores['accuracy'], 'F1:', svm2_scores['f1'])
    
    rnn_results.append({'model': 'GRUx2/relu/mse/adam', 
                        'dataset': data_name, 
                        'RF1-ACC': rf1_scores['accuracy'],
                        'RF1-F1': rf1_scores['f1'],
                        'RF2-ACC': rf2_scores['accuracy'],
                        'RF2-F1': rf2_scores['f1'],
                        'SVM1-ACC': svm1_scores['accuracy'],
                        'SVM1-F1': svm1_scores['f1'],
                        'SVM2-ACC': svm2_scores['accuracy'],
                        'SVM2-F1': svm2_scores['f1']
                       })
    
    encoder, decoder = BI_RNN_AE(X_train.shape[1], X_train.shape[2], activation='relu')
    model = keras.models.Sequential([encoder, decoder])
    model.compile(loss="mse", optimizer='adam', metrics=['mae'])
    history = model.fit(X_train, X_train, epochs=50, batch_size=24, validation_data=[X_test, X_test], callbacks=[es], verbose=2)
    
    # Codings
    codings_train = encoder.predict(X_train)
    codings_test = encoder.predict(X_test)
    
    # RF
    rf_clf.fit(flatten_ts(train_x), train_y)
    pred = rf_clf.predict(flatten_ts(test_x))
    rf1_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='macro')}
    print('RF1 -- ACC:', rf1_scores['accuracy'], 'F1:', rf1_scores['f1'])

    rf_clf.fit(codings_train, train_y)
    pred = rf_clf.predict(codings_test)
    rf2_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='macro')}
    print('RF2 -- ACC:', rf2_scores['accuracy'], 'F1:', rf2_scores['f1'])

    # SVM
    svm_clf.fit(flatten_ts(train_x), train_y)
    pred = svm_clf.predict(flatten_ts(test_x))
    svm1_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='macro')}
    print('SVM1 -- ACC:', svm1_scores['accuracy'], 'F1:', svm1_scores['f1'])

    svm_clf.fit(codings_train, train_y)
    pred = svm_clf.predict(codings_test)
    svm2_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='macro')}
    print('SVM2 -- ACC:', svm2_scores['accuracy'], 'F1:', svm2_scores['f1'])
    
    birnn_results.append({'model': 'BiGRUx2/relu/mse/adam', 
                        'dataset': data_name, 
                        'RF1-ACC': rf1_scores['accuracy'],
                        'RF1-F1': rf1_scores['f1'],
                        'RF2-ACC': rf2_scores['accuracy'],
                        'RF2-F1': rf2_scores['f1'],
                        'SVM1-ACC': svm1_scores['accuracy'],
                        'SVM1-F1': svm1_scores['f1'],
                        'SVM2-ACC': svm2_scores['accuracy'],
                        'SVM2-F1': svm2_scores['f1']
                       })
    
for dataset in list_datasets()[0]:
    evaluate(dataset)
    
pd.DataFrame(rnn_results).to_csv('./uni_rnn_results.csv', index=False)
pd.DataFrame(birnn_results).to_csv('./uni_birnn_results.csv', index=False)