#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from Imputation import remove_and_impute
from Models import SAE, CNN_AE, LSTM_AE, GRU_AE, Bi_LSTM_AE, CNN_Bi_LSTM_AE, Causal_CNN_AE, Wavenet, Attention_Bi_LSTM_AE, Attention_CNN_Bi_LSTM_AE, Attention_Wavenet

np.random.seed(7)
tf.random.set_seed(7)


# In[2]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# In[3]:


rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, random_state=7)
svm_clf = SVC(random_state=7, gamma='scale')
knn_clf = KNeighborsClassifier(n_neighbors=1, weights='distance', n_jobs=-1)
mlp_clf = MLPClassifier(random_state=7)


# In[4]:


result_path = './results/others/'
inception = pd.read_csv(result_path + 'InceptionTime-128.csv')[['dataset_name', 'accuracy']]
resnet_ucr = pd.read_csv(result_path + 'resnet-ucr.csv')[['dataset_name', 'accuracy']]
resnet_uea = pd.read_csv(result_path + 'resnet-uea.csv')[['dataset_name', 'accuracy']]
resnet_mts = pd.read_csv(result_path + 'resnet-mts.csv')[['dataset_name', 'accuracy']]
hive_cote = pd.read_csv(result_path + 'singleTrainTest.csv')[['dataset_name', 'HIVE-COTE']]
dtw_uea = pd.read_csv(result_path + 'usrl_uea.csv')[['dataset_name', 'DTW']]


# In[5]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

def flatten_ts(train, test):
    new_train, new_test = [], []
    train_lens = []
    
    for _, row in train.iterrows():
        for i in row.index:
            train_lens.append(len(row[i]))

    maxlen = np.ceil(np.average(train_lens)).astype(int)
    
    for _, row in train.iterrows():
        new_list = []
        for i in row.index:
            ts = []
            for j in range(len(row[i])):
                ts.append(row[i][j])
            new_list.append(ts)
        new_train.append(pad_sequences(new_list, maxlen=maxlen, dtype='float32'))
        
    for _, row in test.iterrows():
        new_list = []
        for i in row.index:
            ts = []
            for j in range(len(row[i])):
                ts.append(row[i][j])
            new_list.append(ts)
        new_test.append(pad_sequences(new_list, maxlen=maxlen, dtype='float32'))
            
    train_df = pd.DataFrame(np.array(new_train).reshape(train.shape[0], maxlen * train.columns.shape[0]))
    test_df = pd.DataFrame(np.array(new_test).reshape(test.shape[0], maxlen * train.columns.shape[0]))

    scaler = RobustScaler()
    scaler.fit(train_df)
    return scaler.transform(train_df), scaler.transform(test_df), maxlen * train.columns.shape[0]
#     return np.array(train_df), np.array(test_df), maxlen * train.columns.shape[0]

def rnn_reshape(train, test, n_steps, n_features):
#     train, test = flatten_ts(train, test)
    return train.reshape(train.shape[0], n_steps, n_features), test.reshape(test.shape[0], n_steps, n_features)


# In[6]:


es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
# mc = keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)


# In[7]:


from TRepNet import TRepNet


# In[8]:


# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 

from tensorflow.keras.utils import plot_model
from sklearn.model_selection import GridSearchCV

def evaluate(data_name, univariate):
    print('Data: ', data_name)
    train_x, train_y, test_x, test_y = load_data(data_name, univariate=univariate)    
#     n_steps = train_x.iloc[0][0].shape[0]
    n_features = train_x.columns.shape[0]
        
    X_train, X_test, n_steps = flatten_ts(train_x, test_x)
    X_train, X_test = rnn_reshape(X_train, X_test, n_steps // n_features, n_features)
            
    encoder, decoder = TRepNet(n_steps // n_features, n_features, activation='elu')
    model = keras.models.Sequential([encoder, decoder])

    plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)
    plot_model(decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True)

    start_time = time.time()
    model.compile(loss="mae", optimizer=keras.optimizers.Nadam(lr=0.001, clipnorm=1.), metrics=['mae'])
    history = model.fit(X_train, X_train, epochs=500, batch_size=16, validation_data=[X_test, X_test], callbacks=[es], verbose=0, shuffle=False)
    
    # Codings
    codings_train = encoder.predict(X_train)
    codings_test = encoder.predict(X_test)
    
#     # RF
#     rf_clf.fit(codings_train, train_y)
#     pred = rf_clf.predict(codings_test)
#     rf_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='weighted')}
#     print('RF >>', rf_scores)

    # SVM
    svm_clf = SVC(random_state=7, gamma='scale')
    nb_classes = np.unique(train_y).shape[0]
    train_size = codings_train.shape[0]
    if train_size // nb_classes < 5 or train_size < 50:
        svm_clf.fit(codings_train, train_y)
    else:
        grid_search = GridSearchCV(svm_clf, {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]}, cv=5, iid=False, n_jobs=-1)
        if train_size <= 10000:
            grid_search.fit(codings_train, train_y)
        else:
            codings_train, _, train_y, _  = train_test_split(codings_train, train_y, train_size=10000, random_state=7, stratify=train_y)
            grid_search.fit(codings_train, train_y)       
        svm_clf = grid_search.best_estimator_

        svm_clf.fit(codings_train, train_y)
        
#     svm_clf.fit(codings_train, train_y)
    pred = svm_clf.predict(codings_test)
    duration = time.time() - start_time
    svm_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='weighted')}
    print('SVM >>', svm_scores)

#     # 1-NN
#     knn_clf.fit(codings_train, train_y)
#     pred = knn_clf.predict(codings_test)
#     knn_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='weighted')}
#     print('1-NN >>', knn_scores)

#     # MLP
#     mlp_clf.fit(codings_train, train_y)
#     pred = mlp_clf.predict(codings_test)
#     mlp_scores = {'accuracy': accuracy_score(test_y, pred), 'f1': f1_score(test_y, pred, average='weighted')}
#     print('MLP >>', mlp_scores)

    # SOTA Results
    print('*'*10)
    print('InceptionTime:', inception[inception['dataset_name'] == data_name]['accuracy'].values[0] if len(inception[inception['dataset_name'] == data_name]['accuracy'].values) >= 1 else 'N/A')
    print('ResNet:', resnet_ucr[resnet_ucr['dataset_name'] == data_name]['accuracy'].values[0] if len(resnet_ucr[resnet_ucr['dataset_name'] == data_name]['accuracy'].values) >= 1 else 'N/A')
    print('ResNet:', resnet_uea[resnet_uea['dataset_name'] == data_name]['accuracy'].values[0] if len(resnet_uea[resnet_uea['dataset_name'] == data_name]['accuracy'].values) >= 1 else 'N/A')
    print('ResNet:', resnet_mts[resnet_mts['dataset_name'] == data_name]['accuracy'].values[0] if len(resnet_mts[resnet_mts['dataset_name'] == data_name]['accuracy'].values) >= 1 else 'N/A')
    print('HIVE-COTE:', hive_cote[hive_cote['dataset_name'] == data_name]['HIVE-COTE'].values[0] if len(hive_cote[hive_cote['dataset_name'] == data_name]['HIVE-COTE'].values) >= 1 else 'N/A')
    print('DTW:', dtw_uea[dtw_uea['dataset_name'] == data_name]['DTW'].values[0] if len(dtw_uea[dtw_uea['dataset_name'] == data_name]['DTW'].values) == 1 else 'N/A')
    print('*'*10)
    
    results.append({'dataset': data_name, 'dim': codings_train.shape[1], 
#                     'RF-ACC': rf_scores['accuracy'], 
                    'SVM-ACC': svm_scores['accuracy'],
#                     '1NN-ACC': knn_scores['accuracy'], 
                    # 'MLP-ACC': mlp_scores['accuracy'], 
#                     'RF-F1': rf_scores['f1'], 
                    'SVM-F1': svm_scores['f1'],
#                     '1NN-F1': knn_scores['f1'], 
                    # 'MLP-F1': mlp_scores['f1'],
                    'duration (sec)': duration
                    })


# In[9]:


# mul_datasets = list(list_datasets()[1])
# mul_datasets.remove('DuckDuckGeese')
# mul_datasets.remove('EigenWorms')
# mul_datasets.remove('FaceDetection')
# mul_datasets.remove('Heartbeat')
# mul_datasets.remove('InsectWingbeat')
# mul_datasets.remove('LSST')
# mul_datasets.remove('MotorImagery')
# mul_datasets.remove('PEMS-SF')

selected_mul_datasets = ['ArticularyWordRecognition', 'BasicMotions', 'AtrialFibrillation', 'Cricket',
                         'ERing', 'HandMovementDirection', 'Handwriting', 'JapaneseVowels', 'PenDigits', 'RacketSports', 'SelfRegulationSCP1',
                         'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'EthanolConcentration']

# uni_datasets = list(list_datasets()[0])
# uni_datasets.remove('DodgerLoopDay')
# uni_datasets.remove('DodgerLoopGame')
# uni_datasets.remove('DodgerLoopWeekend')
# uni_datasets.remove('ElectricDevices')
# uni_datasets.remove('MelbournePedestrian')
# uni_datasets.remove('PLAID')

selected_uni_datasets = ['Earthquakes', 'ArrowHead', 'BeetleFly', 'ChlorineConcentration', 'Chinatown', 'DiatomSizeReduction', 'ECG200', 'ECG5000', 'ECGFiveDays',
                         'FreezerSmallTrain', 'Fungi', 'GunPoint', 'GunPointAgeSpan','GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Herring', 
                         'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound', 'Lightning2', 'MedicalImages', 'MiddlePhalanxTW',
                         'NonInvasiveFetalECGThorax2', 'OliveOil', 'PhalangesOutlinesCorrect', 'PickupGestureWiimoteZ','PigAirwayPressure', 'PowerCons',
                         'ProximalPhalanxOutlineAgeGroup', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'SmoothSubspace', 'StarLightCurves',
                         'SyntheticControl', 'Trace', 'UMD', 'UWaveGestureLibraryAll', 'Wafer', 'Yoga']


# In[ ]:

results = []

print('-'*10)
print('Multivariate')
print('-'*10)

mul_datasets = ['BasicMotions', 'ERing', 'SpokenArabicDigits', 'AtrialFibrillation', 'EthanolConcentration']
for dataset in selected_mul_datasets:
    evaluate(dataset, univariate=False)
print('='*10)
pd.DataFrame(results).to_csv('./results/mul-TRepNet-results.csv', index=False)

print('- END -')