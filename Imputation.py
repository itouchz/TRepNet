import pandas as pd
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from datasets import load_data, random_benchmark, list_datasets
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

def flatten_ts(train, test):
    new_train, new_test = [], []
    
    for _, row in train.iterrows():
        new_list = []
        for i in row.index:
            row[i] = row[i].dropna()
            for j in range(len(row[i])):
                new_list.append(row[i][j])
        new_train.append(new_list)
        
    for _, row in test.iterrows():
        new_list = []
        for i in row.index:
            row[i] = row[i].dropna()
            for j in range(len(row[i])):
                new_list.append(row[i][j])
        new_test.append(new_list)
        
    train_df = pd.DataFrame(new_train)
    test_df = pd.DataFrame(pad_sequences(new_test, maxlen=train_df.shape[1], dtype='float32'))
    
    scaler = RobustScaler()
    scaler.fit(train_df)
    
    return scaler.transform(train_df.dropna()), scaler.transform(test_df.dropna())

def remove_and_impute(train_data, test_data, missing_rate, method='mean'):
    train, test = flatten_ts(train_data, test_data)
    new_train = pd.DataFrame(train)
    count = 0
    ix = [(row, col) for row in range(train.shape[0]) for col in range(1, train.shape[1]-1)]
    for row, col in random.sample(ix, int(round(missing_rate * len(ix)))):
        new_train.iat[row, col] = np.nan
        count += 1
        
    if method == 'mean':
        new_train = new_train.fillna(new_train.mean())
    elif method == 'last':
        new_train = new_train.fillna(method='ffill').fillna(method='bfill')
    else:
        # GAN
        pass
        
    return new_train, test