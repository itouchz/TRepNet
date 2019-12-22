import pandas as pd
import numpy as np

from sktime.utils.load_data import load_from_tsfile_to_dataframe

u_path = './datasets/benchmark/univariate/'   
m_path = './datasets/benchmark/multivariate/'
    
def load_data(name, benchmark=True, univariate=True, sep=',', test_size=0.2):
    if benchmark and univariate:
        path = u_path
    elif benchmark and not univariate:
        path = m_path
    else:
        path = './datasets/realworld/'
        
    if benchmark:
        train_x, train_y = load_from_tsfile_to_dataframe(path + name + '/' + name + '_TRAIN.ts')
        test_x, test_y = load_from_tsfile_to_dataframe(path + name + '/' + name + '_TEST.ts')
        return train_x, train_y, test_x, test_y
    else:
        data = pd.read_csv(path + name + '/data.csv', sep=sep)
        return data
    
def random_benchmark(n=1):
    u_names = pd.read_csv(u_path + 'summary.csv')['Problem'].values
    m_names = pd.read_csv(m_path + 'summary.csv')['Problem'].values
    return np.random.choice(u_names, n, replace=False), np.random.choice(m_names, n, replace=False)

def list_datasets():
    return pd.read_csv(u_path + 'summary.csv')['Problem'].values, pd.read_csv(m_path + 'summary.csv')['Problem'].values