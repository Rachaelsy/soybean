import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_data(f_x, f_y):
    x = load_pickle(f_x)
    y = load_pickle(f_y)
    y = np.array(y[:, np.newaxis])
    ssx = StandardScaler()
    ssy = StandardScaler()
    for i in range(x.shape[-1]):
        ssx.fit(x[:, :, i])
        x[:, :, i] = ssx.transform(x[:, :, i])
    ssy.fit(y)
    y = ssy.transform(y)

    return x,y
    # train_ids, valid_ids, test_ids = get_ids_for_tvt()
    # x_train = x[train_ids]
    # y_train = y[train_ids]
    # x_valid = x[valid_ids]
    # y_valid = y[valid_ids]
    # print('x_shape: {}  y_shape: {}\nx_train_shape: {}  y_train_shape: {}  x_valid_shape: {}  y_valid_shape: {}  x_test_shape: {}  y_test_shape: {}\n'
    #       .format(x.shape, y.shape, x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape))
    # return x_train, y_train, x_valid, y_valid 