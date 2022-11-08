import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import os
import torch
import utils



def main():
    df_soybean = pd.read_csv('./dataset/data.csv')
    df_soybean.set_index('para',inplace=True)
    feature_names = ['GDD','KDD','EVI','ppt']
    grow_stage = ['emerged','branching','blooming','setting_pods','turning_yellow','falling_leaf']
    x = []
    y = []
    for county in df_soybean.columns:
        df = pd.DataFrame(columns=feature_names,index=grow_stage)
        for i in df.columns:
            for j in df.index:
                df[i][j] = df_soybean[county]['{}_{}'.format(j,i)]
        x_one = np.array(df)
        x.append(x_one)
        y.append(df_soybean[county]['yield'])
    x = np.array(x)
    y = np.array(y)
    print(df_soybean.head())
    print('x_shape: {}  y_shape: {}'.format(x.shape, y.shape))
    utils.save_pickle('./dataset/x.pkl', x)
    utils.save_pickle('./dataset/y.pkl', y)


if __name__ == '__main__':
    main()