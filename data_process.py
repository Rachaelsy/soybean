import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time, datetime

def main():
    df_soybean = pd.read_csv('./dataset/data.csv')
    print(df_soybean)


if __name__ == '__main__':
    main()