import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import models, transforms
import time
import os
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import random
from torchvision import utils as vutils
from sklearn.metrics import r2_score as r2
import utils 
from dataset_fine import Mydataset

def main():
    # Load data
    # print('\nLoading data...\n')
    f_x = "./dataset/x.pkl"
    f_y = "./dataset/y.pkl"
    x, y = utils.load_data(f_x=f_x, f_y=f_y)
    print(x.shape)
    total_len = x.shape[0]
    trainx,trainy=x[:int(0.7*total_len)],y[:int(0.7*total_len)]
    testx,testy=x[int(0.7*total_len):],y[int(0.7*total_len):]
    train_loader=DataLoader(dataset=Mydataset(trainx,trainy,transform=transforms.ToTensor()), batch_size=12, shuffle=True)
    test_loader=DataLoader(dataset=Mydataset(testx,testy), batch_size=12, shuffle=False)


if __name__ == '__main__':
    main()
