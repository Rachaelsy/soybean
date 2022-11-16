import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import time
import os
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import random
from torchvision import utils as vutils
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as MSE
import utils 
import models
from dataset_fine import Mydataset
import config as cfg
from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(net, train_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    for epoch in range(cfg.n_epochs):
        total_loss = 0
        for idx,(data, label) in enumerate(train_loader):
            data1 = data.squeeze(1)
            pred = net(data1)
            label=label.unsqueeze(1)
            loss=criterion(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
    # validation
        if epoch % 25 ==0:
            preds=[]
            labels=[]
            for idx, (x, label) in enumerate(test_loader):
                x = x.squeeze(1)  # batch_size,seq_len,input_size
                pred = net(x)
                preds.extend(pred.data.squeeze(1).tolist())
                labels.extend(label.tolist())
            c = [abs(preds[i]-labels[i])/labels[i] for i in range(0,len(preds))]
            print("epoch = {}".format(epoch))
            print('accuracy = {}'.format(1-np.mean(c)))
            print('r2 = {}'.format(r2(labels, preds)))
            print('RMSE = {}'.format(MSE(preds, labels)**0.5))
            print("-------------------------")

def main():
    # Load data
    # print('\nLoading data...\n')
    set_seed(cfg.rand_seed)
    f_x = "./dataset/x.pkl"
    f_y = "./dataset/y.pkl"
    x, y = utils.load_data(f_x=f_x, f_y=f_y)
    x = x.astype("float32")
    y = y.astype("float32")
    print(x.shape)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.2, random_state=129)
    train_loader = DataLoader(dataset=Mydataset(trainx,trainy,transform=transforms.ToTensor()), batch_size=12, shuffle=True)
    test_loader = DataLoader(dataset=Mydataset(testx,testy), batch_size=12, shuffle=True)
    net = models.SimpleLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    #net = models.TCN(input_size=cfg.input_size, output_size=cfg.output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    train(net, train_loader, test_loader)

if __name__ == '__main__':
    main()
