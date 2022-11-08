from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
import torch

class Mydataset(Dataset):
 
    def __init__(self,xx,yy,transform=None):
        self.x=xx
        self.y=yy
        self.tranform = transform
 
    def __getitem__(self,index):
        x1=self.x[index]
        y1=self.y[index]
        if self.tranform !=None:
            return self.tranform(x1),y1
        return x1,y1
 
    def __len__(self):
        return len(self.x)