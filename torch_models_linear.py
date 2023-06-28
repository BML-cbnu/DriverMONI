import pandas as pd
import numpy as np
import dgl 
from tqdm import tqdm
from torch.utils import data
import networkx as nx
import torch
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
import os
import pickle
import argparse

class feat_L_BRCA(nn.Module):
    def __init__(self):
        super(feat_L_BRCA, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 256)
        self.ll2=nn.Linear(256, 128)
        self.ll3=nn.Linear(128, 64)

        self.bn1= nn.BatchNorm1d(256)
        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.7)
        self.dr2=nn.Dropout(p=0.6)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
    

class feat_L_READ(nn.Module):
    def __init__(self):
        super(feat_L_READ, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 128)
        self.ll2=nn.Linear(128, 128)
        self.ll3=nn.Linear(128, 64)

        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.6)
        self.dr2=nn.Dropout(p=0.5)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
    
class feat_L_SKCM(nn.Module):
    def __init__(self):
        super(feat_L_SKCM, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 128)
        self.ll2=nn.Linear(128, 512)
        self.ll3=nn.Linear(512, 64)

        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(512)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.5)
        self.dr2=nn.Dropout(p=0.5)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
    


class feat_L_LUAD(nn.Module):
    def __init__(self):
        super(feat_L_LUAD, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 128)
        self.ll2=nn.Linear(128, 256)
        self.ll3=nn.Linear(256, 64)

        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(256)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.5)
        self.dr2=nn.Dropout(p=0.6)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
    
class feat_L_KIRC(nn.Module):
    def __init__(self):
        super(feat_L_KIRC, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 128)
        self.ll2=nn.Linear(128, 128)
        self.ll3=nn.Linear(128, 64)

        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.5)
        self.dr2=nn.Dropout(p=0.7)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
class feat_L_PRAD(nn.Module):
    def __init__(self):
        super(feat_L_PRAD, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 128)
        self.ll2=nn.Linear(128, 128)
        self.ll3=nn.Linear(128, 64)

        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.5)
        self.dr2=nn.Dropout(p=0.8)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
    
class feat_L_HNSC(nn.Module):
    def __init__(self):
        super(feat_L_HNSC, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 128)
        self.ll2=nn.Linear(128, 128)
        self.ll3=nn.Linear(128, 64)

        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.6)
        self.dr2=nn.Dropout(p=0.7)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
    

    

class feat_L_BLCA(nn.Module):
    def __init__(self):
        super(feat_L_BLCA, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 512)
        self.ll2=nn.Linear(512, 256)
        self.ll3=nn.Linear(256, 64)

        self.bn1= nn.BatchNorm1d(512)
        self.bn2= nn.BatchNorm1d(256)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.7)
        self.dr2=nn.Dropout(p=0.6)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
    

class feat_L_COAD(nn.Module):
    def __init__(self):
        super(feat_L_COAD, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 128)
        self.ll2=nn.Linear(128, 128)
        self.ll3=nn.Linear(128, 64)

        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.7)
        self.dr2=nn.Dropout(p=0.5)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
    
class feat_L_KIRP(nn.Module):
    def __init__(self):
        super(feat_L_KIRP, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(12, 128)
        self.ll2=nn.Linear(128, 128)
        self.ll3=nn.Linear(128, 64)

        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.6)
        self.dr2=nn.Dropout(p=0.7)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb