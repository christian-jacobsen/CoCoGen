''' 
Pytorch lightning dataloaders
Author: Christian Jacobsen, University of Michigan 2023
'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pytorch_lightning as pl
import os
import os.path as osp
import argparse


class Darcy_Dataset(Dataset):
    def __init__(self, path):
        self.root = path

        # load the sample names
        sample_names = os.listdir(osp.join(path, "data"))
        self.P_names, self.U1_names, self.U2_names = self.seperate_img_names(sample_names)
        self.P_names.sort()
        self.U1_names.sort()
        self.U2_names.sort() # all files are stored as P_xxx.npy, U1_xxx.npy, U2_xxx.npy
        self.img_mean = np.array([0, 0.194094975, 0.115737872]) # P, U1, U2
        self.img_std = np.array([0.08232874, 0.27291843, 0.12989907])

        # load permeability fields
        self.perm_names = os.listdir(osp.join(path, "permeability"))
        self.perm_names.sort()
        self.perm_mean = 1.14906847
        self.perm_std = 7.81547992

        # load the parameter values
        self.param_names = os.listdir(osp.join(path, "params"))
        self.param_names.sort()
        self.param_mean = 1.248473
        self.param_std = 0.7208982

    def seperate_img_names(self, names):
        P, U1, U2 = [], [], []
        for name in names:
            if name[0] == "P":
                P.append(name)
            elif name[0:2] == "U1":
                U1.append(name)
            elif name[0:2] == "U2":
                U2.append(name)
            else:
                raise Exception("File "+name+" isn't a pressure or velocity field!")

        return P, U1, U2

    def __len__(self):
        return len(self.P_names)

    def __getitem__(self, idx):

        W = torch.from_numpy(np.load(osp.join(self.root, "params", self.param_names[idx]))).float()
        W = (np.squeeze(W) - self.param_mean) / self.param_std
        W = W

        K = torch.from_numpy(np.load(osp.join(self.root, "permeability", self.perm_names[idx]))).float()
        K = (np.expand_dims(K, axis=0) - self.perm_mean) / self.perm_std

        P = torch.from_numpy(np.load(osp.join(self.root, "data", self.P_names[idx]))).float()
        P = (np.expand_dims(P, axis=0) - self.img_mean[0]) / self.img_std[0]

        '''
        U1 = torch.from_numpy(np.load(osp.join(self.root, "data", self.U1_names[idx]))).float()
        U1 = (np.expand_dims(U1, axis=0) - self.img_mean[1]) / self.img_std[1]

        U2 = torch.from_numpy(np.load(osp.join(self.root, "data", self.U2_names[idx]))).float()
        U2 = (np.expand_dims(U2, axis=0) - self.img_mean[2]) / self.img_std[2]
        '''

        Data = np.concatenate([P, K], axis=0)

        return Data, W

class DarcyLoader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_dataset = Darcy_Dataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class Burgers_Dataset(Dataset):
    def __init__(self, path):
        self.root = path

        # load the sample names
        self.sample_names = os.listdir(osp.join(path, "data"))
        self.img_mean = -0.751762357#-3.010598882
        self.img_std = 8.041401807#49.02098157

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):

        W = torch.tensor([0.0]).float()

        Data = torch.from_numpy(np.load(osp.join(self.root, "data", self.sample_names[idx]))).float()
        Data = (np.expand_dims(Data, axis=0) - self.img_mean) / self.img_std

        return Data, W

class BurgersLoader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_dataset = Burgers_Dataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

