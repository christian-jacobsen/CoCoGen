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
import math
import h5py


class Darcy_Dataset(Dataset):
    def __init__(self, path, subset='train', split_ratios=(0.7, 0.2, 0.1), seed=0):
        self.root = path
        np.random.seed(seed)

        # load the sample names and partition
        sample_names = os.listdir(osp.join(path, "data"))
        np.random.shuffle(sample_names)  # Shuffle the dataset
        num_samples = len(sample_names)
        num_train = int(split_ratios[0] * num_samples)
        num_val = int(split_ratios[1] * num_samples)
        
        if subset == 'train':
            sample_names = sample_names[:num_train]
        elif subset == 'val':
            sample_names = sample_names[num_train:num_train + num_val]
        elif subset == 'test':
            sample_names = sample_names[num_train + num_val:]
        
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
    def __init__(self, data_dir, batch_size=32, num_workers=8, split_ratios=(0.7, 0.2, 0.1)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Darcy_Dataset(self.data_dir, subset='train', split_ratios=self.split_ratios)
            self.val_dataset = Darcy_Dataset(self.data_dir, subset='val', split_ratios=self.split_ratios)
        if stage == 'test' or stage is None:
            self.test_dataset = Darcy_Dataset(self.data_dir, subset='test', split_ratios=self.split_ratios)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class Mod_Darcy_Dataset(Dataset):
    def __init__(self, path, subset='train', split_ratios=(0.7, 0.2, 0.1), seed=0):
        self.root = path
        np.random.seed(seed)

        # load the sample names and partition
        sample_names = os.listdir(osp.join(path, "data"))
        np.random.shuffle(sample_names)  # Shuffle the dataset
        num_samples = len(sample_names)
        num_train = int(split_ratios[0] * num_samples)
        num_val = int(split_ratios[1] * num_samples)
        
        if subset == 'train':
            sample_names = sample_names[:num_train]
        elif subset == 'val':
            sample_names = sample_names[num_train:num_train + num_val]
        elif subset == 'test':
            sample_names = sample_names[num_train + num_val:]
        
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

        K = torch.from_numpy(np.load(osp.join(self.root, "permeability", self.perm_names[idx]))).float()
        K = (np.expand_dims(K, axis=0) - self.perm_mean) / self.perm_std

        P = torch.from_numpy(np.load(osp.join(self.root, "data", self.P_names[idx]))).float()
        P = (np.expand_dims(P, axis=0) - self.img_mean[0]) / self.img_std[0]

        W = np.mean(P)

        Data = np.concatenate([P, K], axis=0)

        return Data, W

class ModDarcyLoader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8, split_ratios=(0.7, 0.2, 0.1)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Darcy_Dataset(self.data_dir, subset='train', split_ratios=self.split_ratios)
            self.val_dataset = Darcy_Dataset(self.data_dir, subset='val', split_ratios=self.split_ratios)
        if stage == 'test' or stage is None:
            self.test_dataset = Darcy_Dataset(self.data_dir, subset='test', split_ratios=self.split_ratios)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

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

class PDEBench2D_Dataset(Dataset):
    def __init__(self, file_name,
                 train_type='single', # 'single' or 'autoregressive'
                 t_train = 21,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY

        """
        
        # Define path to files
        root_path = os.path.abspath(saved_folder + file_name)
        assert file_name[-2:] != 'h5', 'HDF5 data is assumed!!'
        
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            if 'tensor' not in keys:
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                if len(idx_cfd)==3:  # 1D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          math.ceil(idx_cfd[1]/reduced_resolution_t),
                                          3],
                                         dtype=np.float32)
                    #density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[...,2] = _data   # batch, x, t, ch

                if len(idx_cfd)==4:  # 2D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          math.ceil(idx_cfd[1]/reduced_resolution_t),
                                          4],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[...,3] = _data   # batch, x, t, ch
                    
                if len(idx_cfd)==5:  # 3D
                    self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                          idx_cfd[2]//reduced_resolution,
                                          idx_cfd[3]//reduced_resolution,
                                          idx_cfd[4]//reduced_resolution,
                                          math.ceil(idx_cfd[1]/reduced_resolution_t),
                                          5],
                                         dtype=np.float32)
                    # density
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,0] = _data   # batch, x, t, ch
                    # pressure
                    _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,1] = _data   # batch, x, t, ch
                    # Vx
                    _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,2] = _data   # batch, x, t, ch
                    # Vy
                    _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,3] = _data   # batch, x, t, ch
                    # Vz
                    _data = np.array(f['Vz'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[...,4] = _data   # batch, x, t, ch

            else:  # scalar equations
                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch

                if len(_data.shape) == 4:  # 2D Darcy flow
                    # u: label
                    _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    #if _data.shape[-1]==1:  # if nt==1
                    #    _data = np.tile(_data, (1, 1, 1, 2))
                    self.data = _data
                    # nu: input
                    _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                    _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    self.data = np.concatenate([_data, self.data], axis=-1)
                    self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:num_samples_max]

        # Time steps used as initial conditions
        self.initial_step = initial_step
        self.t_train = t_train
        self.train_type = train_type
        
        self.data = torch.tensor(self.data)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # self.data.shape = (N_sample, Nx, Ny, Nt(21), Nch(4))
        
        #return self.data[idx,...,:self.initial_step,:], self.data[idx]
        if self.train_type == 'single':
            return torch.permute(self.data[idx,...,0,:], (2, 0, 1)), torch.permute(self.data[idx,...,self.t_train-1,:], (2, 0, 1))
        else:
            raise NotImplementedError

class PDEBench2D_Dataloader(pl.LightningDataModule):
    def __init__(self, 
                 file_name,
                 batch_size=32,
                 num_workers=8,
                 train_type='single', # 'single' or 'autoregressive'
                 t_train = 21,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1):
        super().__init__()
        self.saved_folder = saved_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_name = file_name
        self.train_type = train_type
        self.t_train = t_train
        self.initial_step = initial_step
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t
        self.reduced_batch = reduced_batch
        self.if_test = if_test
        self.test_ratio = test_ratio
        self.num_samples_max = num_samples_max

    def setup(self, stage: str):
        self.train_dataset = PDEBench2D_Dataset(self.file_name, self.train_type, self.t_train, self.initial_step, 
                                                self.saved_folder, self.reduced_resolution, self.reduced_resolution_t,
                                                self.reduced_batch, self.if_test, self.test_ratio, self.num_samples_max)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)