'''
Define various sampling procedures to sample after training
This includes using predefined sensor measurements with inpainting
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import torch.nn as nn
import numpy as np

class DarcyPressureRandom(nn.Module):
    def __init__(self,
                 sensors):
        super().__init__()

        self.sensors = sensors # sensors per sample in batch

    def forward(self, data_batch):
        assert len(data_batch.shape) == 4

        batch_size = data_batch.shape[0]
        channels = data_batch.shape[1]
        dim0 = data_batch.shape[2]
        dim1 = data_batch.shape[3]

        indices = torch.zeros((self.sensors*batch_size, 4), dtype=torch.long)
        batch_inds = torch.arange(batch_size).view(-1, 1).repeat(1, self.sensors).view(-1)
        channel_inds = torch.zeros((self.sensors, )).repeat(batch_size) # channel 0 = pressure
        x0 = torch.arange(dim0)
        x1 = torch.arange(dim1)
        g0, g1 = torch.meshgrid(x0, x1)
        combined = torch.stack((g0, g1), dim=2)
        combined = combined.view(-1, 2)
        dim_inds = torch.randperm(dim0*dim1)[:self.sensors].repeat(batch_size)
        dim0_inds = combined[dim_inds, 0]
        dim1_inds = combined[dim_inds, 1]

        indices[:, 0] = batch_inds
        indices[:, 1] = channel_inds
        indices[:, 2] = dim0_inds
        indices[:, 3] = dim1_inds

        values = torch.zeros_like(data_batch)
        values[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = data_batch[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        return values, indices

class DarcyRandom(nn.Module):
    def __init__(self,
                 sensors):
        super().__init__()

        self.sensors = sensors # sensors per sample in batch

    def forward(self, data_batch):
        assert len(data_batch.shape) == 4

        batch_size = data_batch.shape[0]
        channels = data_batch.shape[1]
        dim0 = data_batch.shape[2]
        dim1 = data_batch.shape[3]

        indices = torch.zeros((self.sensors*batch_size*2, 4), dtype=torch.long)

        # pressure field
        batch_inds = torch.arange(batch_size).view(-1, 1).repeat(1, self.sensors).view(-1)
        channel_inds = torch.zeros((self.sensors, )).repeat(batch_size) # channel 0 = pressure
        x0 = torch.arange(dim0)
        x1 = torch.arange(dim1)
        g0, g1 = torch.meshgrid(x0, x1)
        combined = torch.stack((g0, g1), dim=2)
        combined = combined.view(-1, 2)
        dim_inds = torch.randperm(dim0*dim1)[:self.sensors].repeat(batch_size)
        dim0_inds = combined[dim_inds, 0]
        dim1_inds = combined[dim_inds, 1]

        indices[:self.sensors*batch_size, 0] = batch_inds
        indices[:self.sensors*batch_size, 1] = channel_inds
        indices[:self.sensors*batch_size, 2] = dim0_inds
        indices[:self.sensors*batch_size, 3] = dim1_inds

        # permeability
        channel_inds = torch.ones((self.sensors, )).repeat(batch_size) # channel 0 = pressure
        indices[self.sensors*batch_size:, 0] = batch_inds
        indices[self.sensors*batch_size:, 1] = channel_inds
        indices[self.sensors*batch_size:, 2] = dim0_inds
        indices[self.sensors*batch_size:, 3] = dim1_inds

        values = torch.zeros_like(data_batch)
        values[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = data_batch[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        return values, indices
        
class DarcyPressureDiagonal(nn.Module):
    def __init__(self,
                 sensors):
        super().__init__()

        self.sensors = sensors # sensors per sample in batch

    def forward(self, data_batch):
        assert len(data_batch.shape) == 4

        batch_size = data_batch.shape[0]
        channels = data_batch.shape[1]
        dim0 = data_batch.shape[2]
        dim1 = data_batch.shape[3]

        dim_small = np.minimum(dim0, dim1)

        indices = torch.zeros((dim_small*batch_size, 4), dtype=torch.long)

        # pressure field diagonal
        batch_inds = torch.arange(batch_size).view(-1, 1).repeat(1, dim_small).view(-1)
        channel_inds = torch.zeros((dim_small, )).repeat(batch_size) # channel 0 = pressure
        dim0_inds = torch.arange(dim_small).repeat(batch_size)
        dim1_inds = torch.arange(dim_small).repeat(batch_size)

        #dim0_inds = torch.cat((dim0_inds, torch.arange(dim_small))).repeat(batch_size)
        #dim1_inds = torch.cat((dim1_inds, torch.arange(dim_small).flip(0))).repeat(batch_size)

        indices[:, 0] = batch_inds
        indices[:, 1] = channel_inds
        indices[:, 2] = dim0_inds
        indices[:, 3] = dim1_inds

        values = torch.zeros_like(data_batch)
        values[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = data_batch[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        return values, indices

class DarcyPressureTriDiag(nn.Module):
    def __init__(self,
                 sensors):
        super().__init__()

        self.sensors = sensors # sensors per sample in batch

    def forward(self, data_batch):
        assert len(data_batch.shape) == 4

        batch_size = data_batch.shape[0]
        channels = data_batch.shape[1]
        dim0 = data_batch.shape[2]
        dim1 = data_batch.shape[3]

        dim_small = np.minimum(dim0, dim1)

        indices = torch.zeros((batch_size*(dim_small*3-2), 4), dtype=torch.long)

        # pressure field diagonal
        batch_inds = torch.arange(batch_size).view(-1, 1).repeat(1, dim_small*3-2).view(-1)
        channel_inds = torch.zeros((3*dim_small-2, )).repeat(batch_size) # channel 0 = pressure

        # main diagonal
        dim0_inds = torch.arange(dim_small)#.repeat(batch_size)
        dim1_inds = torch.arange(dim_small)#.repeat(batch_size)
        
        # lower diagonal
        dim0_inds = torch.cat((dim0_inds, torch.arange(dim_small-1)))#.repeat(batch_size)
        dim1_inds = torch.cat((dim1_inds, torch.arange(dim_small-1)+1))#.repeat(batch_size)

        # upper diagonal
        dim0_inds = torch.cat((dim0_inds, torch.arange(dim_small-1)+1)).repeat(batch_size)
        dim1_inds = torch.cat((dim1_inds, torch.arange(dim_small-1))).repeat(batch_size)

        indices[:, 0] = batch_inds
        indices[:, 1] = channel_inds
        indices[:, 2] = dim0_inds
        indices[:, 3] = dim1_inds

        values = torch.zeros_like(data_batch)
        values[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = data_batch[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        return values, indices

