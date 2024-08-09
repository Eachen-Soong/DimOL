from pathlib import Path
import torch
import h5py
import numpy as np
from einops import repeat
import scipy
from torch.utils.data import ConcatDataset, Dataset, DataLoader

TEMPERATURE = 'temperature'
VELX = 'velx'
VELY = 'vely'
PRESSURE = 'pressure'
DFUN = 'dfun'
X = 'x'
Y = 'y'
CONSTS = ['ins_gravy']

class BubbleMLDataset(Dataset):
    def __init__(self, data):
        self.data = data
        field_shape = self.data[TEMPERATURE][:].shape
        self.timesteps = field_shape[0]
        self.space_dim = field_shape[1:]
        params = {}
        for i in range(self.data['real-runtime-params'].shape[0]):
            raw_key, value = self.data['real-runtime-params'][i]
            key = raw_key.decode('utf-8').rstrip()
            params[key]=value
        self.consts = torch.tensor(np.array([params[name] for name in CONSTS]))
        self.consts_spanned = repeat(self.consts, ' c -> c x y', x=self.space_dim[0], y=self.space_dim[1])
    
    def __len__(self):
        return self.timesteps - 1
    
    def _get_input(self, idx):
        r"""
        The input is the temperature, x-velocity, and y-velocity at time == idx
        """
        temp = torch.from_numpy(self.data[TEMPERATURE][idx])
        velx = torch.from_numpy(self.data[VELX][idx])
        vely = torch.from_numpy(self.data[VELY][idx])
        pres = torch.from_numpy(self.data[PRESSURE][idx])
        dfun = torch.from_numpy(self.data[DFUN][idx])
        x = torch.from_numpy(self.data[X][idx])
        y = torch.from_numpy(self.data[Y][idx])
        # returns a stack with shape [5 x Y x X]
        return torch.cat([torch.stack((temp, velx, vely, pres, dfun, x, y), dim=0), self.consts_spanned], dim=0)
    
    def _get_label(self, idx):
        r"""
        The output is the temperature at time == idx
        """
        return torch.from_numpy(self.data[TEMPERATURE][idx]).unsqueeze(0)
    
    def __getitem__(self, idx):
        r"""
        As input, get temperature and velocities at time == idx.
        As the output label, get the temperature at time == idx + 1.
        """
        input = self._get_input(idx)
        label = self._get_label(idx+1)
        return {'x': input, 'y': label}

def load_pb_gravity(
        data_path, n_train, n_test, batch_train=4, batch_test=16,
        ):
    range_gravy = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    all_files = [data_path + f'/BubbleML/PoolBoiling-Gravity-FC72-2D/gravY-{gravy}.hdf5' for gravy in range_gravy]

    selected_files = [all_files[i] for i in [1,3,4,6,7]]

    train_datasets = []; test_datasets = []
    for file_path in selected_files:
        data = h5py.File(file_path, 'r')

        train_data = {}; test_data = {}
        train_data[TEMPERATURE] = data[TEMPERATURE][:n_train, ...]
        test_data[TEMPERATURE] = data[TEMPERATURE][-n_test:, ...]
        train_data[VELX] = data[VELX][:n_train, ...]
        test_data[VELX] = data[VELX][-n_test:, ...]
        train_data[VELY] = data[VELY][:n_train, ...]
        test_data[VELY] = data[VELY][-n_test:, ...]
        train_data[PRESSURE] = data[PRESSURE][:n_train, ...]
        test_data[PRESSURE] = data[PRESSURE][-n_test:, ...]
        train_data[DFUN] = data[DFUN][:n_train, ...]
        test_data[DFUN] = data[DFUN][-n_test:, ...]
        train_data[X] = data[X][:n_train, ...]
        test_data[X] = data[X][-n_test:, ...]
        train_data[Y] = data[Y][:n_train, ...]
        test_data[Y] = data[Y][-n_test:, ...]
        train_data['real-runtime-params'] = data['real-runtime-params']
        test_data['real-runtime-params'] = data['real-runtime-params']

        train_datasets.append(BubbleMLDataset(train_data))
        test_datasets.append(BubbleMLDataset(test_data))

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_train)
    test_loader =  DataLoader(ConcatDataset(test_datasets), batch_size=batch_test)

    return train_loader, test_loader


