
import torch
from einops import repeat
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
import numpy as np
import h5py
import os
from ..utils import UnitGaussianNormalizer
# from .tensor_dataset import TensorDataset


class Builder(LightningDataModule):
    @property
    def batches_per_epochs(self):
        return len(self.train_dataloader())


class NSContextualBuilder(Builder):
    def __init__(self, data_path: str,
                 transform_x=None, transform_y=None,
                 append_force=True, append_mu=True,
                 encode_input=False, encoding="channel-wise"):
        super().__init__()
        data_path = os.path.expandvars(data_path)
        self.h5f = h5py.File(data_path)

        self.transform_x=transform_x
        self.transform_y=transform_y
        self.append_force=append_force
        self.append_mu=append_mu
        self.encoding=encoding

        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(len(self.h5f['train']['u'].shape)))
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            input_encoder = UnitGaussianNormalizer(self.h5f['train']['u'], reduce_dim=reduce_dims)
            self.h5f['train']['u'] = input_encoder.encode(self.h5f['train']['u'])
            self.h5f['test']['u'] = input_encoder.encode(self.h5f['test']['u'])
        else:
            input_encoder = None

        self.train_dataset = None
        self.valid_dataset = None
        self.test_datasets = {}
        
        # self.train_dataset = NavierStokesTrainingDataset(h5f['train'], ssr, k, transform_x=transform_x, transform_y=transform_y, append_force=append_force, append_mu=append_mu)
        # self.valid_dataset = NavierStokesDataset(h5f['valid'], ssr, k, transform_x=transform_x, transform_y=transform_y, append_force=append_force, append_mu=append_mu)
        # self.test_dataset = NavierStokesDataset(h5f['test'], ssr, k, transform_x=transform_x, transform_y=transform_y, append_force=append_force, append_mu=append_mu)

    def get_output_encoder(self):
        assert self.train_dataset is not None, "First get train loader!"
        tmp_loader = DataLoader(self.train_dataset,
                            batch_size=len(self.train_dataset),
                            shuffle=True,
                            drop_last=False,)
        for _, y_train in tmp_loader:
            break
        if self.encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif self.encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(y_train, reduce_dim=reduce_dims)
        y_train = output_encoder.encode(y_train)
        return output_encoder

    def get_train_dataloader(self, n_train, ssr, k, batch_size) -> DataLoader:
        self.train_dataset = NavierStokesTrainingDataset(
            self.h5f['train'], n_train, ssr, k, transform_x=self.transform_x, transform_y=self.transform_y, 
            append_force=self.append_force, append_mu=self.append_mu)
        loader = DataLoader(self.train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,)
        for item in loader:
            print(item)
            break
        return loader

    def get_val_dataloader(self, n_data, ssr, k, batch_size) -> DataLoader:
        self.valid_dataset = NavierStokesDataset(
            self.h5f['valid'], n_data, ssr, k, transform_x=self.transform_x, transform_y=self.transform_y, 
            append_force=self.append_force, append_mu=self.append_mu)
        loader = DataLoader(self.valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,)
        return loader

    def get_one_test_dataloader(self, n_data, ssr, k, batch_size) -> DataLoader:
        self.test_datasets[ssr] = (NavierStokesDataset(
            self.h5f['test'], n_data, ssr, k, transform_x=self.transform_x, transform_y=self.transform_y, 
            append_force=self.append_force, append_mu=self.append_mu))
        loader = DataLoader(self.test_datasets[ssr],
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,)
        return loader


class NavierStokesTrainingDataset(Dataset):
    def __init__(self, data, n_train, ssr, k, transform_x=None, transform_y=None, append_force=True, append_mu=True):
        self.u = torch.tensor(data['u'][:n_train, :, :, :])
        self.f = torch.tensor(data['f'][:n_train, :, :])
        self.mu = torch.tensor(data['mu'][:n_train])
        self.ssr = ssr
        self.k = k
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.append_force = append_force
        self.append_mu = append_mu
        self.constant_force = len(self.f.shape) == 3

        self.B = self.u.shape[0]
        self.T = self.u.shape[-1] - k

    def __len__(self):
        return self.B * self.T

    def __getitem__(self, idx):
        b = idx // self.T
        t = idx % self.T
        if self.constant_force:
            f = self.f[b, ::self.ssr, ::self.ssr]
        else:
            f = self.f[b, ::self.ssr, ::self.ssr, t + self.k]
        
        return {
            'x': self.u[b, ::self.ssr, ::self.ssr, t:t+1],
            'y': self.u[b, ::self.ssr, ::self.ssr, t+self.k:t+self.k+1],
            'mu': self.mu[b],
            'f': f,
        }


class NavierStokesDataset(Dataset):
    def __init__(self, data, n_data, ssr, k, transform_x=None, transform_y=None, append_force=True, append_mu=True):
        self.u = torch.tensor(data['u'][:n_data, :, :, :])
        self.f = torch.tensor(data['f'][:n_data, :, :])
        self.mu = torch.tensor(data['mu'][:n_data])
        self.ssr = ssr
        self.k = k
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.append_force = append_force
        self.append_mu = append_mu
        self.constant_force = len(self.f.shape) == 3

        self.B = self.u.shape[0]
        self.T = self.u.shape[-1] - k
        self.times = np.arange(0, 20, 0.1 * k)

    def __len__(self):
        return self.B

    def __getitem__(self, b):
        if self.constant_force:
            f = self.f[b, ::self.ssr, ::self.ssr]
        else:
            f = self.f[b, ::self.ssr, ::self.ssr, ::self.k]

        batch = {
            'data': self.u[b, ::self.ssr, ::self.ssr, ::self.k],
            'mu': self.mu[b],
            'f': f,
            'times': self.times}
        return self._build_feature(batch)
    
    def _build_feature(self, batch):
        data = batch['data']
        inputs = data

        B, *dim_sizes, T = inputs.shape
        X, Y = dim_sizes
        # data.shape == [batch_size, *dim_sizes, total_steps]

        inputs = repeat(inputs, '... -> ... 1').to(self.device)
        # inputs.shape == [batch_size, *dim_sizes, total_steps, 1]

        n_steps = self.n_steps or T - 1
        inputs = inputs[..., -n_steps-1:-1, :]
        # inputs.shape == [batch_size, *dim_sizes, n_steps, 3]

        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            x = self.transform_y(x)

        xx = inputs

        if self.append_force:
            if len(batch['f'].shape) == 3:
                force = repeat(batch['f'], 'b m n -> b m n t 1',
                               t=xx.shape[-2]).to(self.device)
            elif len(batch['f'].shape) == 4:
                f = batch['f'][..., -n_steps:]
                force = repeat(f, 'b m n t -> b m n t 1').to(self.device)

            xx = torch.cat([xx, force], dim=-1)

        if self.append_mu:
            mu = repeat(batch['mu'], 'b -> b m n t 1',
                        m=X, n=Y, t=xx.shape[-2]).to(self.device)
            xx = torch.cat([xx, mu], dim=-1)
        
        return xx
    
# class NS_Trainer(Trainer):
#     def __init__(self, model, wandb_log=False, tensorboard_log=False, tensorboard_writer=None, device=None,
#                  log_test_interval=1, log_output=False, verbose=True,
#                  append_force=True, append_mu=True, n_steps=0
#                  ):
#         super().__init__(model, wandb_log, tensorboard_log,tensorboard_writer, device,
#                  log_test_interval, log_output, verbose)
#         self.append_force = append_force
#         self.append_mu = append_mu
#         self.n_steps = n_steps

#     def _build_features(self, batch):
#         x = batch['x']
#         x = x.to(self.device)
#         B, *dim_sizes, _ = x.shape
#         X, Y = dim_sizes
#         # data.shape == [batch_size, *dim_sizes]

#         if self.append_force:
#             f = repeat(batch['f'], 'b m n -> b m n 1')
#             f = f.to(self.device)
#             x = torch.cat([x, f], dim=-1)

#         if self.append_mu:
#             mu = repeat(batch['mu'], 'b -> b m n 1', m=X, n=Y)
#             mu = mu.to(self.device)
#             x = torch.cat([x, mu], dim=-1)  

#         return x

#     def _get_loss(self, out, y, criterion=LpLoss()):
#         return criterion(out.float(), y)

#     def training_step(self, batch, optimizer, criterion=LpLoss()):
#         x = batch['x']; y = batch['y']
#         x = x.to(self.device)
#         y = y.to(self.device)
#         x = self._build_features(batch)
#         x = x.to(self.device)

#         optimizer.zero_grad()
#         out = self.model(x)
#         loss = self._get_loss(out, y, criterion)
#         loss.backward()
#         optimizer.step()
        
#         return loss.item()
    
#     def valid_step(self, batch, criterion=LpLoss()):
#         data = batch['data']
#         inputs = data

#         B, *dim_sizes, T = inputs.shape
#         X, Y = dim_sizes
#         # data.shape == [batch_size, *dim_sizes, total_steps]

#         inputs = repeat(inputs, '... -> ... 1').to(self.device)
#         # inputs.shape == [batch_size, *dim_sizes, total_steps, 1]

#         n_steps = self.n_steps or T - 1
#         inputs = inputs[..., -n_steps-1:-1, :]
#         # inputs.shape == [batch_size, *dim_sizes, n_steps, 3]

#         xx = inputs

#         if self.append_force:
#             if len(batch['f'].shape) == 3:
#                 force = repeat(batch['f'], 'b m n -> b m n t 1',
#                                t=xx.shape[-2]).to(self.device)
#             elif len(batch['f'].shape) == 4:
#                 f = batch['f'][..., -n_steps:]
#                 force = repeat(f, 'b m n t -> b m n t 1').to(self.device)

#             xx = torch.cat([xx, force], dim=-1)

#         if self.append_mu:
#             mu = repeat(batch['mu'], 'b -> b m n t 1',
#                         m=X, n=Y, t=xx.shape[-2]).to(self.device)
#             xx = torch.cat([xx, mu], dim=-1)


#         yy = data[:, ..., -n_steps:].to(self.device)
#         # yy.shape == [batch_size, *dim_sizes, n_steps]

#         loss = 0
#         step_losses = []
#         # We predict one future one step at a time
#         pred_layer_list = []
        
#         for t in range(n_steps):
#             if t==0:
#                 im = xx[..., 0, :]
#             else:
#                 if self.append_force:
#                     im = torch.cat([im, force[..., t, :]], dim=-1)
#                 if self.append_mu:
#                     im = torch.cat([im, mu[..., t, :]], dim=-1)
#             x = im
#             # x.shape == [batch_size, *dim_sizes, 1 + appended_dim]

#             out = self.model(x)
#             im = out

#             y = yy[..., t]
#             l = criterion(im.reshape(B, -1), y.reshape(B, -1))
#             step_losses.append(l)
#             if l > 1000:
#                 print(f't = {t}, loss = {l}')
#             loss += l
#             preds = im if t == 0 else torch.cat((preds, im), dim=-1)
#         # preds.shape == [batch_size, *dim_sizes, n_steps]
#         # yy.shape == [batch_size, *dim_sizes, n_steps]

#         return loss, step_losses, preds, pred_layer_list
    
# class NS_Trainer_Quad(NS_Trainer):
#     def __init__(self, model, extract_threshold:float, wandb_log=False, tensorboard_log=False, tensorboard_writer=None, device=None, log_test_interval=1, log_output=False, verbose=True, append_force=True, append_mu=True, n_steps=0):
#         super().__init__(model, wandb_log, tensorboard_log, tensorboard_writer, device, log_test_interval, log_output, verbose, append_force, append_mu, n_steps)
#         self.extract_threshold = extract_threshold

#     def _get_loss(self, out, y, criterion=LpLoss()):
#         loss = criterion(out.float(), y)
#         loss += self.model.get_ortho_loss()
#         return loss

#     def on_epoch_end(self, epoch):
#         self.model.extract(self.extract_threshold, epoch)
#         self.model.quads[0].orthonormalize_prep()
        
