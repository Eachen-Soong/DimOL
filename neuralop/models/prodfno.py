from functools import partialmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.spectral_convolution import SpectralConv2d
from ..layers.spherical_convolution import SphericalConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.mlp import MLP
from ..layers.quad_layer import QuadPath, ProductLayer

class ProdFNO_2D(nn.Module):
    def __init__(self, in_dim, appended_dim, out_dim, modes1, modes2, width,
                 num_prod, skip_connection=True, use_position=True, input_prods=None):
        super(ProdFNO_2D, self).__init__()
        self.in_dim = in_dim
        self.appended_dim = appended_dim # 2 parts: from the dataset and from the model itself
        self.out_dim = out_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_prod = num_prod
        self.width = width
        self.padding = 9
        self.skip_connection = skip_connection
        self.use_position = use_position
        
        self.conv0 = SpectralConv2d(self.width, self.width, [self.modes1, self.modes2])
        self.conv1 = SpectralConv2d(self.width, self.width, [self.modes1, self.modes2])
        self.conv2 = SpectralConv2d(self.width, self.width, [self.modes1, self.modes2])
        self.conv3 = SpectralConv2d(self.width, self.width, [self.modes1, self.modes2])

        self.w0 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w1 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w2 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w3 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')

        self.prod0 = ProductLayer(2 * self.width, num_prod, self.width)
        self.prod1 = ProductLayer(2 * self.width, num_prod, self.width)
        self.prod2 = ProductLayer(2 * self.width, num_prod, self.width)
        self.prod3 = ProductLayer(2 * self.width, num_prod, self.width)

        # self.q0 = QuadraticLayer2D(self.in_dim + self.appended_dim, 3, 2, self.width)
        num_prod0 = 4
        self.q0 = QuadPath(self.in_dim + self.appended_dim, num_prod0, num_prod0, num_prod0)
        assert self.width - self.in_dim - self.appended_dim - num_prod0 > 0, "width <= in_dim + appended_dim + num_prod !"
        self.fc0 = MLP(self.in_dim + self.appended_dim + num_prod0, self.width - self.in_dim - self.appended_dim - num_prod0)
        # self.fc1 = nn.Linear(self.width, self.out_dim)
        self.fc1 = MLP(self.width, self.out_dim)

        self.quads = [self.q0]

        if input_prods is not None:
            self.q0.prep_stage = False
            shp = self.q0.prod_indices.shape
            self.q0.prod_indices = input_prods
            assert input_prods.shape == shp, "input_prods.shape doesn't match!"

    def forward(self, x, **kwargs):
        if self.use_position:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)        

        q = self.q0(x)
        x = torch.cat((x, q), dim=1) # cat product terms
        x = torch.cat((x, self.fc0(x)), dim=1)
        # x = x.permute(0, 3, 1, 2)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        xo = torch.cat((x1, x2), dim=1) # [batch, width, x, y] -> [batch, 2*width, x, y]
        xo = self.prod0(xo)
        xo = torch.tanh(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod1(xo)
        xo = torch.tanh(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod2(xo)
        xo = torch.tanh(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod3(xo)
        xo = torch.tanh(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        # x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x
    

    def get_grid(self, shape, device):
        batchsize, _, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)


    def extract(self, threshold, epoch, max_epoch=30):
        for (idx, quad) in enumerate(self.quads):
            if quad.prep_stage == False:
                continue
            proportion = quad.replace_previous(threshold)
            if_pass = proportion >= threshold
            if torch.all(if_pass) or epoch >= max_epoch:
                print(f'Extracted quadratic term {idx} at epoch {epoch}')
                print(quad.prod_indices)

    def get_ortho_loss(self):
        ortho_loss = 0.0
        for (idx, quad) in enumerate(self.quads):
            if quad.prep_stage == False:
                continue
            ortho_loss += quad.orthogonality_loss()
        return ortho_loss


class ProdFNO_2D_test(nn.Module):
    def __init__(self, in_dim, appended_dim, out_dim, modes1, modes2, width,
                 num_prod, ortho_lambda=10, skip_connection=True, use_position=True, input_prods=None):
        super(ProdFNO_2D_test, self).__init__()
        self.in_dim = in_dim
        self.appended_dim = appended_dim # 2 parts: from the dataset and from the model itself
        self.out_dim = out_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_prod = num_prod
        self.width = width
        self.padding = 9
        self.skip_connection = skip_connection
        self.use_position = use_position
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w1 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w2 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w3 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')

        self.prod0 = ProductLayer(2 * self.width, num_prod, self.width)
        self.prod1 = ProductLayer(2 * self.width, num_prod, self.width)
        self.prod2 = ProductLayer(2 * self.width, num_prod, self.width)
        self.prod3 = ProductLayer(2 * self.width, num_prod, self.width)

        # self.q0 = QuadraticLayer2D(self.in_dim + self.appended_dim, 3, 2, self.width)
        num_prod0 = 4
        self.q0 = QuadPath(self.in_dim + self.appended_dim, num_prod0, num_prod0, num_prod0)
        assert self.width - self.in_dim - self.appended_dim - num_prod0 > 0, "width <= in_dim + appended_dim + num_prod !"
        self.fc0 = nn.Linear(self.in_dim + self.appended_dim + num_prod0, self.width - self.in_dim - self.appended_dim - num_prod0)
        self.fc1 = nn.Linear(self.width, self.out_dim)

        self.quads = [self.q0]

        if input_prods is not None:
            self.q0.prep_stage = False
            shp = self.q0.prod_indices.shape
            self.q0.prod_indices = input_prods
            assert input_prods.shape == shp, "input_prods.shape doesn't match!"

    def forward(self, x):
        if self.use_position:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)

        q = self.q0(x)
        x = torch.cat((x, q), dim=-1) # cat product terms
        x = torch.cat((x, self.fc0(x)), dim=-1)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        xo = torch.cat((x1, x2), dim=1) # [batch, width, x, y] -> [batch, 2*width, x, y]
        xo = self.prod0(xo)
        xo = torch.tanh(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod1(xo)
        xo = torch.tanh(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod2(xo)
        xo = torch.tanh(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod3(xo)
        xo = torch.tanh(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def extract(self, threshold, epoch, max_epoch=30):
        for (idx, quad) in enumerate(self.quads):
            if quad.prep_stage == False:
                continue
            proportion = quad.replace_previous(threshold)
            if_pass = proportion >= threshold
            if torch.all(if_pass) or epoch >= max_epoch:
                print(f'Extracted quadratic term {idx} at epoch {epoch}')
                print(quad.prod_indices)

    def get_ortho_loss(self):
        ortho_loss = 0.0
        for (idx, quad) in enumerate(self.quads):
            if quad.prep_stage == False:
                continue
            ortho_loss += quad.orthogonality_loss()
        return ortho_loss