import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import argparse
import torch
import numpy as np
import math
import os
from neuralop.models import FNO
from neuralop import LpLoss, H1Loss
from neuralop.datasets.darcy import load_darcy_mat
from neuralop.utils import count_params


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser('Latent Spectral Models', add_help=False)
# dataset
parser.add_argument('--data-path', default='./data', type=str, help='dataset folder')
parser.add_argument('--ntotal', default=1200, type=int, help='number of overall data')
parser.add_argument('--ntrain', default=1000, type=int, help='number of train set')
parser.add_argument('--ntest', default=200, type=int, help='number of test set')
parser.add_argument('--in_dim', default=1, type=int, help='input data dimension')
parser.add_argument('--out_dim', default=1, type=int, help='output data dimension')

parser.add_argument('--h', default=1, type=int, help='input data height')
parser.add_argument('--w', default=1, type=int, help='input data width')
parser.add_argument('--T-in', default=10, type=int,
                    help='input data time points (only for temporal related experiments)')
parser.add_argument('--T-out', default=10, type=int,
                    help='predict data time points (only for temporal related experiments)')
parser.add_argument('--h-down', default=1, type=int, help='height downsampe rate of input')
parser.add_argument('--w-down', default=1, type=int, help='width downsampe rate of input')
# optimization
parser.add_argument('--batch-size', default=20, type=int, help='batch size of training')
parser.add_argument('--learning-rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=501, type=int, help='training epochs')
parser.add_argument('--step-size', default=100, type=int, help='interval of model save')
parser.add_argument('--gamma', default=0.5, type=float, help='parameter of learning rate scheduler')
# Model parameters
parser.add_argument('--model', default='lsm', type=str, help='model name')
parser.add_argument('--d-model', default=32, type=int, help='channels of hidden variates')
parser.add_argument('--num-basis', default=12, type=int, help='number of basis operators')
parser.add_argument('--num-token', default=4, type=int, help='number of latent tokens')
parser.add_argument('--patch-size', default='3,3', type=str, help='patch size of different dimensions')
parser.add_argument('--padding', default='3,3', type=str, help='padding size of different dimensions')
# save
parser.add_argument('--model-save-path', default='./checkpoints/', type=str, help='model save path')
parser.add_argument('--model-save-name', default='lsm.pt', type=str, help='model name')


################################################################
# configs
################################################################
args = parser.parse_args()
TRAIN_PATH = args.data_path + '/piececonst_r241_N1024_smooth1.mat'
# TRAIN_PATH = os.path.join(args.data_path, './piececonst_r421_N1024_smooth1.mat')
# TEST_PATH = os.path.join(args.data_path, './piececonst_r421_N1024_smooth2.mat')

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

################################################################
# models
################################################################
model = FNO(in_channels=3, n_modes=(21, 21), hidden_channels=32, 
            projection_channels=64, factorization='tucker', rank=1,
            # channel_mixing='prod-layer',
            n_layers=4
            )
print(count_params(model))
model.to(device='cuda')
################################################################
# load data and data normalization
################################################################
# reader = MatReader(TRAIN_PATH)
# x_train = reader.read_field('coeff')[:ntrain, ::r1, ::r2][:, :s1, :s2]
# y_train = reader.read_field('sol')[:ntrain, ::r1, ::r2][:, :s1, :s2]

# reader.load_file(TEST_PATH)
# x_test = reader.read_field('coeff')[:ntest, ::r1, ::r2][:, :s1, :s2]
# y_test = reader.read_field('sol')[:ntest, ::r1, ::r2][:, :s1, :s2]

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)

# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)
# y_normalizer.cuda()



# x_train = x_train.reshape(ntrain, s1, s2, 1)
# x_test = x_test.reshape(ntest, s1, s2, 1)

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
#                                           shuffle=False)

train_loader, test_loaders, y_normalizer= load_darcy_mat(
    data_path=TRAIN_PATH, n_train=768, n_tests=[256], batch_size=32, test_batch_sizes=[32],
    )
################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=1e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(d=2, p=2)


for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for item in train_loader:
        x = item['x']
        y = item['y']
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s1, s2)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loaders[0]:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s1, s2)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)
    if ep % step_size == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name))
