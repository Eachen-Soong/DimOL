import torch
import numpy as np
import torch.utils

from neuralop.models import F_FNO2D
# from neuralop.models import FNO_2D_test1

from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

import os
import sys
import time
import shutil
import tempfile
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import math
import os
# from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import time
localtime = time.localtime(time.time())
time_now = f"{localtime.tm_mon}-{localtime.tm_mday}-{localtime.tm_hour}-{localtime.tm_min}"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

import argparse

def get_parser():
    parser = argparse.ArgumentParser('FNO Models', add_help=False)
    parser.add_argument('--model', type=str, default='FNO')
    parser.add_argument('--model_name',  type=str, default='FNO')
    # # # Data Loader Configs # # #
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_test', type=int, default=490)
    parser.add_argument('--batch_size', type=int, default=32) #
    # parser.add_argument('--test_batch_size', type=int, default=64) #
    parser.add_argument('--train_subsample_rate', type=int, default=1)
    parser.add_argument('--test_subsample_rate', type=int, default=1)
    parser.add_argument('--data_path', default='../data/airfoil', type=str, help='dataset folder')
    parser.add_argument('--data_name', type=str, default='AirFoil', help="the name of dataset")
    # # # Model Configs # # #
    parser.add_argument('--n_modes', type=int, default=21) #
    parser.add_argument('--num_prod', type=int, default=2) #
    parser.add_argument('--n_layers', type=int, default=4) ##
    parser.add_argument('--raw_in_channels', type=int, default=2, help='TorusLi: 1; ns_contextual: 3; AirFoil: 2')
    parser.add_argument('--hidden_channels', type=int, default=32) #
    parser.add_argument('--lifting_channels', type=int, default=256) #
    parser.add_argument('--projection_channels', type=int, default=64) #
    parser.add_argument('--factorization', type=str, default='tucker') #####
    parser.add_argument('--channel_mixing', type=str, default='', help='') #####
    parser.add_argument('--rank', type=float, default=0.42, help='the compression rate of tensor') #
    parser.add_argument('--load_path', type=str, default='', help='load checkpoint')
    # # # Optimizer Configs # # #
    parser.add_argument('--lr', type=float, default=1e-2) #
    parser.add_argument('--weight_decay', type=float, default=1e-4) #
    parser.add_argument('--scheduler_steps', type=int, default=100) #
    parser.add_argument('--scheduler_gamma', type=float, default=0.5) #
    parser.add_argument('--train_loss', type=str, default='l2', help='h1 or l2') #
    # # # Log and Save Configs # # #
    parser.add_argument('--log_path', type=str, default='./runs')
    parser.add_argument('--save_path', type=str, default='./ckpt')
    parser.add_argument('--prefix', type=str, default='', help='prefix of log and save file')
    parser.add_argument('--time_suffix', type=bool, default=True, help='whether to use program start time as suffix')
    parser.add_argument('--config_details', type=bool, default=True, help='whether to include config details to the log and save file name')
    parser.add_argument('--log_interval', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=20)
    # # # Trainer Configs # # #
    parser.add_argument('--epochs', type=int, default=501) #
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--random_seed', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)

    return parser


def run(args):
    seed = args.seed
    if args.random_seed:
        import random
        seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    verbose = args.verbose

    # # # Data Preparation # # #
    INPUT_X = os.path.join(args.data_path, './naca/NACA_Cylinder_X.npy')
    INPUT_Y = os.path.join(args.data_path, './naca/NACA_Cylinder_Y.npy')
    OUTPUT_Sigma = os.path.join(args.data_path, './naca/NACA_Cylinder_Q.npy')

    n_epochs = args.epochs
    ntrain = args.n_train
    ntest = args.n_test
    N = ntrain + ntest
    batch_size = args.batch_size
    # test_batch_size = args.test_batch_size

    # plot_step_size = 10
    log_interval = args.log_interval
    save_interval = args.save_interval

    n_modes = args.n_modes
    num_prod = args.num_prod

    h = 221
    w = 51
    r1 = int(args.train_subsample_rate)
    r2 = int(args.train_subsample_rate)
    s1 = int(((h - 1) / r1) + 1)
    s2 = int(((w - 1) / r2) + 1)
    # test_subsample_rate = args.test_subsample_rate


    ################################################################
    # load data and data normalization
    ################################################################
    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 4]
    output = torch.tensor(output, dtype=torch.float)
    # print(input.shape, output.shape)

    x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    x_train = x_train.reshape(ntrain, 2, s1, s2)
    x_test = x_test.reshape(ntest, 2, s1, s2)
    # print(x_train.shape, x_test.shape)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                            shuffle=False)


    # # # Loss Definition # # #
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    if args.train_loss == 'h1':
        train_loss = h1loss
    elif args.train_loss == 'l2':
        train_loss = l2loss
    else: assert False, "Unsupported training loss!"
    eval_losses={'h1': h1loss, 'l2': l2loss}

    # # # Model Definition # # #
    n_modes=args.n_modes
    num_prod=args.num_prod
    in_channels = args.raw_in_channels
    model = F_FNO2D(in_channels=in_channels, n_modes=(n_modes, n_modes), hidden_channels=args.hidden_channels, lifting_channels=args.lifting_channels,
                projection_channels=args.projection_channels, n_layers=args.n_layers, factorization=args.factorization, channel_mixing=args.channel_mixing, rank=args.rank, num_prod=num_prod)
    # model = FNO_2D_test1(in_dim=2, appended_dim=2, out_dim=1,
    #            modes1=n_modes, modes2=n_modes, width=args.hidden_channels)
    if args.load_path != '':
        model.load_state_dict(torch.load(args.load_path))

    model = model.to(device)

    n_params = count_params(model)
    print(f'\nOur model has {n_params} parameters.')
    sys.stdout.flush()

    # # # Optimizer Definition # # #
    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_steps, gamma=args.scheduler_gamma)

    if verbose:
        print('\n### MODEL ###\n', model)
        print('\n### OPTIMIZER ###\n', optimizer)
        print('\n### SCHEDULER ###\n', scheduler)
        print('\n### LOSSES ###')
        print(f'\n * Train: {train_loss}')
        print(f'\n * Test: {eval_losses}')
        sys.stdout.flush()

    # # # Logs and Saves Definition (path and file name) # # #
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    file_name = f'{args.data_name}_{args.model_name}'
    prefix = args.prefix
    if prefix != '': file_name = file_name + '_' + prefix
    # config_name = ''
    config_file_path=''
    if args.config_details:
        # config_name = f'_b{args.batch_size}_mode{args.n_modes}_prod{args.num_prod}_layer{args.n_layers}_hid{args.hidden_channels}_lift{args.lifting_channels}_proj{args.projection_channels}_fact-{args.factorization}_rank{args.rank}_mix-{args.channel_mixing}_pos-enc-{args.pos_encoding}_lr{args.lr}_wd{args.weight_decay}_sche-step{args.scheduler_steps}_gamma{args.scheduler_gamma}_loss{args.train_loss}'
        config_file_path = f"/layer_{args.n_layers}/fact-{args.factorization}/rank_{args.rank}/mix-{args.channel_mixing}/prod_{args.num_prod}/loss-{args.train_loss}/mode_{args.n_modes}/hid_{args.hidden_channels}/lift_{args.lifting_channels}/proj_{args.projection_channels}/b_{args.batch_size}/lr_{args.lr}/wd_{args.weight_decay}/sche-step_{args.scheduler_steps}/gamma_{args.scheduler_gamma}/"
    time_name = ''
    if args.time_suffix:
        localtime = time.localtime(time.time())
        time_name = f"{localtime.tm_mon}-{localtime.tm_mday}-{localtime.tm_hour}-{localtime.tm_min}"
    # file_name = file_name + config_name + time_name
    file_name = file_name + config_file_path + time_name

    log_dir = args.log_path
    if log_dir[-1]!='/': log_dir = log_dir + '/'
    log_dir = log_dir + file_name
    save_dir = args.save_path
    if save_dir[-1]!='/': save_dir = save_dir + '/'
    save_dir = save_dir + file_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    # log_dir = Path(log_dir)
    # save_dir = Path(save_dir)

    writer=None
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    for ep in range(n_epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)
            loss = train_loss(out.squeeze(1).view(out.shape[0], -1), y.view(out.shape[0], -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)
                test_l2 += train_loss(out.squeeze(1).view(out.shape[0], -1), y.view(out.shape[0], -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        # print(ep, t2 - t1, train_l2, test_l2)

        if not(ep%log_interval):
            writer.add_scalar('train_err', train_l2, ep)
            writer.add_scalar('test_err', test_l2, ep)
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], ep)
            print(f'Epoch {ep}\t| time={t2-t1:.2f}s\t| train_err:\t{train_l2:.4f}\t| test_err: {test_l2:.4f}')

        if not(ep%save_interval):
            checkpoint_path = save_dir + f"ep_{ep}.pt"
            torch.save(model.state_dict(), checkpoint_path)

    return

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)