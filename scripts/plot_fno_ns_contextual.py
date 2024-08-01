"""
    Only supports the time series datasets! (model definition assumes it's 2d problem here)
    (the train numbers and test numbers should not be predefined in the dataset, otherwise merge the dataset)
    the predicted feature now only support 1 channel
    logger currently only supports tensorboard 
    TODO: support more channels
"""
import torch
import numpy as np

from neuralop.models import FNO
# from neuralop.training import OutputEncoderCallback
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
from neuralop.datasets.autoregressive_dataset import load_autoregressive_traintestsplit_v3
from neuralop.training import MultipleInputCallback, SimpleTensorBoardLoggerCallback, ModelCheckpointCallback

import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

import argparse

def get_parser():
    parser = argparse.ArgumentParser('FNO Models', add_help=False)
    parser.add_argument('--model', type=str, default='FNO')
    parser.add_argument('--model_name',  type=str, default='FNO')
    # # # Data Loader Configs # # #
    parser.add_argument('--n_train', type=int, default=2)
    parser.add_argument('--n_test', nargs='+', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32) #
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--train_subsample_rate', type=int, default=4)
    parser.add_argument('--test_subsample_rate', nargs='+',  type=int, default=4)
    parser.add_argument('--time_step', type=int, default=10)
    parser.add_argument('--predict_feature', type=str, default='u')
    parser.add_argument('--data_path', type=str, default='./data/ns_random_forces_1.h5', help="the path of data file")
    parser.add_argument('--test_data_path', nargs='+', type=str, default='', help="the path of test data file")
    parser.add_argument('--data_name', type=str, default='NS_Contextual', help="the name of dataset")
    parser.add_argument('--simaug_train_data', type=int, default=0, help="whether to augment the dataset with similar ones")
    parser.add_argument('--simaug_test_data', type=int, default=0, help="whether to augment the test dataset with similar ones")
    # # # Model Configs # # #
    parser.add_argument('--n_modes', type=int, default=21) #
    parser.add_argument('--num_prod', type=int, default=2) #
    parser.add_argument('--n_layers', type=int, default=4) ##
    parser.add_argument('--raw_in_channels', type=int, default=3, help='TorusLi: 1; ns_contextual: 3')
    parser.add_argument('--pos_encoding', type=int, default=1) ##
    parser.add_argument('--hidden_channels', type=int, default=32) #
    parser.add_argument('--lifting_channels', type=int, default=256) #
    parser.add_argument('--projection_channels', type=int, default=64) #
    parser.add_argument('--factorization', type=str, default='tucker') #####
    parser.add_argument('--channel_mixing', type=str, default='', help='') #####
    parser.add_argument('--rank', type=float, default=0.42, help='the compression rate of tensor') #
    parser.add_argument('--load_path', type=str, default='', help='load checkpoint')

    # # # Optimizer Configs # # #
    parser.add_argument('--lr', type=float, default=1e-3) #Path
    parser.add_argument('--weight_decay', type=float, default=1e-4) #
    parser.add_argument('--scheduler_steps', type=int, default=100) #
    parser.add_argument('--scheduler_gamma', type=float, default=0.5) #
    parser.add_argument('--train_loss', type=str, default='h1', help='h1 or l2') #
    # # # Log and Save Configs # # #
    parser.add_argument('--log_path', type=str, default='./runs')
    parser.add_argument('--save_path', type=str, default='./ckpt')
    parser.add_argument('--prefix', type=str, default='', help='prefix of log and save file')
    parser.add_argument('--time_suffix', type=int, default=1, help='whether to use program start time as suffix')
    parser.add_argument('--config_details', type=int, default=1, help='whether to include config details to the log and save file name')
    parser.add_argument('--log_interval', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=20)
    # # # Trainer Configs # # #
    parser.add_argument('--epochs', type=int, default=501) #
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    return parser

def test(args):
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
    n_train = args.n_train
    n_test = args.n_test
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    train_subsample_rate = args.train_subsample_rate
    test_subsample_rate = args.test_subsample_rate
    time_step = args.time_step
    data_path = args.data_path
    train_loader, test_loaders = load_autoregressive_traintestsplit_v3(
        data_path,
        n_train, n_test,
        batch_size, test_batch_size,
        train_subsample_rate, test_subsample_rate,
        time_step,
        test_data_paths=args.test_data_path,
        predict_feature=args.predict_feature,
        append_positional_encoding=args.pos_encoding
    )
    resolution = train_loader.dataset[0]['x'].shape[0]

    # # # Model Definition # # #
    n_modes=args.n_modes
    num_prod=args.num_prod
    in_channels = args.raw_in_channels
    if args.pos_encoding:
        in_channels += 2
    model = FNO(in_channels=in_channels, n_modes=(n_modes, n_modes), hidden_channels=args.hidden_channels, lifting_channels=args.lifting_channels,
                projection_channels=args.projection_channels, n_layers=args.n_layers, factorization=args.factorization, channel_mixing=args.channel_mixing, rank=args.rank, num_prod=num_prod)
    
    if args.load_path != '':
        model.load_state_dict(torch.load(args.load_path))

    model = model.to(device)

    n_params = count_params(model)
    print(f'\nOur model has {n_params} parameters.')
    sys.stdout.flush()

    # # # Optimizer Definition # # #

    # # # Loss Definition # # #
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    if args.train_loss == 'h1':
        train_loss = h1loss
    elif args.train_loss == 'l2':
        train_loss = l2loss
    else: assert False, "Unsupported training loss!"
    eval_losses={'h1': h1loss, 'l2': l2loss}




    # # # Trainer Definition # # #
    from scripts.ns_contextual_trainer import ns_contextual_trainer


    for loader in test_loaders:
        for item in loader:
            out = model.forward(item)[0].squeeze().detach().cpu().numpy()
            gt = item['y'][0].squeeze().detach().cpu().numpy()
            lims = dict(cmap='RdBu_r', vmin=gt.min(), vmax=gt.max())
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
            ax[0].imshow(out)
            ax[1].imshow(gt)
            ax[2].imshow(out-gt)
            fig.savefig(fname='./img/ns1')
            fig.show()
            break
        break

    return

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    test(args)