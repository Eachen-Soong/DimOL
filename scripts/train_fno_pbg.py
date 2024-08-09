import h5py
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from einops import repeat
import numpy as np

from neuralop.models import FNO
from neuralop import Trainer
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
from neuralop.training import OutputEncoderCallback, SimpleTensorBoardLoggerCallback, ModelCheckpointCallback

import os
import sys
import time
from pathlib import Path

# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {DEVICE} device')

import argparse

def get_parser():
    parser = argparse.ArgumentParser('FNO Models', add_help=False)
    parser.add_argument('--model', type=str, default='FNO')
    parser.add_argument('--model_name',  type=str, default='FNO')
    # # # Data Loader Configs # # #
    parser.add_argument('--n_train', type=int, default=2)
    parser.add_argument('--n_test', nargs='+', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4) #
    parser.add_argument('--test_batch_size', type=int, default=32)
    # parser.add_argument('--train_subsample_rate', type=int, default=4)
    # parser.add_argument('--test_subsample_rate', nargs='+',  type=int, default=4)
    parser.add_argument('--time_step', type=int, default=1)
    # parser.add_argument('--predict_feature', type=str, default='u')
    parser.add_argument('--data_path', type=str, default='../../data', help="the path of data file")
    # parser.add_argument('--test_data_path', nargs='+', type=str, default='', help="the path of test data file")
    parser.add_argument('--data_name', type=str, default='PB_Gravity', help="the name of dataset")
    # # # Model Configs # # #
    parser.add_argument('--n_modes', type=int, default=21) #
    parser.add_argument('--num_prod', type=int, default=2) #
    parser.add_argument('--n_layers', type=int, default=4) ##
    parser.add_argument('--raw_in_channels', type=int, default=8, help='')
    parser.add_argument('--pos_encoding', type=int, default=0) ##
    parser.add_argument('--hidden_channels', type=int, default=32) #
    parser.add_argument('--lifting_channels', type=int, default=256) #
    parser.add_argument('--projection_channels', type=int, default=64) #
    parser.add_argument('--factorization', type=str, default='') #####
    parser.add_argument('--channel_mixing', type=str, default='', help='') #####
    parser.add_argument('--mixing_layers', type=int, default=2, help='') #####
    parser.add_argument('--stabilizer', type=str, default='', help='') #####
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
    parser.add_argument('--config_details', type=int, default=2, help='whether to include config details to the log and save file name')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=20)
    # # # Trainer Configs # # #
    parser.add_argument('--epochs', type=int, default=501) #
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=1, help='whether to use random seed') 
    parser.add_argument('--seed', type=int, default=0)

    return parser

TEMPERATURE = 'temperature'
VELX = 'velx'
VELY = 'vely'
PRESSURE = 'pressure'
DFUN = 'dfun'
X = 'x'
Y = 'y'
CONSTS = ['ins_gravy']


class HDF5Dataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.data = h5py.File(self.filename, 'r')
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
    n_train = args.n_train
    n_test = args.n_test
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    # train_subsample_rate = args.train_subsample_rate
    # test_subsample_rate = args.test_subsample_rate
    time_step = args.time_step
    # data_path = args.data_path

    # # # Data Preparation # # #
    range_gravy = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    all_files = [args.data_path + f'/BubbleML/PoolBoiling-Gravity-FC72-2D/gravY-{gravy}.hdf5' for gravy in range_gravy]

    train_files = [all_files[i] for i in [1,3,4,6]]
    val_files = [all_files[7]]

    train_dataset = ConcatDataset(HDF5Dataset(file) for file in train_files)
    val_dataset = ConcatDataset(HDF5Dataset(file) for file in val_files)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')

    # # # Model Definition # # #
    n_modes=args.n_modes
    num_prod=args.num_prod
    in_channels = args.raw_in_channels
    # if args.pos_encoding:
    #     in_channels += 2
    model = FNO(in_channels=in_channels, n_modes=(n_modes, n_modes), hidden_channels=args.hidden_channels, lifting_channels=args.lifting_channels,
                projection_channels=args.projection_channels, n_layers=args.n_layers, factorization=args.factorization, channel_mixing=args.channel_mixing, mixing_layers=args.mixing_layers,
                stabilizer=args.stabilizer, rank=args.rank, num_prod=num_prod)
    
    if args.load_path != '':
        model.load_state_dict(torch.load(args.load_path))
    model.to(DEVICE)
    n_params = count_params(model)
    print(f'\nOur model has {n_params} parameters.')
    sys.stdout.flush()

    # # # Optimizer Definition # # #
    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_steps, gamma=args.scheduler_gamma)

    # # # Loss Definition # # #
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    if args.train_loss == 'h1':
        train_loss = h1loss
    elif args.train_loss == 'l2':
        train_loss = l2loss
    else: assert False, "Unsupported training loss!"
    eval_losses={'h1': h1loss, 'l2': l2loss}

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
        config_file_path = f"/layer_{args.n_layers}/fact-{args.factorization}/rank_{args.rank}/mix-{args.channel_mixing}/mixing_layers-{args.mixing_layers}/prod_{args.num_prod}/pos-enc-{args.pos_encoding}/loss-{args.train_loss}/mode_{args.n_modes}/hid_{args.hidden_channels}/lift_{args.lifting_channels}/proj_{args.projection_channels}/b_{args.batch_size}/lr_{args.lr}/wd_{args.weight_decay}/sche-step_{args.scheduler_steps}/gamma_{args.scheduler_gamma}/"
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
    
    log_dir = Path(log_dir)
    save_dir = Path(save_dir)

    # # # Trainer Definition # # #
    trainer = Trainer(model=model, n_epochs=args.epochs,
                    device=DEVICE,
                    callbacks=[ 
                                SimpleTensorBoardLoggerCallback(log_dir=log_dir),
                                ModelCheckpointCallback(
                                checkpoint_dir=save_dir,
                                interval=args.save_interval)], 
                    wandb_log=False,
                    log_test_interval=args.log_interval,
                    use_distributed=False,
                    verbose=verbose)

    trainer.train(train_loader=train_loader,
                test_loaders={'val':val_loader},
                optimizer=optimizer, 
                scheduler=scheduler, 
                regularizer=False, 
                training_loss=train_loss, 
                eval_losses=eval_losses)

    return

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)