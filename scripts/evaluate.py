import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9505))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import time
import yaml
from types import SimpleNamespace

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary

import lightning as L
from lightning.pytorch import seed_everything
from lightning_modules import MultiMetricModule
from utils.losses import LpLoss, H1Loss

from scripts.get_parser import Fetcher
from scripts.models import FNOParser, LSMParser, CNOParser, FNO_OriginalParser
from scripts.datasets import TorusVisForceDimParser#, TorusVisForceDimTimeParser, PDEBenchDimParser

ModelParsers = [FNOParser]
DataParsers = [TorusVisForceDimParser#, TorusVisForceDimTimeParser, PDEBenchDimParser
               ]

from lightning.pytorch.callbacks import Callback

class SetDimNormCoeff(Callback):
    def on_train_start(self, trainer, pl_module):
        num_consts = trainer.model.model.num_consts
        total_mu, total_std = torch.zeros(num_consts), torch.zeros(num_consts)
        cnt=0
        for batch in trainer.train_dataloader:
            # for  in dataloader:
                aligned_mu, aligned_std = trainer.model.model.dim_aligner(**batch)
                total_mu += torch.mean(aligned_mu, dim=0); total_std += torch.mean(aligned_std, dim=0); cnt+=1
        mean_mu = total_mu / cnt; mean_std = total_std / cnt
        keep_origin = True
        if keep_origin:
            mean_mu = torch.cat([torch.tensor([1.], dtype=mean_mu.dtype), mean_mu])
            mean_std = torch.cat([torch.tensor([1.], dtype=mean_std.dtype), mean_std])

        trainer.model.model.set_dim_coeffs(1/mean_std, torch.zeros_like(mean_std))
        print(mean_mu, mean_std)
        return super().on_train_start(trainer, pl_module)

def find_files_by_suffix(directory, suffix):
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                result.append(os.path.join(root, file))
    return result

def run(raw_args=None):
    fetcher = Fetcher(DataParsers=DataParsers, ModelParsers=ModelParsers, mode='test')

    args = fetcher.parse_args(raw_args)
    verbose = args.verbose
    # # # Seed # # #
    # if args.random_seed:
    #     seed_everything()
    # else:
    #     seed_everything(args.seed)
    
    # # # Data Preparation # # #
    train_loader, val_loader = fetcher.get_data(args)
    
    # # # Create Lightning Module # # #
    # 1. Model Definition
    model = fetcher.get_model(args)

    hparams_path = os.path.join(args.load_path, 'hparams.yaml')
    if os.path.exists(hparams_path):
        with open(hparams_path) as stream:
            try:
                hparams = SimpleNamespace(**yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
    else:
        hparams = args
    # print(hparams)
    model = fetcher.get_model(hparams)
    use_dim = (hparams.norm == 'dim_norm' or hparams.norm == 'dim_norm1' or hparams.append_dimless)

    if use_dim:
        model.set_dim_aligner(fetcher.data_fetcher[args.data]().get_dim_aligner(hparams))
    
    # if args.scale_shifting:
    #     model.set_scale_shifting(fetcher.data_fetcher[args.data]().get_scale_shifting(args))

        print(hparams)

    ckpt_list = find_files_by_suffix(args.load_path, '.ckpt')
    assert len(ckpt_list) == 1, f"Number of files ended with .ckpt should be 1, instead of {len(ckpt_list)}."
    
    ckpt = torch.load(ckpt_list[0])
    prefix = "model."
    state_dict = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}

    model.load_state_dict(state_dict)


    del fetcher
    
    # 3. Loss Definition
    loss_dict = {'h1': H1Loss(d=2, reductions=args.loss_reduction), 'l2': LpLoss(d=2, p=2, reductions=args.loss_reduction) , 'l1': LpLoss(d=2, p=1, reductions=args.loss_reduction)}

    # try: train_loss = loss_dict[args.train_loss]
    # except: print(f"Unsupported training loss! {args.train_loss}")
    try:
        eval_loss_names = args.eval_loss
        if type(eval_loss_names) == type(''):
            eval_loss_names = [eval_loss_names]
        eval_losses = {key: loss_dict[key] for key in eval_loss_names}
    except: print(f"Unsupported eval loss! {args.eval_loss}")

    if verbose:
        print('\n### MODEL ###\n', model)
        print('\n### LOSSES ###')
        print(f'\n * Evaluation: {eval_losses}')
        sys.stdout.flush()

    use_sum_reduction = (args.loss_reduction == 'sum')
    module = MultiMetricModule(model=model, optimizer=None, scheduler=None, train_loss=None, metric_dict=loss_dict, average_over_batch=use_sum_reduction, apply_rollout=args.apply_rollout)

    # # # Logs # # #
    save_dir = args.save_dir + '/' + args.data + '/' + args.model + '/evaluation/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    localtime = time.localtime(time.time())
    time_name = f"{localtime.tm_mon}-{localtime.tm_mday}-{localtime.tm_hour}-{localtime.tm_min}"
    name = time_name


    logger = TensorBoardLogger(save_dir=save_dir, name=name)
    # logger.log_hyperparams(args)
    # log_path = logger.log_dir

    # with open(log_path + '/model_summary.txt', 'w+') as file:
    #     file.write(ModelSummary(module).__str__())

    # # # Training # # #
    callbacks=[
        ]
    if use_dim:
        callbacks.insert(0, SetDimNormCoeff())

    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        )
    trainer.test(model=module, dataloaders=val_loader)

if __name__ == '__main__':
    run()