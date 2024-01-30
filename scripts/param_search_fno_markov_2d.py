from .train_fno_markov_2d import get_parser, run
import argparse

# python -m scripts.param_search_fno_markov_2d
new_args = argparse.Namespace(
    # Fixed Data Configs
    epochs=1,
    data_path = "/home/yichen/repo/cfd/myFNO/data/zongyi/NavierStokes_V1e-5_N1200_T20.mat",
    data_name = "TorusLi",
    n_train = 1000,
    n_test = 200,
    raw_in_channels = 1,
    time_suffix = True,
    config_details = True,
    # Fixed Model Configs
    model_name = "FNO",
    random_seed = False,
    # Others
    seed = 0,
    log_interval=1,
    save_path='./test_param_search',
    log_path='./test_param_search',
    verbose=False
)

# Unfixed Training Configs
lrs = [0.001]
batch_sizes = [64]
weight_decays = [0]
scheduler_steps = [100]
scheduler_gammas = [0.5]
training_losses = ["h1", "l2"]

# Unfixed Model Configs
n_layerses = [4]
pos_encodings = [True, False]
factorizations = ["", "tucker"]
ranks = [0.42]
channel_mixings = ["", "prod-layer"]
num_prods = [2]

def update_args(default_args, new_args):
    for key, value in vars(new_args).items():
        setattr(default_args, key, value)

parser = get_parser()
args = parser.parse_args(args=[]) # default args
update_args(args, new_args)

# Run Experiments
for lr in lrs:
 args.lr = lr
 for batch_size in batch_sizes:
  args.batch_size = batch_size
  for weight_decay in weight_decays:
   args.weight_decay = weight_decay
   for scheduler_step in scheduler_steps:
    args.scheduler_step = scheduler_step
    for scheduler_gamma in scheduler_gammas:
     args.scheduler_gamma = scheduler_gamma
     for training_loss in training_losses:
      args.training_loss = training_loss
      for n_layers in n_layerses:
       args.n_layers = n_layers
       for pos_encoding in pos_encodings:
        args.pos_encoding = pos_encoding
        for factorization in factorizations:
         args.factorization = factorization
         for rank in ranks:
          args.rank = rank
          for channel_mixing in channel_mixings:
           args.channel_mixing = channel_mixing
           for num_prod in num_prods:
            args.num_prod = num_prod
            run(args)
            if "prod" not in channel_mixing:
             break
          if factorization == "":
           break