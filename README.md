# myNeuralOperator

Experiment Structure:

We have 4 models (FNO/F-FNO/T-FNO/LSM), from which we can chose the first 3 of them as backbone model.

1. Regular Grid
4 datasets: darcy, burgers, torus_li (from the original FNO paper), ns_contextual (from F_FNO)

Note that in this setting, we can check whether a method is well-functioned as a operator enough by using some augmented datasets to test (for the ns_contextual dataset).

```bash
python -m scripts.train_fno_torus_li --...
```

2. Irregular Grid



## Dataset Preparation
We adopt the F-FNO datasets, which has intergrated the FNO datasets too.

```shell
wget --continue https://object-store.rc.nectar.org.au/v1/AUTH_c0e4d64401cf433fb0260d211c3f23f8/fourierflow/data-2021-12-24.tar.gz
tar -zxvf data-2021-12-24.tar.gz
```

One thing to consider: for the model, 

say, the feature to predict is u, and the input feature has other dimensions such as f, mu, and even fixed-weight terms like x_grid and y_grid, then where should we append the dimensions?

Choice 1: for x_grid and y_grid: in the model; for f and mu: elsewhere

This choice is reasonable, but a bit complex in implementation.

Choice 2: all appended in dataloader

This is adopted by F-FNO, but may cause extra waste of memory.

Choice 3: all appended in the trainer.on_batch_start()

We adopt this implementation, but for inference tasks, we would need to use an special inferencer to append the self-generated inputs like x_grid and y_grid.

The only thing that changed is:

f_fno_spectral_conv.py : Reproducing the F-FNO spectral convolution

quad_layer.py: 

fno_block.py: Added class ProdFNO_Blocks(FNOBlocks)

training/callbacks.py : Added SimpleTensorBoardLoggerCallback(callback)

datasets/autoregressive_dataset.py : for time series field datasets, creates autoregressive markov datasets and dataloaders

datasets/dataloader.py : for time series field datasets, creates autoregressive markov datasets and dataloaders

