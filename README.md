# myNeuralOperator

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

