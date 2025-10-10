export CUDA_VISIBLE_DEVICES=4

python -m scripts.train_dim PDEBench FNO \
    --data_folder /data/ycsong/pdebench/1D/Burgers/Train \
    --task burgers \
    --n_train 900 \
    --n_test 100 \
    --raw_in_channels 1 \
    --raw_in_consts 1 \
    --out_channels 1 \
    --n_dim 1 \
    --batch_size 128 \
    --train_subsample_rate 2 \
    --test_subsample_rate 2 \
    --time_step 1 \
    --time_skips 10 \
    --n_modes 21 \
    --channel_mixing mlp \
    --mixing_layers 2 \
    --n_layers 4 \
    --pos_encoding 0 \
    --append_const 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --factorization tucker \
    --rank 0.42 \
    --norm dim_norm1 \
    --pre_norm 1 \
    --preactivation 1 \
    --prediction_dims 0 \
    --num_consts 1 \
    --append_dimless 0 \
    --pos_aug_consts 0 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --epochs 1001 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0


