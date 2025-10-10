export CUDA_VISIBLE_DEVICES=0

python -m scripts.train_dim PDEBench FNO \
    --data_folder /data/ycsong/pdebench/2D/diffusion-reaction/ \
    --task diff-react-2d \
    --n_train 900 \
    --n_test 100 \
    --raw_in_channels 2 \
    --raw_in_consts 2 \
    --out_channels 2 \
    --n_dim 2 \
    --batch_size 32 \
    --train_subsample_rate 2 \
    --test_subsample_rate 2 \
    --time_step 1 \
    --time_skips 1 \
    --n_modes 21 \
    --channel_mixing mlp \
    --mixing_layers 2 \
    --n_layers 4 \
    --pos_encoding 1 \
    --append_const 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --factorization tucker \
    --rank 0.42 \
    --norm dim_norm1 \
    --pre_norm 0 \
    --preactivation 1 \
    --prediction_dims 0 1 \
    --num_consts 4 \
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




