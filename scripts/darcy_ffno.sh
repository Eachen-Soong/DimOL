CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_ffno_darcy \
    --data_path ../data/zongyi/piececonst_r421_N1024_smooth1.mat \
    --data_name DarcyFlow \
    --n_train 960 \
    --n_test 64 \
    --batch_size 64 \
    --train_subsample_rate 5 \
    --test_subsample_rate 5 \
    --time_step 4 \
    --n_modes 21 \
    --num_prod 2 \
    --n_layers 4 \
    --pos_encoding True \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --time_suffix True \
    --config_details True \
    --log_interval 4 \
    --save_interval 20 \
    --epochs 500 \
    --verbose True \
    --random_seed False \
    --seed 0 \
    --ffno_channel_mixing linear \
    --channel_mixing prod-layer \
    # --factorization tucker \
    # --rank 0.42 \
    