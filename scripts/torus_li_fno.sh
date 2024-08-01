CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_fno_torus_li \
    --data_path ./data/NavierStokes_V1e-5_N1200_T20.mat \
    --data_name Toris_Li \
    --n_train 1000 \
    --n_test 200 \
    --batch_size 32 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --time_step 10 \
    --n_modes 21 \
    --num_prod 2 \
    --n_layers 4 \
    --pos_encoding 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 4 \
    --save_interval 20 \
    --epochs 500 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0 \
    --channel_mixing prod-layer \
    # --factorization tucker \
    # --rank 0.42 \
    


