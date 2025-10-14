export CUDA_VISIBLE_DEVICES=2

python -m scripts.train_init_steps_dim PDEBench CROP \
    --data_folder /data/ycsong/pdebench/2D/diffusion-reaction/ \
    --task diff-react-2d \
    --n_train 90 \
    --n_test 10 \
    --initial_steps 10 \
    --t_train 101 \
    --raw_in_channels 2 \
    --raw_in_consts 2 \
    --out_channels 2 \
    --n_dim 2 \
    --batch_size 4 \
    --train_subsample_rate 2 \
    --test_subsample_rate 2 \
    --time_step 1 \
    --time_skips 1 \
    --modes 21 \
    --ini_channels 32 \
    --N_layers 3 \
    --N_res 4 \
    --N_res_neck 6 \
    --in_out_size 64 \
    --latent_size 64 \
    --kernel_size 3 \
    --append_const 1 \
    --use_dim 1 \
    --norm layer_norm \
    --pre_norm 0 \
    --align_final 1 \
    --num_consts 4 \
    --prediction_dims 0 1 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --epochs 1001 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0


