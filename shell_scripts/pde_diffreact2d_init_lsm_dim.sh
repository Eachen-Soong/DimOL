export CUDA_VISIBLE_DEVICES=4

python -m scripts.train_init_steps_dim PDEBench LSM \
    --data_folder /data/ycsong/pdebench/2D/diffusion-reaction/ \
    --task diff-react-2d \
    --n_train 90 \
    --n_test 10 \
    --initial_steps 10 \
    --t_train 101 \
    --raw_in_channels 2 \
    --raw_in_consts 2 \
    --out_channels 2 \
    --batch_size 2 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --time_step 1 \
    --time_skips 1 \
    --d-model 64 \
    --num-basis 12 \
    --num-token 4 \
    --patch-size '3,3' \
    --padding '16,16' \
    --norm layer_norm \
    --append_const 1 \
    --use_dim 1 \
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


