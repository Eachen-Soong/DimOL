export CUDA_VISIBLE_DEVICES=5

python -m scripts.train_dim TorusVisForceDim LSM \
    --data_path /data/ycsong/data/ns_contextual/ns_random_forces_v0.h5 \
    --n_train 1000 \
    --n_test 200 \
    --raw_in_channels 2 \
    --raw_in_consts 1 \
    --out_channels 1 \
    --batch_size 32 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --predict_feature u \
    --time_step 1 \
    --time_skips 10 \
    --d-model 64 \
    --num-basis 12 \
    --num-token 4 \
    --patch-size '5,5' \
    --padding '16,16' \
    --norm layer_norm \
    --append_const 1 \
    --use_dim 1 \
    --pre_norm 0 \
    --align_final 1 \
    --prediction_dims 0 \
    --num_consts 3 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 80 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0

