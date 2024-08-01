CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_lsm_ns_contextual \
    --data_path ../../data/ns_contextual/ns_random_forces_top100_mu.h5 \
    --data_name NS_Contextual_100 \
    --model_name LSM_prod \
    --n_train 100 \
    --n_test 20 \
    --batch_size 64 \
    --num_prod 0 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --time_step 10 \
    --pos_encoding 1 \
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
    --simaug_train_data 0 \
    --simaug_test_data 1

