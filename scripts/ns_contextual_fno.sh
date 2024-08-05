CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_fno_ns_contextual \
    --data_path ../../data/ns_contextual/ns_random_forces_v0.h5 \
    --data_name NS_Contextual_100 \
    --n_train 100 \
    --n_test 20 \
    --batch_size 64 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --time_step 4 \
    --n_modes 21 \
    --channel_mixing mlp \
    --mixing_layers 3 \
    --num_prod 0 \
    --n_layers 4 \
    --pos_encoding 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --factorization tucker \
    --rank 0.42 \
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
    --seed 0

# CUDA_VISIBLE_DEVICES=0 \
# python -m scripts.train_fno_ns_contextual \
#     --data_path ../../data/ns_contextual/ns_random_forces_v0.h5 \
#     --data_name NS_Contextual \
#     --n_train 1000 \
#     --n_test 200 \
#     --batch_size 64 \
#     --train_subsample_rate 4 \
#     --test_subsample_rate 4 \
#     --time_step 10 \
#     --n_modes 21 \
#     --num_prod 2 \
#     --n_layers 4 \
#     --pos_encoding 1 \
#     --hidden_channels 32 \
#     --lifting_channels 256 \
#     --projection_channels 64 \
#     --factorization tucker \
#     --rank 0.42 \
#     --channel_mixing mlp \
#     --lr 1e-3 \
#     --weight_decay 1e-4 \
#     --scheduler_steps 100 \
#     --scheduler_gamma 0.5 \
#     --train_loss h1 \
#     --time_suffix 1 \
#     --config_details 1 \
#     --log_interval 1 \
#     --save_interval 20 \
#     --epochs 501 \
#     --verbose 1 \
#     --random_seed 0 \
#     --seed 4507


