CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_fno_pbg1 \
    --data_name PB_Gravity_ID \
    --data_path ../../data \
    --n_layers 4 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss l2 \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 1 \
    --save_interval 20 \
    --epochs 201 \
    --verbose 1 \
    --random_seed 0 \
    --seed 1825 \
    --factorization tucker \
    --rank 0.42 \
    --channel_mixing mlp \
    --num_prod 0 \

CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_fno_pbg1 \
    --data_name PB_Gravity_ID \
    --data_path ../../data \
    --n_layers 4 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss l2 \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 1 \
    --save_interval 20 \
    --epochs 201 \
    --verbose 1 \
    --random_seed 0 \
    --seed 1825 \
    --factorization tucker \
    --rank 0.42 \
    --channel_mixing prod-layer \
    --num_prod 2 \

CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_fno_pbg1 \
    --data_name PB_Gravity_ID \
    --data_path ../../data \
    --n_layers 4 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss l2 \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 1 \
    --save_interval 20 \
    --epochs 201 \
    --verbose 1 \
    --random_seed 0 \
    --seed 1825 \
    --factorization tucker \
    --rank 0.42 \
    # --channel_mixing mlp \
    # --num_prod 0 \