python -m scripts.train_fno_burgers \
    --data_path ../../data/zongyi/burgers_data_R10.mat \
    --data_name Burgers \
    --n_train 1536 \
    --n_test 512 \
    --batch_size 32 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --n_modes 21 \
    --num_prod 1 \
    --n_layers 4 \
    --pos_encoding 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --channel_mixing prod-layer \
    --rank 0.42 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --log_path ./runs \
    --save_path ./ckpt \
    --prefix test \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 1 \
    --save_interval 20 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 0 \
    --seed 1825

python -m scripts.train_fno_burgers \
    --data_path ../../data/zongyi/burgers_data_R10.mat \
    --data_name Burgers \
    --n_train 1536 \
    --n_test 512 \
    --batch_size 32 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --n_modes 21 \
    --num_prod 4 \
    --n_layers 4 \
    --pos_encoding 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --channel_mixing prod-layer \
    --rank 0.42 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --log_path ./runs \
    --save_path ./ckpt \
    --prefix test \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 1 \
    --save_interval 20 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 0 \
    --seed 1825

python -m scripts.train_fno_burgers \
    --data_path ../../data/zongyi/burgers_data_R10.mat \
    --data_name Burgers \
    --n_train 1536 \
    --n_test 512 \
    --batch_size 32 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --n_modes 21 \
    --num_prod 8 \
    --n_layers 4 \
    --pos_encoding 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --channel_mixing prod-layer \
    --rank 0.42 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --log_path ./runs \
    --save_path ./ckpt \
    --prefix test \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 1 \
    --save_interval 20 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 0 \
    --seed 1825

python -m scripts.train_fno_burgers \
    --data_path ../../data/zongyi/burgers_data_R10.mat \
    --data_name Burgers \
    --n_train 1536 \
    --n_test 512 \
    --batch_size 32 \
    --train_subsample_rate 1 \
    --test_subsample_rate 1 \
    --n_modes 21 \
    --num_prod 16 \
    --n_layers 4 \
    --pos_encoding 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --channel_mixing prod-layer \
    --rank 0.42 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --log_path ./runs \
    --save_path ./ckpt \
    --prefix test \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 1 \
    --save_interval 20 \
    --epochs 501 \
    --verbose 1 \
    --random_seed 0 \
    --seed 1825