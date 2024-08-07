CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_lsm_darcy \
    --data_path ../../data/zongyi/piececonst_r421_N1024_smooth1.mat \
    --data_name DarcyFlow \
    --n_train 960 \
    --n_test 64 \
    --batch_size 64 \
    --train_subsample_rate 5 \
    --test_subsample_rate 5 \
    --train_loss h1 \
    --log_interval 1 \
    --random_seed 0 \
    --seed 1825 \
    # --channel_mixing prod-layer \
    # --num_prod 1

CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_lsm_darcy \
    --data_path ../../data/zongyi/piececonst_r421_N1024_smooth1.mat \
    --data_name DarcyFlow \
    --n_train 960 \
    --n_test 64 \
    --batch_size 64 \
    --train_subsample_rate 5 \
    --test_subsample_rate 5 \
    --train_loss h1 \
    --log_interval 1 \
    --random_seed 0 \
    --seed 1825 \
    --channel_mixing prod-layer \
    --num_prod 1

CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_lsm_darcy \
    --data_path ../../data/zongyi/piececonst_r421_N1024_smooth1.mat \
    --data_name DarcyFlow \
    --n_train 960 \
    --n_test 64 \
    --batch_size 64 \
    --train_subsample_rate 5 \
    --test_subsample_rate 5 \
    --train_loss h1 \
    --log_interval 1 \
    --random_seed 0 \
    --seed 1825 \
    --channel_mixing prod-layer \
    --num_prod 2

CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_lsm_darcy \
    --data_path ../../data/zongyi/piececonst_r421_N1024_smooth1.mat \
    --data_name DarcyFlow \
    --n_train 960 \
    --n_test 64 \
    --batch_size 64 \
    --train_subsample_rate 5 \
    --test_subsample_rate 5 \
    --train_loss h1 \
    --log_interval 1 \
    --random_seed 0 \
    --seed 1825 \
    --channel_mixing prod-layer \
    --num_prod 4

CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_lsm_darcy \
    --data_path ../../data/zongyi/piececonst_r421_N1024_smooth1.mat \
    --data_name DarcyFlow \
    --n_train 960 \
    --n_test 64 \
    --batch_size 64 \
    --train_subsample_rate 5 \
    --test_subsample_rate 5 \
    --train_loss h1 \
    --log_interval 1 \
    --random_seed 0 \
    --seed 1825 \
    --channel_mixing prod-layer \
    --num_prod 8

CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_lsm_darcy \
    --data_path ../../data/zongyi/piececonst_r421_N1024_smooth1.mat \
    --data_name DarcyFlow \
    --n_train 960 \
    --n_test 64 \
    --batch_size 64 \
    --train_subsample_rate 5 \
    --test_subsample_rate 5 \
    --train_loss h1 \
    --log_interval 1 \
    --random_seed 0 \
    --seed 1825 \
    --channel_mixing prod-layer \
    --num_prod 16