export CUDA_VISIBLE_DEVICES=6

python -m scripts.evaluate TorusVisForceDim FNO \
    --data_path /data/ycsong/data/ns_contextual/ns_random_forces_v0.h5 \
    --n_train 0 \
    --n_test 20 \
    --batch_size 128 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --predict_feature u \
    --time_step 1 \
    --time_skips 1 \
    --simaug_coeff 2 4 8 16 \
    --load_path ./runs/TorusVisForceDim/FNO/exp_8-2-2-39/version_0 \
