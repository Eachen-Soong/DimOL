export CUDA_VISIBLE_DEVICES=3

python -m scripts.evaluate TorusVisForceDim FNO \
    --data_path /data/ycsong/data/ns_contextual/ns_random_forces_v0.h5 \
    --n_train 0 \
    --n_test 200 \
    --batch_size 16 \
    --train_subsample_rate 4 \
    --test_subsample_rate 4 \
    --predict_feature u \
    --time_step 1 \
    --time_skips 10 \
    --simaug_coeff 2 4 8 \
    --load_path ./runs/TorusVisForceDim/FNO/exp_7-29-1-14/version_0 \
