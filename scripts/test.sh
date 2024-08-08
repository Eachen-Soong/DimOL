#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_fno_darcy \
    --data_path ../../data/zongyi/piececonst_r421_N1024_smooth1.mat \
    --data_name test \
    --n_train 960 \
    --n_test 64 \
    --batch_size 64 \
    --train_subsample_rate 5 \
    --test_subsample_rate 5 \
    --time_step 4 \
    --n_modes 21 \
    --n_layers 4 \
    --pos_encoding 1 \
    --hidden_channels 32 \
    --lifting_channels 256 \
    --projection_channels 64 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --scheduler_steps 100 \
    --scheduler_gamma 0.5 \
    --train_loss h1 \
    --time_suffix 1 \
    --config_details 1 \
    --log_interval 1 \
    --save_interval 20 \
    --epochs 1 \
    --verbose 1 \
    --random_seed 0 \
    --seed 0 \
    --factorization tucker \
    --rank 0.42 \


# fixed_string="s"

# while true; do
#     read -p "请输入字符串: " input_string

#     if [[ $input_string != *"$fixed_string"* ]]; then
#         echo "输入的字符串不包含固定字符串: $fixed_string"
#         break
#     else
#         echo "输入的字符串包含固定字符串: $fixed_string, 请再次尝试。" # fuck
#     fi
# done


# import types

# class fucker:
#     def __init__(self, cum_amount, fuck_cunt_cnt):
#         self.cum_amount = cum_amount
#         self.fuck_cunt_cnt = fuck_cunt_cnt

#     def fuck(self, girl:str):
#         self.fuck_cunt_cnt +=1

# Qiyu = fucker(114, 514)

# def new_fuck(self, girl:str):
#     self.fuck_cunt_cnt +=1
#     print(f"{girl}, I'm fucking cumming! you're the {self.fuck_cunt_cnt}-th girl I've fucked.\n\
#           You'll recieve {self.cum_amount} mL of my cum.")

# Qiyu.fuck = types.MethodType(new_fuck, Qiyu)

# Qiyu.fuck("Girl")
# # genders=("male" "female")
# ages=(10 20 30 40 50 60 70 80 90 100)

# for gender in ${genders[@]}
# do
#     for age in ${ages[@]}
#     do
#         echo $gender $age
#         if [ $gender == "male" ]
#         then break
#         fi
#     done
# done


# int=1
# flag=0  # true
# while [ $flag -eq 0 ]
# do
#     echo $int
#     if [ $int -le 5 ] 
#     then flag=0  # true
#     else flag=1  # false
#     fi
#     let "int++"
# done
