
epochs=16
# Fixed Data Configs
data_path="/home/yichen/repo/cfd/myFNO/data/zongyi/NavierStokes_V1e-5_N1200_T20.mat"
data_name="TorusLi"
n_train=1000
n_test=200
raw_in_channels=1
time_suffix=1
config_details=1
log_interval=1

# Fixed Model Configs: 
# The folling use default values: n_modes, hidden_channels, lifting_channels, projection_channels
model_name="FNO"
randon_seed=0
seed=0

# Unfixed Training Configs:
lrs=(0.001)
batch_sizes=(64)
weight_decays=(0)
scheduler_steps=(100)
scheduler_gammas=(0.5)
training_losses=("h1" "l2")

# Unfixed Model Configs
n_layerses=(4)
pos_encodings=(1 0)
factorizations=("" "tucker")
ranks=(0.42)
channel_mixings=("" "prod-layer")
num_prods=(2)


for lr in "${lrs[@]}"
do
  for batch_size in "${batch_sizes[@]}"
  do
    for weight_decay in "${weight_decays[@]}"
    do
      for scheduler_step in "${scheduler_steps[@]}"
      do
        for scheduler_gamma in "${scheduler_gammas[@]}"
        do
          for training_loss in "${training_losses[@]}"
          do
            for n_layers in "${n_layerses[@]}"
            do
              for pos_encoding in "${pos_encodings[@]}"
              do
                for factorization in "${factorizations[@]}"
                do
                  for rank in "${ranks[@]}"
                  do 
                    for channel_mixing in "${channel_mixings[@]}"
                    do
                      for num_prod in "${num_prods[@]}"
                      do
                        python -m scripts.train_fno_markov_2d \
                            --data_path $data_path \
                            --data_name $data_name \
                            --n_train $n_train \
                            --n_test $n_test \
                            --raw_in_channels $raw_in_channels \
                            --time_suffix $time_suffix \
                            --config_details $config_details \
                            --model_name $model_name \
                            --randon_seed $randon_seed \
                            --seed $seed \
                            --lr $lr \
                            --batch_size $batch_size \
                            --weight_decay $weight_decay \
                            --scheduler_step $scheduler_step \
                            --scheduler_gamma $scheduler_gamma \
                            --training_loss $training_loss \
                            --n_layers $n_layers \
                            --pos_encoding $pos_encoding \
                            --factorization $factorization \
                            --rank $rank \
                            --channel_mixing $channel_mixing \
                            --num_prod $num_prod
                            # -- $
                      done
                      if [ $channel_mixing != *"prod"*]
                      then break 
                      fi
                    done
                  done
                  if [ $factorization == ""] 
                  then break 
                  fi
                done
              done
            done
          done
        done
      done
    done
  done
done