for epsilon in 0 0.1
do
  for bias in 0.8
  do
    for iv_lr in 1e-4
    do
      for is_robust in train
      do
        for p in inf 2
        do
          python3 train_model.py \
              --mode CausalRep \
              --epoch_max 150 \
              --dataset pcic \
              --feature_data False \
              --train_mode $is_robust \
              --embedding_dim 128 \
              --class_weight 1 1 \
              --norm $p \
              --downstream MLP \
              --x_dim 2 \
              --user_dim 1 \
              --user_size 1000 \
              --user_item_size 1000 1720 \
              --ipm_layer_dims 64 32 16 \
              --ctr_layer_dims 64 32 16 \
              --epsilon $epsilon \
              --bias $bias \
              --iv_lr $iv_lr
        done
      done
    done
  done
done
