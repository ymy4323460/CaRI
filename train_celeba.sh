for epsilon in 0 0.1 0.3 0.7
do
  for bias in 0.8
  do
    for iv_lr in 1e-4
    do
      for is_robust in train
      do
        for p in inf 2
        do
          for data in  celeba
          do
   
              python3 train_celeb.py \
                  --mode CausalRep \
                  --epoch_max 150 \
                  --dataset $data \
                  --feature_data True \
                  --train_mode $is_robust \
                  --embedding_dim 8 \
                  --class_weight 1 1 \
                  --norm $p \
                  --downstream MLP \
                  --x_dim 7 \
                  --ipm_layer_dims 32 8 16 \
                  --ctr_layer_dims 32 8 16 \
                  --epsilon $epsilon \
                  --bias $bias \
                  --iv_lr $iv_lr
              

          done
        done
      done
    done
  done
done

