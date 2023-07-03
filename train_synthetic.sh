for epsilon in 0 0.3
do
  for bias in 0.8
  do
    for iv_lr in 1e-4
    do
      for is_robust in train
      do
        for p in inf 2
        do
          for data in  nonliearlogitdata_500_0.1 nonliearlogitdata_500_0.3 nonliearlogitdata_500_0.5 nonliearlogitdata_500_0.7 nonliearlogitdata_500_1  nonliearlogitdata_500_3
          do
              python3 train_syn.py \
                  --mode CausalRep \
                  --epoch_max 150 \
                  --dataset $data \
                  --feature_data True \
                  --train_mode $is_robust \
                  --embedding_dim 128 \
                  --class_weight 1 1 \
                  --norm $p \
                  --downstream MLP \
                  --x_dim 15 \
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
done
