#!/bin/bash

tag='SCBM_experiments'
for i in 42 73 666 777 1009 1279 1597 1811 1949 2053
do
  ### CUB data
  data='CUB'
  encoder_arch='resnet18'

  # # Baselines
  for model in 'AR' 'CEM' 'CBM'
  do
    python -u train.py +model=$model +data=$data \
      experiment_name="${data}_${model}_${i}" seed=$i \
      logging.project=SCBM logging.mode=offline \
      model.tag=$tag model.encoder_arch=$encoder_arch \
      model.learning_rate=0.0001 model.weight_decay=0.0001 \
      model.train_batch_size=64 model.val_batch_size=256 \
      workers=12
  done;
  
  # SCBM
  # python -u train.py +model=SCBM +data=$data \
  #   model.cov_type='amortized' model.reg_precision='l1' \
  #   model.reg_weight=1 experiment_name="${data}_SCBM_amortized_${i}" \
  #   seed=$i logging.project=SCBM logging.mode=offline \
  #   model.tag=$tag model.encoder_arch=$encoder_arch \
  #   model.learning_rate=0.0001 model.weight_decay=0.0001 \
  #   model.train_batch_size=64 model.val_batch_size=256 \
  #   workers=12 model.compile='true'
  # python -u train.py +model=SCBM +data=$data \
  #   model.cov_type='global' model.reg_precision=None \
  #   experiment_name="${data}_SCBM_global_${i}" seed=$i \
  #   logging.project=SCBM logging.mode=offline \
  #   model.tag=$tag model.encoder_arch=$encoder_arch \
  #   model.learning_rate=0.0001 model.weight_decay=0.0001 \
  #   model.train_batch_size=64 model.val_batch_size=256 \
  #   workers=12 model.compile='true'

done;