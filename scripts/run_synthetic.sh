#!/bin/bash

tag='SCBM_experiments'
data='synthetic'
encoder_arch='FCNN'

for i in 42 73 666 777 1009 1279 1597 1811 1949 2053
do
  for model in 'AR' 'CEM' 'CBM' 
  do
    python -u train.py +model=$model +data=$data \
      experiment_name="${data}_${model}_${i}" seed=$i \
      logging.project=SCBM logging.mode=offline \
      model.tag=$tag model.encoder_arch=$encoder_arch \
      model.j_epochs=150 model.c_epochs=100 model.t_epochs=50 \
      model.learning_rate=0.0002 model.weight_decay=0.0001 \
      model.train_batch_size=256 model.val_batch_size=512 \
      workers=2
  done

  python -u train.py +model=SCBM +data=$data model.cov_type='amortized' \
    model.reg_precision='l1' model.reg_weight=1 experiment_name="${data}_SCBM_amortized_${i}" \
    seed=$i logging.project=SCBM logging.mode=offline \
    model.tag=$tag model.encoder_arch=$encoder_arch \
    model.j_epochs=150 model.c_epochs=100 model.t_epochs=50\
    model.learning_rate=0.0002 model.weight_decay=0.0001 \
    model.train_batch_size=256 model.val_batch_size=512 \
    workers=2
  python -u train.py +model=SCBM +data=$data model.cov_type='global' \
    model.reg_precision=None experiment_name="${data}_SCBM_global_${i}" \
    seed=$i logging.project=SCBM logging.mode=offline \
    model.tag=$tag model.encoder_arch=$encoder_arch \
    model.j_epochs=150 model.c_epochs=100 model.t_epochs=50 \
    model.learning_rate=0.0002 model.weight_decay=0.0001 \
    model.train_batch_size=256 model.val_batch_size=512 \
    workers=2
done