#!/bin/bash
set -euo pipefail

tag='SCBM_speed_benchmark'
seed="${SEED:-42}"
train_only="${TRAIN_ONLY:-true}"
timing_file="${TIMING_FILE:-speed_benchmark_times.tsv}"

printf 'name\tstatus\tstart_epoch\tend_epoch\telapsed_sec\telapsed_hms\n' > "$timing_file"

run_timed() {
  local name="$1"
  shift

  local start
  local end
  local elapsed
  local status

  start=$(date +%s)
  set +e
  "$@"
  status=$?
  set -e
  end=$(date +%s)
  elapsed=$((end - start))
  printf '%s\t%d\t%d\t%d\t%d\t%02d:%02d:%02d\n' \
    "$name" \
    "$status" \
    "$start" \
    "$end" \
    "$elapsed" \
    $((elapsed / 3600)) \
    $(((elapsed % 3600) / 60)) \
    $((elapsed % 60)) >> "$timing_file"

  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
}

run_baseline() {
  local data="$1"
  local encoder_arch="$2"
  local learning_rate="$3"
  local weight_decay="$4"
  local train_batch_size="$5"
  local val_batch_size="$6"
  local workers="$7"
  local model="$8"

  run_timed "${data}-${model}" \
  python -u train.py +model="$model" +data="$data" \
    experiment_name="speed_${data}_${model}_${seed}" seed="$seed" \
    logging.project=SCBM logging.mode=offline \
    model.tag="$tag" model.encoder_arch="$encoder_arch" \
    model.learning_rate="$learning_rate" model.weight_decay="$weight_decay" \
    model.train_batch_size="$train_batch_size" model.val_batch_size="$val_batch_size" \
    model.j_epochs=5 model.c_epochs=5 model.t_epochs=5 \
    workers="$workers" model.compile=false train_only="$train_only"
}

run_ar() {
  local data="$1"
  local encoder_arch="$2"
  local learning_rate="$3"
  local weight_decay="$4"
  local train_batch_size="$5"
  local val_batch_size="$6"
  local workers="$7"

  run_timed "${data}-AR" \
  python -u train.py +model=AR +data="$data" \
    experiment_name="speed_${data}_AR_${seed}" seed="$seed" \
    logging.project=SCBM logging.mode=offline \
    model.tag="$tag" model.encoder_arch="$encoder_arch" \
    model.learning_rate="$learning_rate" model.weight_decay="$weight_decay" \
    model.train_batch_size="$train_batch_size" model.val_batch_size="$val_batch_size" \
    model.p_epochs=5 model.c_epochs=5 model.t_epochs=5 \
    workers="$workers" model.compile=false train_only="$train_only"
}

run_scbm() {
  local data="$1"
  local encoder_arch="$2"
  local learning_rate="$3"
  local weight_decay="$4"
  local train_batch_size="$5"
  local val_batch_size="$6"
  local workers="$7"
  local cov_type="$8"
  local reg_precision="$9"
  local suffix="${cov_type}"

  run_timed "${data}-SCBM-${suffix}" \
  python -u train.py +model=SCBM +data="$data" \
    model.cov_type="$cov_type" model.reg_precision="$reg_precision" \
    model.reg_weight=1 experiment_name="speed_${data}_SCBM_${suffix}_${seed}" \
    seed="$seed" logging.project=SCBM logging.mode=offline \
    model.tag="$tag" model.encoder_arch="$encoder_arch" \
    model.learning_rate="$learning_rate" model.weight_decay="$weight_decay" \
    model.train_batch_size="$train_batch_size" model.val_batch_size="$val_batch_size" \
    model.j_epochs=5 model.c_epochs=5 model.t_epochs=5 \
    workers="$workers" model.compile=false train_only="$train_only"
}

run_dataset() {
  local data="$1"
  local encoder_arch="$2"
  local learning_rate="$3"
  local weight_decay="$4"
  local train_batch_size="$5"
  local val_batch_size="$6"
  local workers="$7"

  echo "Running speed benchmark for ${data}"
  run_ar "$data" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers"
  run_baseline "$data" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers" CEM
  run_baseline "$data" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers" CBM
  run_scbm "$data" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers" amortized l1
  run_scbm "$data" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers" global None
}

run_dataset synthetic FCNN 0.0002 0.0001 256 512 2
run_dataset CUB resnet18 0.0001 0.0001 64 256 12
run_dataset cifar10 simple_CNN 0.0002 0.0001 256 512 12
