#!/bin/bash
set -euo pipefail

seed="${SEED:-42}"
timing_file="${TIMING_FILE:-intervention_benchmark_times.tsv}"
log_dir="${LOG_DIR:-verification/intervention_benchmark}"

mkdir -p "$log_dir"
printf 'dataset\tmodel\torder\tstatus\tintervention_sec\tlog_file\n' > "$timing_file"

run_intervention() {
  local data="$1"
  local model="$2"
  local order="$3"
  local encoder_arch="$4"
  local learning_rate="$5"
  local weight_decay="$6"
  local train_batch_size="$7"
  local val_batch_size="$8"
  local workers="$9"
  shift 9

  local name="${data}-${model}-${order}"
  local log_file="${log_dir}/${name}.log"
  local status
  local intervention_sec

  echo "Running intervention benchmark for ${name}"
  set +e
  python -u train.py +model="$model" +data="$data" \
    experiment_name="intervention_benchmark_${data}_${model}_${seed}" seed="$seed" \
    logging.project=SCBM logging.mode=disabled \
    model.encoder_arch="$encoder_arch" \
    model.learning_rate="$learning_rate" model.weight_decay="$weight_decay" \
    model.train_batch_size="$train_batch_size" model.val_batch_size="$val_batch_size" \
    model.inter_policy=prob_unc \
    workers="$workers" model.compile=false train_only=false intervene_only=true \
    intervention_order="$order" \
    "$@" > "$log_file" 2>&1
  status=$?
  set -e

  intervention_sec="$(awk -F'intervention_sec=' '/^\[timing\] intervention_sec=/{ value=$2 } END { print value }' "$log_file")"
  printf '%s\t%s\t%s\t%d\t%s\t%s\n' \
    "$data" "$model" "$order" "$status" "${intervention_sec:-NA}" "$log_file" >> "$timing_file"

  if [ "$status" -ne 0 ]; then
    echo "Benchmark failed for ${name}; see ${log_file}" >&2
    exit "$status"
  fi
}

run_dataset() {
  local data="$1"
  local encoder_arch="$2"
  local learning_rate="$3"
  local weight_decay="$4"
  local train_batch_size="$5"
  local val_batch_size="$6"
  local workers="$7"

  for order in step_first batch_first; do
    run_intervention "$data" AR "$order" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers"
    run_intervention "$data" CEM "$order" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers"
    run_intervention "$data" CBM "$order" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers"
    run_intervention "$data" SCBM "$order" "$encoder_arch" "$learning_rate" "$weight_decay" "$train_batch_size" "$val_batch_size" "$workers" \
      model.cov_type=amortized model.reg_precision=l1
  done
}

run_dataset synthetic FCNN 0.0002 0.0001 256 512 2
run_dataset CUB resnet18 0.0001 0.0001 64 256 12
run_dataset cifar10 simple_CNN 0.0002 0.0001 256 512 12
