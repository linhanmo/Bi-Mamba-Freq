#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

DATASETS=(
  "ETTh1"
  "ETTh2"
  "ETTm1"
  "ETTm2"
  "electricity"
  "exchange_rate"
  "traffic"
  "weather"
)

HORIZONS=(96 192 336 720)

RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/datasets}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/data}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-${PROJECT_ROOT}/experiments}"
LOG_ROOT="${LOG_ROOT:-${EXPERIMENTS_ROOT}/batch_logs}"

SEQ_LEN="${SEQ_LEN:-96}"
STRIDE="${STRIDE:-1}"
LR="${LR:-0.00005}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
D_MODEL="${D_MODEL:-128}"
D_STATE="${D_STATE:-16}"
RANK_RATIO="${RANK_RATIO:-0.5}"
N_LAYERS="${N_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
NUM_WORKERS="${NUM_WORKERS:-}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
AMP="${AMP:-1}"

mkdir -p "${LOG_ROOT}"

if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -f "/home/lin/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/home/lin/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" || {
      echo "Failed to activate conda env: ${CONDA_ENV}"
      exit 1
    }
  else
    echo "conda.sh not found, cannot activate CONDA_ENV=${CONDA_ENV}"
    exit 1
  fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_ROOT}/summary_${TIMESTAMP}.log"
FAILED_RUNS=()
SUCCESS_COUNT=0
TOTAL_COUNT=0

echo "Project root: ${PROJECT_ROOT}" | tee -a "${SUMMARY_LOG}"
echo "Raw root: ${RAW_ROOT}" | tee -a "${SUMMARY_LOG}"
echo "Processed root: ${PROCESSED_ROOT}" | tee -a "${SUMMARY_LOG}"
echo "Experiments root: ${EXPERIMENTS_ROOT}" | tee -a "${SUMMARY_LOG}"
echo "Python: $(command -v "${PYTHON_BIN}")" | tee -a "${SUMMARY_LOG}"
echo "Start time: $(date)" | tee -a "${SUMMARY_LOG}"
echo | tee -a "${SUMMARY_LOG}"

for dataset in "${DATASETS[@]}"; do
  for horizon in "${HORIZONS[@]}"; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    RUN_NAME="${dataset}_h${horizon}"
    LOG_FILE="${LOG_ROOT}/${RUN_NAME}_${TIMESTAMP}.log"

    echo "==============================" | tee -a "${SUMMARY_LOG}"
    echo "Running ${RUN_NAME}" | tee -a "${SUMMARY_LOG}"
    echo "Log file: ${LOG_FILE}" | tee -a "${SUMMARY_LOG}"

    CMD=(
      "${PYTHON_BIN}" -m training.train_long_forecast
      --dataset-name "${dataset}"
      --raw-root "${RAW_ROOT}"
      --processed-root "${PROCESSED_ROOT}"
      --experiments-root "${EXPERIMENTS_ROOT}"
      --seq-len "${SEQ_LEN}"
      --pred-len "${horizon}"
      --stride "${STRIDE}"
      --lr "${LR}"
      --weight-decay "${WEIGHT_DECAY}"
      --batch-size "${BATCH_SIZE}"
      --epochs "${EPOCHS}"
      --patience "${PATIENCE}"
      --d-model "${D_MODEL}"
      --d-state "${D_STATE}"
      --rank-ratio "${RANK_RATIO}"
      --n-layers "${N_LAYERS}"
      --dropout "${DROPOUT}"
      --seed "${SEED}"
    )

    if [[ -n "${NUM_WORKERS}" ]]; then
      CMD+=(--num-workers "${NUM_WORKERS}")
    fi

    if [[ "${AMP}" == "1" ]]; then
      CMD+=(--amp)
    fi

    if [[ -n "${EXTRA_ARGS}" ]]; then
      # Allow callers to append flags such as "--cpu" or "--disable-pscan".
      # shellcheck disable=SC2206
      EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
      CMD+=("${EXTRA_ARGS_ARRAY[@]}")
    fi

    echo "Command: PYTHONPATH=src ${CMD[*]}" | tee -a "${SUMMARY_LOG}"

    # Keep stderr attached to terminal so tqdm progress bars stay visible.
    # Stdout is tee'd into a per-run log file.
    if PYTHONPATH=src "${CMD[@]}" | tee "${LOG_FILE}"; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      echo "Status: SUCCESS" | tee -a "${SUMMARY_LOG}"
    else
      FAILED_RUNS+=("${RUN_NAME}")
      echo "Status: FAILED" | tee -a "${SUMMARY_LOG}"
      echo "Last 20 log lines:" | tee -a "${SUMMARY_LOG}"
      tail -n 20 "${LOG_FILE}" | tee -a "${SUMMARY_LOG}"
    fi
    echo | tee -a "${SUMMARY_LOG}"
  done
done

echo "==============================" | tee -a "${SUMMARY_LOG}"
echo "Finished at: $(date)" | tee -a "${SUMMARY_LOG}"
echo "Succeeded: ${SUCCESS_COUNT}/${TOTAL_COUNT}" | tee -a "${SUMMARY_LOG}"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo "Failed runs: ${FAILED_RUNS[*]}" | tee -a "${SUMMARY_LOG}"
  exit 1
fi

echo "All runs completed successfully." | tee -a "${SUMMARY_LOG}"
