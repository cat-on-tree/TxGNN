#!/usr/bin/env bash

DEVICE=${1:-0}

SPLIT=${SPLIT:-"full_graph"}
SEED=${SEED:-42}

# Official pretrained TxGNN checkpoint directory.
# It must contain config.pkl and model.pt.
TXGNN_PRETRAINED_PATH=${TXGNN_PRETRAINED_PATH:-"./TxGNNExplorer"}

# Paths
TRAIN_FOLDER=${TRAIN_FOLDER:-"./data/kg"}
MODEL_ROOT=${MODEL_ROOT:-"./saved_models"}
TEST_DIR=${TEST_DIR:-"./data/benchmark/GNN"}
RESULT_DIR=${RESULT_DIR:-"./result"}
mkdir -p "${MODEL_ROOT}"
mkdir -p "${RESULT_DIR}"

# Debug/local run settings
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-500}

if [[ "${DEVICE}" == "cpu" ]]; then
  EVAL_DEVICE="cpu"
else
  EVAL_DEVICE="cuda:${DEVICE}"
fi

if [[ -z "${TXGNN_PRETRAINED_PATH}" ]]; then
  echo "ERROR: TXGNN_PRETRAINED_PATH is empty."
  echo "Usage:"
  echo "  TXGNN_PRETRAINED_PATH=./TxGNNExplorer bash script/train.sh 0"
  echo "  TXGNN_PRETRAINED_PATH=./TxGNNExplorer FINETUNE_EPOCHS=1 bash script/train.sh cpu"
  exit 1
fi

echo "============================================================"
echo "Training + evaluation configuration"
echo "============================================================"
echo "DEVICE=${DEVICE}"
echo "EVAL_DEVICE=${EVAL_DEVICE}"
echo "SPLIT=${SPLIT}"
echo "SEED=${SEED}"
echo "TXGNN_PRETRAINED_PATH=${TXGNN_PRETRAINED_PATH}"
echo "TRAIN_FOLDER=${TRAIN_FOLDER}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "TEST_DIR=${TEST_DIR}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "FINETUNE_EPOCHS=${FINETUNE_EPOCHS}"
echo "============================================================"

# Random-init baselines, finetune only.
for MODEL in GNN TxGAT HGT
do
  echo "============================================================"
  echo "Training ${MODEL} on ${SPLIT}, seed=${SEED}"
  echo "============================================================"

  python script/train.py \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --split "${SPLIT}" \
    --model "${MODEL}" \
    --data_folder "${TRAIN_FOLDER}" \
    --save_root "${MODEL_ROOT}" \
    --finetune_epochs "${FINETUNE_EPOCHS}"
done

# Official pretrained TxGNN, finetune only.
echo "============================================================"
echo "Training TxGNN pretrained on ${SPLIT}, seed=${SEED}"
echo "============================================================"

python script/train.py \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --split "${SPLIT}" \
  --model TxGNN \
  --data_folder "${TRAIN_FOLDER}" \
  --save_root "${MODEL_ROOT}" \
  --pretrained_path "${TXGNN_PRETRAINED_PATH}" \
  --finetune_epochs "${FINETUNE_EPOCHS}"

echo "============================================================"
echo "Evaluating trained models on fixed GNN test sets"
echo "============================================================"

python script/evaluate.py \
  --data-folder "${TRAIN_FOLDER}" \
  --split "${SPLIT}" \
  --seed "${SEED}" \
  --device "${EVAL_DEVICE}" \
  --test-dir "${TEST_DIR}" \
  --model-paths \
    "${MODEL_ROOT}/GNN_${SEED}_${SPLIT}" \
    "${MODEL_ROOT}/TxGAT_${SEED}_${SPLIT}" \
    "${MODEL_ROOT}/HGT_${SEED}_${SPLIT}" \
    "${MODEL_ROOT}/TxGNN_${SEED}_${SPLIT}_pretrained" \
  --result-dir "${RESULT_DIR}"

echo "============================================================"
echo "Training and evaluation finished."
echo "Results saved to: ${RESULT_DIR}"
echo "============================================================"