#!/bin/bash
# =============================================================================
# UltraFeedback 1k self-align_v2 generation (single GPU, no hint, prefix=0)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# GPU & environment
# -----------------------------------------------------------------------------
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

ENV_BIN="/apdcephfs_jn3/share_535475/common/dellwu/envs/qwen3_env/bin"
PY="${ENV_BIN}/python"
REPO_ROOT="/apdcephfs_jn3/share_535475/common/dellwu/RE-START"

GEN_SCRIPT="${REPO_ROOT}/evaluate/context_distillation_with_hint.py"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# MODEL_PATH="/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
# MODEL_TAG="DeepSeek-R1-Distill-Qwen-14B"
MODEL_PATH="/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
MODEL_TAG="DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"

DATASET_PATH="${REPO_ROOT}/data/ultrafeedback_train_1k.json"
DATASET_NAME="ultrafeedback_train_1k"

TAG="self_align_v2"
TEMPERATURE=0.6
TOP_P=1.0
N_GENERATION=1
MAX_NEW_TOKENS=4096

OUT_DIR="${REPO_ROOT}/evaluate/results/${DATASET_NAME}_${MODEL_TAG}"
mkdir -p "${OUT_DIR}"
OUTPUT_PATH="${OUT_DIR}/${MODEL_TAG}_${TAG}_prefix_0.json"

echo "========================================"
echo "  GPU_ID      = ${GPU_ID}"
echo "  MODEL       = ${MODEL_PATH}"
echo "  DATASET     = ${DATASET_PATH}"
echo "  OUTPUT_PATH = ${OUTPUT_PATH}"
echo "========================================"

"${PY}" "${GEN_SCRIPT}" \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --tag "${TAG}" \
    --prefix 0 \
    --hint 0 \
    --apply_hint 0 \
    --start_idx 0 \
    --end_idx -1 \
    --tensor_parallel_size 1 \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --n_generation "${N_GENERATION}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --output_path "${OUTPUT_PATH}"

echo -e "\nDone. Result: ${OUTPUT_PATH}"
