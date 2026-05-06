#!/bin/bash
# =============================================================================
# UltraFeedback 1k RealSafe-R1 generation (single GPU, no hint, prefix=0)
#   + wildguard moderation (产物: ${MODEL_TAG}_RealSafe-R1_prefix_0_wildguard.json)
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
JUDGE_SCRIPT="${REPO_ROOT}/evaluate/moderation_as_judge_v4.py"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_PATH="/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
MODEL_TAG="DeepSeek-R1-Distill-Qwen-14B"
# MODEL_PATH="/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
# MODEL_TAG="DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"

DATASET_PATH="${REPO_ROOT}/data/ultrafeedback_train_1k.json"
DATASET_NAME="ultrafeedback_train_1k"

TAG="RealSafe-R1"
TEMPERATURE=0.6
TOP_P=1.0
N_GENERATION=1
MAX_NEW_TOKENS=4096
MODERATION_MODEL="wildguard"

OUT_DIR="${REPO_ROOT}/evaluate/results/${DATASET_NAME}_${MODEL_TAG}"
mkdir -p "${OUT_DIR}"
OUTPUT_PATH="${OUT_DIR}/${MODEL_TAG}_${TAG}_prefix_0.json"
WILDGUARD_PATH="${OUT_DIR}/${MODEL_TAG}_${TAG}_prefix_0_${MODERATION_MODEL}.json"

echo "========================================"
echo "  GPU_ID         = ${GPU_ID}"
echo "  MODEL          = ${MODEL_PATH}"
echo "  DATASET        = ${DATASET_PATH}"
echo "  OUTPUT_PATH    = ${OUTPUT_PATH}"
echo "  WILDGUARD_PATH = ${WILDGUARD_PATH}"
echo "========================================"

# -----------------------------------------------------------------------------
# Stage A: vLLM 生成（RealSafe-R1 模板，prefix=0，无 hint）
#   - 若产物已存在且非空则跳过（幂等）
# -----------------------------------------------------------------------------
if [[ -s "${OUTPUT_PATH}" ]]; then
    echo "[run_ultrafeedback] generation output already exists, skipping: ${OUTPUT_PATH}"
else
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
fi

if [[ ! -s "${OUTPUT_PATH}" ]]; then
    echo "[FATAL] generation produced no output: ${OUTPUT_PATH}"
    exit 3
fi
echo "[stage A done] generation: ${OUTPUT_PATH}"

# -----------------------------------------------------------------------------
# Stage B: wildguard moderation
#   - 显式通过 --save_name 指定绝对路径，避免 JUDGE_SCRIPT 按响应文件名再拼一次
#   - 若产物已存在且非空则跳过（幂等）
# -----------------------------------------------------------------------------
if [[ -s "${WILDGUARD_PATH}" ]]; then
    echo "[run_ultrafeedback] wildguard output already exists, skipping: ${WILDGUARD_PATH}"
else
    "${PY}" "${JUDGE_SCRIPT}" \
        --response_file "${OUTPUT_PATH}" \
        --moderation "${MODERATION_MODEL}" \
        --save_name "${WILDGUARD_PATH}"
fi

if [[ ! -s "${WILDGUARD_PATH}" ]]; then
    echo "[FATAL] wildguard artifact missing or empty: ${WILDGUARD_PATH}"
    exit 3
fi

echo -e "\nDone. Generation : ${OUTPUT_PATH}"
echo "      Wildguard  : ${WILDGUARD_PATH}"
