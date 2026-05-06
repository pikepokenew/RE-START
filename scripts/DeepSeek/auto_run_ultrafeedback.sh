#!/bin/bash
# =============================================================================
# [AUTO] UltraFeedback 1k self-align_v2 generation + wildguard moderation
#
# 本脚本是 run_ultrafeedback.sh 的自动化副本（auto_ 前缀），在业务行为上与原
# 脚本保持一致（单卡 vLLM 生成 prefix=0, hint=0, tag=self_align_v2），额外在末尾
# 追加一次 wildguard moderation，使最终产物命名为：
#   ${MODEL_TAG}_self_align_v2_prefix_0_wildguard.json
# 以便 auto_make_train_data.py 直接引用。
#
# Required env vars:
#   MODEL_PATH  - 生成模型的权重路径
#   MODEL_TAG   - 生成模型的 tag（用于产物文件命名）
# Optional env vars:
#   GPU_ID      - 默认 0
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# [AUTO] 强制校验必填环境变量
# -----------------------------------------------------------------------------
if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "[FATAL][auto_run_ultrafeedback] MODEL_PATH env var is required."
    exit 2
fi
if [[ -z "${MODEL_TAG:-}" ]]; then
    echo "[FATAL][auto_run_ultrafeedback] MODEL_TAG env var is required."
    exit 2
fi

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
# Config (MODEL_PATH / MODEL_TAG from env)
# -----------------------------------------------------------------------------
DATASET_PATH="${REPO_ROOT}/data/ultrafeedback_train_1k.json"
DATASET_NAME="ultrafeedback_train_1k"

TAG="self_align_v2"
TEMPERATURE=0.6
TOP_P=1.0
N_GENERATION=1
MAX_NEW_TOKENS=4096
MODERATION_MODEL="wildguard"

OUT_DIR="${REPO_ROOT}/evaluate/results/${DATASET_NAME}_${MODEL_TAG}"
mkdir -p "${OUT_DIR}"
OUTPUT_PATH="${OUT_DIR}/${MODEL_TAG}_${TAG}_prefix_0.json"
WILDGUARD_PATH="${OUT_DIR}/${MODEL_TAG}_${TAG}_prefix_0_wildguard.json"

echo "========================================"
echo "  [auto] GPU_ID        = ${GPU_ID}"
echo "  [auto] MODEL_PATH    = ${MODEL_PATH}"
echo "  [auto] MODEL_TAG     = ${MODEL_TAG}"
echo "  DATASET              = ${DATASET_PATH}"
echo "  OUTPUT_PATH          = ${OUTPUT_PATH}"
echo "  WILDGUARD_PATH       = ${WILDGUARD_PATH}"
echo "========================================"

# -----------------------------------------------------------------------------
# Stage A: vLLM 生成（与原 run_ultrafeedback.sh 行为一致）
#   - 如果产物已存在且非空，则跳过生成阶段（幂等）
# -----------------------------------------------------------------------------
if [[ -s "${OUTPUT_PATH}" ]]; then
    echo "[auto_run_ultrafeedback] generation output already exists, skipping: ${OUTPUT_PATH}"
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
    echo "[FATAL][auto_run_ultrafeedback] generation produced no output: ${OUTPUT_PATH}"
    exit 3
fi
echo "[stage A done] generation: ${OUTPUT_PATH}"

# -----------------------------------------------------------------------------
# Stage B: wildguard moderation（填补原脚本缺失的 moderation 步骤）
#   - 通过 --save_name 显式指定绝对路径，产物为 ${MODEL_TAG}_self_align_v2_prefix_0_wildguard.json
#   - 如果已存在则跳过
# -----------------------------------------------------------------------------
if [[ -s "${WILDGUARD_PATH}" ]]; then
    echo "[auto_run_ultrafeedback] wildguard output already exists, skipping: ${WILDGUARD_PATH}"
else
    "${PY}" "${JUDGE_SCRIPT}" \
        --response_file "${OUTPUT_PATH}" \
        --moderation "${MODERATION_MODEL}" \
        --save_name "${WILDGUARD_PATH}"
fi

# -----------------------------------------------------------------------------
# [AUTO] 最终产物校验
# -----------------------------------------------------------------------------
if [[ ! -s "${WILDGUARD_PATH}" ]]; then
    echo "[FATAL][auto_run_ultrafeedback] required wildguard artifact missing or empty: ${WILDGUARD_PATH}"
    exit 3
fi

echo -e "\nDone. Generation : ${OUTPUT_PATH}"
echo "      Wildguard  : ${WILDGUARD_PATH}"
echo "[auto_run_ultrafeedback] OK"
