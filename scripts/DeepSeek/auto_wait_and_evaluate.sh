#!/bin/bash
# =============================================================================
# [AUTO] 等待目标模型就绪后，自动触发 auto_run_evaluation_pool.sh 评估
# -----------------------------------------------------------------------------
# 行为：
#   1. 轮询 ${MODEL_PATH} 是否就绪：每 ${POLL_INTERVAL} 秒检测一次，直到就绪或超时
#      - 就绪判据：路径存在，且（若是目录）目录内存在 config.json 文件（兼容 HF/merged 模型）
#        单纯的"目录存在"不够，因为训练过程中目录可能先被创建但尚未写入权重
#   2. 就绪后，额外等待 ${POST_READY_WAIT} 秒（默认 60s）给文件系统同步 & 进程收尾留缓冲
#   3. 以 MODEL_LIST_OVERRIDE="${MODEL_TAG}|${MODEL_PATH}" 调用 auto_run_evaluation_pool.sh
#
# Required env vars:
#   MODEL_PATH  - 目标模型的绝对路径（通常是 merged 模型目录）
#   MODEL_TAG   - 评估中使用的模型 tag（日志/产物命名）
#
# Optional env vars:
#   POLL_INTERVAL   - 未就绪时的轮询间隔秒数，默认 60
#   POST_READY_WAIT - 就绪后启动评估前的额外等待秒数，默认 60
#   WAIT_TIMEOUT    - 最长等待秒数，默认 0 (=无限等待)；超过则非零退出
#   READINESS_FILE  - 自定义就绪文件名，默认 "config.json"
#                    （若你希望更严格，可改成 "model.safetensors.index.json" 等权重清单）
#   STRICT_READY    - =1 时要求目录同时含 READINESS_FILE 且有至少一个 *.safetensors 权重文件
#   EVAL_SCRIPT     - 评估脚本路径，默认指向 auto_run_evaluation_pool.sh
#
# Usage:
#   MODEL_PATH=/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/<tag> \
#   MODEL_TAG=<tag> \
#       bash scripts/DeepSeek/auto_wait_and_evaluate.sh
#
#   # 后台跑：
#   MODEL_PATH=... MODEL_TAG=... \
#       nohup bash scripts/DeepSeek/auto_wait_and_evaluate.sh \
#       > logs/wait_eval.$(date +%Y%m%d_%H%M%S).log 2>&1 &
# =============================================================================

set -u

# ---------------------------------------------------------------------------
# 必填校验
# ---------------------------------------------------------------------------
if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "[FATAL][auto_wait_and_evaluate] MODEL_PATH env var is required."
    exit 2
fi
if [[ -z "${MODEL_TAG:-}" ]]; then
    echo "[FATAL][auto_wait_and_evaluate] MODEL_TAG env var is required."
    exit 2
fi

POLL_INTERVAL="${POLL_INTERVAL:-60}"
POST_READY_WAIT="${POST_READY_WAIT:-60}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-0}"
READINESS_FILE="${READINESS_FILE:-config.json}"
STRICT_READY="${STRICT_READY:-0}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/apdcephfs_jn3/share_535475/common/dellwu/RE-START/scripts/DeepSeek/auto_run_evaluation_pool.sh}"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "[FATAL][auto_wait_and_evaluate] EVAL_SCRIPT not found: ${EVAL_SCRIPT}"
    exit 2
fi

# ---------------------------------------------------------------------------
# 就绪判据
#   - 路径是文件：存在且非空即就绪
#   - 路径是目录：需要包含 READINESS_FILE；STRICT_READY=1 时还要至少一个 *.safetensors
# ---------------------------------------------------------------------------
is_model_ready() {
    local p="$1"
    if [[ ! -e "${p}" ]]; then
        return 1
    fi
    if [[ -f "${p}" ]]; then
        [[ -s "${p}" ]] && return 0 || return 1
    fi
    if [[ -d "${p}" ]]; then
        if [[ ! -s "${p}/${READINESS_FILE}" ]]; then
            return 1
        fi
        if [[ "${STRICT_READY}" == "1" ]]; then
            # 至少存在一个 safetensors 权重文件
            if ! compgen -G "${p}"/*.safetensors > /dev/null; then
                return 1
            fi
        fi
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# 打印规划
# ---------------------------------------------------------------------------
echo "========================================"
echo "[auto_wait_and_evaluate] plan"
echo "  MODEL_TAG        = ${MODEL_TAG}"
echo "  MODEL_PATH       = ${MODEL_PATH}"
echo "  POLL_INTERVAL    = ${POLL_INTERVAL}s"
echo "  POST_READY_WAIT  = ${POST_READY_WAIT}s"
echo "  WAIT_TIMEOUT     = ${WAIT_TIMEOUT}s (0 = infinite)"
echo "  READINESS_FILE   = ${READINESS_FILE}"
echo "  STRICT_READY     = ${STRICT_READY}"
echo "  EVAL_SCRIPT      = ${EVAL_SCRIPT}"
echo "========================================"

# ---------------------------------------------------------------------------
# 轮询等待模型就绪
# ---------------------------------------------------------------------------
start_ts=$(date +%s)
tick=0
while true; do
    if is_model_ready "${MODEL_PATH}"; then
        echo "[$(date '+%F %T')] ✅ model ready: ${MODEL_PATH}"
        break
    fi

    now_ts=$(date +%s)
    elapsed=$(( now_ts - start_ts ))
    if [[ "${WAIT_TIMEOUT}" -gt 0 && "${elapsed}" -ge "${WAIT_TIMEOUT}" ]]; then
        echo "[FATAL][auto_wait_and_evaluate] timed out after ${elapsed}s waiting for: ${MODEL_PATH}"
        exit 3
    fi

    tick=$(( tick + 1 ))
    echo "[$(date '+%F %T')] ⏳ tick ${tick}: model not ready yet (elapsed=${elapsed}s), sleeping ${POLL_INTERVAL}s ..."
    sleep "${POLL_INTERVAL}"
done

# ---------------------------------------------------------------------------
# 就绪后冷静期
# ---------------------------------------------------------------------------
if [[ "${POST_READY_WAIT}" -gt 0 ]]; then
    echo "[$(date '+%F %T')] sleeping ${POST_READY_WAIT}s before launching evaluation ..."
    sleep "${POST_READY_WAIT}"
fi

# ---------------------------------------------------------------------------
# 再次校验（冷静期内可能有异常被移除等极端情况）
# ---------------------------------------------------------------------------
if ! is_model_ready "${MODEL_PATH}"; then
    echo "[FATAL][auto_wait_and_evaluate] model became unavailable during post-ready wait: ${MODEL_PATH}"
    exit 3
fi

# ---------------------------------------------------------------------------
# 启动评估
#   auto_run_evaluation_pool.sh 要求 MODEL_LIST_OVERRIDE 格式 "name|path"
# ---------------------------------------------------------------------------
export MODEL_LIST_OVERRIDE="${MODEL_TAG}|${MODEL_PATH}"
echo "========================================"
echo "[$(date '+%F %T')] 🚀 launching evaluation"
echo "  MODEL_LIST_OVERRIDE = ${MODEL_LIST_OVERRIDE}"
echo "  CMD                 = bash ${EVAL_SCRIPT}"
echo "========================================"

bash "${EVAL_SCRIPT}"
rc=$?

echo "========================================"
echo "[$(date '+%F %T')] evaluation finished (rc=${rc})"
echo "========================================"
exit "${rc}"
