#!/bin/bash
# =============================================================================
# Hint-based self-distillation pipeline  (8-GPU shard-parallel, with merge)
#
# 每一步都是：
#   1) 从「上一步的合并结果」取 [s, e) 切片，分到 8 张卡并行处理
#   2) 8 个分片跑完后，调用 merge_shards.py 合并为一份完整文件
#   3) 下一步再以该合并文件作为输入，重新切片
#
# 中间临时产物放在 SHARD_DIR/step{N}/，合并后的完整文件放在 MERGED_DIR/。
#
# Steps:
#   Step 0: cold-start generation (tag=None, no hint, no prefix)
#   Step 1: self_align_v2 + random prefix, no hint
#   Step 2: moderation judge on Step 1
#   Step 3: re-generate on FN (apply_hint=1, prefix=0, hint=16)
#   Step 4: moderation judge on Step 3
#
# Usage:
#   bash run_pipline.sh
#   NUM_SHARDS=4 GPU_IDS="0,1,2,3" bash run_pipline.sh
#   PARALLEL_JUDGE=0 bash run_pipline.sh        # judge 改为顺序跑
#   NUM_SAMPLES=80 bash run_pipline.sh          # 小规模 smoke test
#   SKIP_STEPS="0,1" bash run_pipline.sh        # 复用已有中间结果，跳过某些步骤
#   FORCE_STEP0=1 bash run_pipline.sh           # 强制重跑 Step 0（默认检测到缓存就自动跳过）
# =============================================================================

set -uo pipefail

# -----------------------------------------------------------------------------
# Shard / GPU configuration
# -----------------------------------------------------------------------------
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
if [[ "${#GPU_ARR[@]}" -ne "${NUM_SHARDS}" ]]; then
    echo "[FATAL] NUM_SHARDS=${NUM_SHARDS} but got ${#GPU_ARR[@]} GPUs in GPU_IDS='${GPU_IDS}'"
    exit 1
fi

PARALLEL_JUDGE="${PARALLEL_JUDGE:-1}"
SKIP_STEPS="${SKIP_STEPS:-}"   # 逗号分隔，如 "0,1" 表示跳过 Step 0 和 Step 1
FORCE_STEP0="${FORCE_STEP0:-0}"  # =1 时即使缓存存在也强制重跑 Step 0

export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV_BIN="/apdcephfs_jn3/share_535475/common/dellwu/envs/qwen3_env/bin"
PY="${ENV_BIN}/python"
REPO_ROOT="/apdcephfs_jn3/share_535475/common/dellwu/RE-START"

GEN_SCRIPT="${REPO_ROOT}/evaluate/context_distillation_with_hint.py"
JUDGE_SCRIPT="${REPO_ROOT}/evaluate/moderation_as_judge_v4.py"
MERGE_SCRIPT="${REPO_ROOT}/evaluate/merge_shards.py"

# -----------------------------------------------------------------------------
# Experiment config
# -----------------------------------------------------------------------------
MODEL_PATH="/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
MODEL_TAG="DeepSeek-R1-Distill-Qwen-14B"
# MODEL_PATH="/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
# MODEL_TAG="DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
# MODEL_PATH="/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
# MODEL_TAG="DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"

# Step 0 只依赖基础模型+数据集，不随 peft 版本变化，因此用 BASE_MODEL_TAG 单独标识，
# 同一基础模型下的所有 peft 实验共享同一份 Step 0 产物，避免重复生成。
BASE_MODEL_TAG="${BASE_MODEL_TAG:-DeepSeek-R1-Distill-Qwen-14B}"

DATASET_NAME="wildjailbreak_train"
DATASET_PATH="${REPO_ROOT}/evaluate/harmful_questions/${DATASET_NAME}.json"

START_IDX="${START_IDX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
END_IDX=$((START_IDX + NUM_SAMPLES))

TEMPERATURE=0.6
TOP_P=1.0
N_GENERATION=1
MAX_NEW_TOKENS=4096
TP_SIZE="${TP_SIZE:-1}"       # 每个分片独占一卡时为 1
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

TAG="self_align_v2"
HINT=16
MODERATION_MODEL="wildguard"

# -----------------------------------------------------------------------------
# Paths: SHARD_DIR 放分片临时产物，MERGED_DIR 放每步合并后的完整结果
# -----------------------------------------------------------------------------
OUT_DIR="${REPO_ROOT}/evaluate/results/${DATASET_NAME}_${MODEL_TAG}"
SHARD_DIR="${OUT_DIR}/shards"
MERGED_DIR="${OUT_DIR}/merged"
LOG_DIR="${OUT_DIR}/logs"

# Step 0 的共享缓存目录：由 BASE_MODEL_TAG 决定，与 peft MODEL_TAG 解耦
STEP0_CACHE_DIR="${REPO_ROOT}/evaluate/results/${DATASET_NAME}_${BASE_MODEL_TAG}"

mkdir -p "${OUT_DIR}" "${SHARD_DIR}" "${MERGED_DIR}" "${LOG_DIR}" "${STEP0_CACHE_DIR}"
mkdir -p "${SHARD_DIR}/step0" "${SHARD_DIR}/step1" "${SHARD_DIR}/step2" \
         "${SHARD_DIR}/step3" "${SHARD_DIR}/step4"

# 每一步的合并产物（完整文件）
# Step 0 命名里用 BASE_MODEL_TAG，与 peft MODEL_TAG 解耦，实现跨 peft 实验共享
STEP0_MERGED="${STEP0_CACHE_DIR}/step0_raw_${BASE_MODEL_TAG}_${START_IDX}_${END_IDX}.json"
STEP1_MERGED="${MERGED_DIR}/step1_${MODEL_TAG}_${TAG}_prefix_random_${START_IDX}_${END_IDX}.json"
STEP2_MERGED="${MERGED_DIR}/step2_${MODEL_TAG}_${TAG}_prefix_random_${MODERATION_MODEL}_${START_IDX}_${END_IDX}.json"
STEP3_MERGED="${MERGED_DIR}/step3_${MODEL_TAG}_${TAG}_prefix_0_hint${HINT}_${START_IDX}_${END_IDX}.json"
STEP4_MERGED="${MERGED_DIR}/step4_${MODEL_TAG}_${TAG}_prefix_0_hint${HINT}_${MODERATION_MODEL}_${START_IDX}_${END_IDX}.json"

# ---- Step 0 缓存检测 ----
# 若共享缓存已存在，则默认自动跳过 Step 0（可通过 FORCE_STEP0=1 强制重跑）。
if [[ "${FORCE_STEP0}" != "1" && -s "${STEP0_MERGED}" ]]; then
    if [[ ",${SKIP_STEPS}," != *",0,"* ]]; then
        if [[ -n "${SKIP_STEPS}" ]]; then SKIP_STEPS="${SKIP_STEPS},0"; else SKIP_STEPS="0"; fi
        echo "[cache] Step 0 artifact already exists, auto-added 0 to SKIP_STEPS -> '${SKIP_STEPS}'"
        echo "        ${STEP0_MERGED}"
        echo "        (set FORCE_STEP0=1 to regenerate)"
    fi
fi

echo "========================================"
echo "  NUM_SHARDS     = ${NUM_SHARDS}"
echo "  GPU_IDS        = ${GPU_IDS}"
echo "  PARALLEL_JUDGE = ${PARALLEL_JUDGE}"
echo "  RANGE          = [${START_IDX}, ${END_IDX})  (NUM_SAMPLES=${NUM_SAMPLES})"
echo "  SKIP_STEPS     = '${SKIP_STEPS}'"
echo "  MERGED_DIR     = ${MERGED_DIR}"
echo "  SHARD_DIR      = ${SHARD_DIR}"
echo "  LOG_DIR        = ${LOG_DIR}"
echo "  BASE_MODEL_TAG = ${BASE_MODEL_TAG}"
echo "  STEP0_MERGED   = ${STEP0_MERGED}"
echo "========================================"

# -----------------------------------------------------------------------------
# Helper: 是否跳过某步
# -----------------------------------------------------------------------------
should_skip() {
    local step=$1
    [[ ",${SKIP_STEPS}," == *",${step},"* ]]
}

# -----------------------------------------------------------------------------
# Helper: 将 [0, total) 均分成 NUM_SHARDS 份；返回全局数组 SHARD_S / SHARD_E
# -----------------------------------------------------------------------------
compute_shards() {
    local total=$1
    local base=$((total / NUM_SHARDS))
    local rem=$((total % NUM_SHARDS))
    local cursor=0
    SHARD_S=()
    SHARD_E=()
    local len
    for ((i=0; i<NUM_SHARDS; i++)); do
        len=${base}
        if (( i < rem )); then len=$((len + 1)); fi
        SHARD_S[i]=${cursor}
        SHARD_E[i]=$((cursor + len))
        cursor=${SHARD_E[i]}
    done
}

# -----------------------------------------------------------------------------
# Helper: 等待所有后台 pid，任一失败就 FATAL
# -----------------------------------------------------------------------------
wait_all() {
    local label=$1
    shift
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "[ERROR] ${label}: pid ${pid} failed"
            failed=1
        fi
    done
    if (( failed )); then
        echo "[FATAL] ${label} has failed shards. Check ${LOG_DIR}/"
        exit 1
    fi
    echo "[OK] ${label} all ${#pids[@]} shards done."
}

# -----------------------------------------------------------------------------
# Helper: 合并分片
# -----------------------------------------------------------------------------
do_merge() {
    local step=$1       # "step0" / "step1" / ...
    local pattern=$2    # glob，不含目录
    local output=$3
    local allow_empty=${4:-0}
    local extra=()
    if (( allow_empty )); then extra+=(--allow_empty); fi
    "${PY}" "${MERGE_SCRIPT}" \
        --shard_dir "${SHARD_DIR}/${step}" \
        --pattern "${pattern}" \
        --output "${output}" \
        "${extra[@]}"
}

# =============================================================================
# Step 0: cold-start generation
# =============================================================================
if should_skip 0; then
    echo -e "\n[Step 0] SKIPPED (reusing ${STEP0_MERGED})"
else
    echo -e "\n[Step 0] Cold-start generation (${NUM_SHARDS} shards)"
    compute_shards "${NUM_SAMPLES}"
    declare -a PIDS=()
    for ((i=0; i<NUM_SHARDS; i++)); do
        gpu=${GPU_ARR[i]}
        s=$((START_IDX + SHARD_S[i]))     # 全局下标
        e=$((START_IDX + SHARD_E[i]))
        out="${SHARD_DIR}/step0/shard${i}_${s}_${e}.json"
        log="${LOG_DIR}/step0_shard${i}_gpu${gpu}.log"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            "${PY}" "${GEN_SCRIPT}" \
                --model "${MODEL_PATH}" \
                --dataset "${DATASET_PATH}" \
                --tag None \
                --prefix 0 \
                --hint 0 \
                --apply_hint 0 \
                --start_idx "${s}" \
                --end_idx "${e}" \
                --tensor_parallel_size "${TP_SIZE}" \
                --gpu_memory_utilization "${GPU_MEM_UTIL}" \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --n_generation "${N_GENERATION}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --output_path "${out}"
        ) >"${log}" 2>&1 &
        PIDS+=("$!")
        echo "  shard ${i} gpu=${gpu} range=[${s},${e}) pid=$! log=${log}"
    done
    wait_all "Step 0" "${PIDS[@]}"
    do_merge "step0" "shard*_*_*.json" "${STEP0_MERGED}"
fi

# =============================================================================
# Step 1: self_align_v2 + random prefix (读 STEP0_MERGED，再切片)
# =============================================================================
if should_skip 1; then
    echo -e "\n[Step 1] SKIPPED (reusing ${STEP1_MERGED})"
else
    if [[ ! -f "${STEP0_MERGED}" ]]; then
        echo "[FATAL] ${STEP0_MERGED} not found, cannot run Step 1"; exit 1
    fi
    # STEP0_MERGED 里条数 = NUM_SAMPLES，所以切片还是 NUM_SAMPLES 份
    TOTAL=$("${PY}" -c "import json; print(len(json.load(open('${STEP0_MERGED}'))))")
    echo -e "\n[Step 1] Initial generation with random prefix (${NUM_SHARDS} shards over ${TOTAL} items)"
    compute_shards "${TOTAL}"
    PIDS=()
    for ((i=0; i<NUM_SHARDS; i++)); do
        gpu=${GPU_ARR[i]}
        s=${SHARD_S[i]}
        e=${SHARD_E[i]}
        out="${SHARD_DIR}/step1/shard${i}_${s}_${e}.json"
        log="${LOG_DIR}/step1_shard${i}_gpu${gpu}.log"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            "${PY}" "${GEN_SCRIPT}" \
                --model "${MODEL_PATH}" \
                --dataset "${STEP0_MERGED}" \
                --tag "${TAG}" \
                --prefix random \
                --hint 0 \
                --apply_hint 0 \
                --start_idx "${s}" \
                --end_idx "${e}" \
                --tensor_parallel_size "${TP_SIZE}" \
                --gpu_memory_utilization "${GPU_MEM_UTIL}" \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --n_generation "${N_GENERATION}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --output_path "${out}"
        ) >"${log}" 2>&1 &
        PIDS+=("$!")
        echo "  shard ${i} gpu=${gpu} range=[${s},${e}) pid=$! log=${log}"
    done
    wait_all "Step 1" "${PIDS[@]}"
    do_merge "step1" "shard*_*_*.json" "${STEP1_MERGED}"
fi

# =============================================================================
# Step 2: moderation judge on STEP1_MERGED (切片 → 8 卡并行 judge → 合并)
# =============================================================================
if should_skip 2; then
    echo -e "\n[Step 2] SKIPPED (reusing ${STEP2_MERGED})"
else
    if [[ ! -f "${STEP1_MERGED}" ]]; then
        echo "[FATAL] ${STEP1_MERGED} not found, cannot run Step 2"; exit 1
    fi
    TOTAL=$("${PY}" -c "import json; print(len(json.load(open('${STEP1_MERGED}'))))")
    echo -e "\n[Step 2] Moderation judge (parallel=${PARALLEL_JUDGE}, ${NUM_SHARDS} shards over ${TOTAL} items)"
    compute_shards "${TOTAL}"

    PIDS=()
    for ((i=0; i<NUM_SHARDS; i++)); do
        gpu=${GPU_ARR[i]}
        s=${SHARD_S[i]}
        e=${SHARD_E[i]}
        out="${SHARD_DIR}/step2/shard${i}_${s}_${e}.json"
        log="${LOG_DIR}/step2_shard${i}_gpu${gpu}.log"
        if (( PARALLEL_JUDGE )); then
            (
                export CUDA_VISIBLE_DEVICES="${gpu}"
                "${PY}" "${JUDGE_SCRIPT}" \
                    --response_file "${STEP1_MERGED}" \
                    --start_idx "${s}" \
                    --end_idx "${e}" \
                    --moderation "${MODERATION_MODEL}" \
                    --save_name "${out}"
            ) >"${log}" 2>&1 &
            PIDS+=("$!")
            echo "  shard ${i} gpu=${gpu} range=[${s},${e}) pid=$! log=${log}"
        else
            echo "  [seq] shard ${i} gpu=${gpu} range=[${s},${e}) log=${log}"
            (
                export CUDA_VISIBLE_DEVICES="${gpu}"
                "${PY}" "${JUDGE_SCRIPT}" \
                    --response_file "${STEP1_MERGED}" \
                    --start_idx "${s}" \
                    --end_idx "${e}" \
                    --moderation "${MODERATION_MODEL}" \
                    --save_name "${out}"
            ) >"${log}" 2>&1 || { echo "[FATAL] Step 2 shard ${i} failed"; exit 1; }
        fi
    done
    if (( PARALLEL_JUDGE )); then
        wait_all "Step 2" "${PIDS[@]}"
    fi
    do_merge "step2" "shard*_*_*.json" "${STEP2_MERGED}"
fi

# =============================================================================
# Step 3: re-generate on FN (apply_hint=1, prefix=0)
#   - load_dataset 内部会筛 FN，所以我们把整个 STEP2_MERGED 分 8 份喂进去，
#     每个分片各自过滤后生成。
# =============================================================================
if should_skip 3; then
    echo -e "\n[Step 3] SKIPPED (reusing ${STEP3_MERGED})"
else
    if [[ ! -f "${STEP2_MERGED}" ]]; then
        echo "[FATAL] ${STEP2_MERGED} not found, cannot run Step 3"; exit 1
    fi
    TOTAL=$("${PY}" -c "import json; print(len(json.load(open('${STEP2_MERGED}'))))")
    echo -e "\n[Step 3] Re-generate FN with hint (${NUM_SHARDS} shards over ${TOTAL} items)"
    compute_shards "${TOTAL}"
    PIDS=()
    for ((i=0; i<NUM_SHARDS; i++)); do
        gpu=${GPU_ARR[i]}
        s=${SHARD_S[i]}
        e=${SHARD_E[i]}
        out="${SHARD_DIR}/step3/shard${i}_${s}_${e}.json"
        log="${LOG_DIR}/step3_shard${i}_gpu${gpu}.log"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            "${PY}" "${GEN_SCRIPT}" \
                --model "${MODEL_PATH}" \
                --dataset "${STEP2_MERGED}" \
                --tag "${TAG}" \
                --prefix 0 \
                --hint "${HINT}" \
                --apply_hint 1 \
                --start_idx "${s}" \
                --end_idx "${e}" \
                --tensor_parallel_size "${TP_SIZE}" \
                --gpu_memory_utilization "${GPU_MEM_UTIL}" \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --n_generation "${N_GENERATION}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --output_path "${out}"
        ) >"${log}" 2>&1 &
        PIDS+=("$!")
        echo "  shard ${i} gpu=${gpu} range=[${s},${e}) pid=$! log=${log}"
    done
    wait_all "Step 3" "${PIDS[@]}"
    # Step 3 某些分片可能 FN=0 → 无输出，allow_empty 允许空合集
    do_merge "step3" "shard*_*_*.json" "${STEP3_MERGED}" 1
fi

# =============================================================================
# Step 4: moderation judge on STEP3_MERGED
# =============================================================================
if should_skip 4; then
    echo -e "\n[Step 4] SKIPPED (reusing ${STEP4_MERGED})"
else
    if [[ ! -s "${STEP3_MERGED}" ]]; then
        echo "[Step 4] ${STEP3_MERGED} empty or missing (no FN across all shards). Skipping."
    else
        TOTAL=$("${PY}" -c "import json; print(len(json.load(open('${STEP3_MERGED}'))))")
        if [[ "${TOTAL}" -eq 0 ]]; then
            echo "[Step 4] STEP3_MERGED has 0 items. Skipping."
        else
            # judge 分片数不超过样本数
            JUDGE_SHARDS=${NUM_SHARDS}
            if (( TOTAL < JUDGE_SHARDS )); then JUDGE_SHARDS=${TOTAL}; fi
            echo -e "\n[Step 4] Moderation judge (parallel=${PARALLEL_JUDGE}, ${JUDGE_SHARDS} shards over ${TOTAL} items)"
            # 让每张卡只对 STEP3_MERGED 的一个 [s, e) 片段做 judge，
            # moderation_as_judge_v4.py 支持 --start_idx/--end_idx，直接切片即可。
            PREV_NUM_SHARDS=${NUM_SHARDS}
            NUM_SHARDS=${JUDGE_SHARDS}
            compute_shards "${TOTAL}"
            NUM_SHARDS=${PREV_NUM_SHARDS}

            PIDS=()
            for ((i=0; i<JUDGE_SHARDS; i++)); do
                gpu=${GPU_ARR[i]}
                s=${SHARD_S[i]}
                e=${SHARD_E[i]}
                out="${SHARD_DIR}/step4/shard${i}_${s}_${e}.json"
                log="${LOG_DIR}/step4_shard${i}_gpu${gpu}.log"
                if (( PARALLEL_JUDGE )); then
                    (
                        export CUDA_VISIBLE_DEVICES="${gpu}"
                        "${PY}" "${JUDGE_SCRIPT}" \
                            --response_file "${STEP3_MERGED}" \
                            --start_idx "${s}" \
                            --end_idx "${e}" \
                            --moderation "${MODERATION_MODEL}" \
                            --save_name "${out}"
                    ) >"${log}" 2>&1 &
                    PIDS+=("$!")
                    echo "  shard ${i} gpu=${gpu} range=[${s},${e}) pid=$! log=${log}"
                else
                    echo "  [seq] shard ${i} gpu=${gpu} range=[${s},${e}) log=${log}"
                    (
                        export CUDA_VISIBLE_DEVICES="${gpu}"
                        "${PY}" "${JUDGE_SCRIPT}" \
                            --response_file "${STEP3_MERGED}" \
                            --start_idx "${s}" \
                            --end_idx "${e}" \
                            --moderation "${MODERATION_MODEL}" \
                            --save_name "${out}"
                    ) >"${log}" 2>&1 || { echo "[FATAL] Step 4 shard ${i} failed"; exit 1; }
                fi
            done
            if (( PARALLEL_JUDGE )); then
                wait_all "Step 4" "${PIDS[@]}"
            fi
            do_merge "step4" "shard*_*_*.json" "${STEP4_MERGED}" 1
        fi
    fi
fi

echo -e "\n========================================"
echo "Pipeline finished."
echo "  Step 0 merged : ${STEP0_MERGED}"
echo "  Step 1 merged : ${STEP1_MERGED}"
echo "  Step 2 merged : ${STEP2_MERGED}"
echo "  Step 3 merged : ${STEP3_MERGED}"
echo "  Step 4 merged : ${STEP4_MERGED}"
echo "  Shard temps   : ${SHARD_DIR}"
echo "  Logs          : ${LOG_DIR}"
echo "========================================"
