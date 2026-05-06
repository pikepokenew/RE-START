#!/bin/bash
# =============================================================================
# [AUTO] 8 卡 GPU-pool 并行评测（推理 + 裁判一体化作业）
# -----------------------------------------------------------------------------
# 本脚本是 run_evaluation_pool.sh 的自动化副本（auto_ 前缀），任务逻辑、判官选择、
# CodeAttack 作业、GPU 池调度与原脚本完全一致；唯一差别在于：
#   * model_list 不再使用脚本内硬编码列表，而是由环境变量 MODEL_LIST_OVERRIDE
#     （格式 "name1|path1;name2|path2"，分号分隔）动态注入。
#   * 如未设置 MODEL_LIST_OVERRIDE 则直接非零码退出（auto 版本强制 CLI）。
#
# Usage:
#   MODEL_LIST_OVERRIDE="tag1|/path1;tag2|/path2" \
#       bash scripts/DeepSeek/auto_run_evaluation_pool.sh
# =============================================================================
set -u
conda activate /apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env

# ================= [AUTO] MODEL_LIST_OVERRIDE required =================
if [[ -z "${MODEL_LIST_OVERRIDE:-}" ]]; then
    echo "[FATAL][auto_run_evaluation_pool] MODEL_LIST_OVERRIDE env var is required."
    echo "  Format: 'name1|path1;name2|path2'"
    exit 2
fi

# ================= 配置 =================
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}

stage1="${STAGE_SAFETY:-true}"
stage1_1="${STAGE_OVER_SAFETY:-true}"
stage1_2="${STAGE_JAILBREAK:-true}"
stage_codeattack="${STAGE_CODEATTACK:-true}"

WORK_ROOT="/apdcephfs_jn3/share_535475/common/dellwu/RE-START/"
OVER_SAFETY_DIR="${WORK_ROOT}/evaluate/over_safety"
LOG_DIR="${WORK_ROOT}/logs/auto_eval_pool_$(date +%Y%m%d_%H%M%S)"

TAG="${TAG:-none}"
N_GENERATION=1
MAX_NEW_TOKENS=4096
TEMPERATURE=0.0
TOP_P=1.0
SEED=42
TP_SIZE=1
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

SAVE_PATH="evaluate/results"
AGGREGATE="best-of-n"

# ================= [AUTO] Parse MODEL_LIST_OVERRIDE =================
# 将 "a|p1;b|p2" 切成数组 model_list=("a|p1" "b|p2")
model_list=()
IFS=';' read -r -a _model_entries <<< "${MODEL_LIST_OVERRIDE}"
for _e in "${_model_entries[@]}"; do
    # 剪去可能的前后空白
    _e="${_e#"${_e%%[![:space:]]*}"}"
    _e="${_e%"${_e##*[![:space:]]}"}"
    [[ -z "${_e}" ]] && continue
    if [[ "${_e}" != *"|"* ]]; then
        echo "[FATAL][auto_run_evaluation_pool] invalid entry (missing '|'): '${_e}'"
        exit 2
    fi
    model_list+=("${_e}")
done
if [[ "${#model_list[@]}" -eq 0 ]]; then
    echo "[FATAL][auto_run_evaluation_pool] MODEL_LIST_OVERRIDE parsed to empty list."
    exit 2
fi
echo "[auto_run_evaluation_pool] parsed ${#model_list[@]} model entries:"
for _m in "${model_list[@]}"; do echo "  - ${_m}"; done

safety_dataset_name_list=("advbench" "HEx-PHI")
over_safety_dataset_name_list=("OKTest" "XSTest")
jailbreak_dataset_name_list=(
    "CodeChameleon_target_llama-2_tree"
    "CodeChameleon_target_llama-2_list"
    "CodeChameleon_target_llama-2_reverse"
    "CodeChameleon_target_llama-2_half"
    "ReNeLLM_target_llama-2"
    "Salad-attack_enhanced_set_sub_v1"
    "DeepInception_target_llama-2"
    "wildjailbreak_eval_v1"
)

# ---- CodeAttack ----
codeattack_exp_name_list=("python_string_full" "python_list_full" "python_stack_full")
CODEATTACK_DIR="${WORK_ROOT}/CodeAttack"
CODEATTACK_DATA_KEY="code_wrapped_plain_attack"
CODEATTACK_MAX_NEW_TOKENS=16000
CODEATTACK_TAG="${CODEATTACK_TAG:-${TAG}}"
CODEATTACK_PROMPT_TYPE="code_python"

# ================= 工具函数 =================

ensure_over_safety_symlinks() {
    local target_dir="${WORK_ROOT}/evaluate/harmful_questions"
    mkdir -p "${target_dir}"
    for name in "${over_safety_dataset_name_list[@]}"; do
        local src="${OVER_SAFETY_DIR}/${name}.json"
        local dst="${target_dir}/${name}.json"
        [[ -e "${dst}" ]] && continue
        if [[ -e "${src}" ]]; then
            ln -sf "${src}" "${dst}"
        else
            echo "[WARN] over-safety source missing: ${src}"
        fi
    done
}

build_response_file() {
    local dataset_name="$1" base_model="$2"
    local mb; mb="$(basename "${base_model}")"
    if [[ "${N_GENERATION}" != "1" ]]; then
        echo "${SAVE_PATH}/${dataset_name}/${mb}_${N_GENERATION}_tag_${TAG}.json"
    else
        echo "${SAVE_PATH}/${dataset_name}/${mb}_tag_${TAG}.json"
    fi
}

build_codeattack_response_file() {
    local exp_name="$1" base_model="$2"
    local mb; mb="$(basename "${base_model}")"
    echo "${SAVE_PATH}/CodeAttack_${exp_name}_${CODEATTACK_DATA_KEY}/${mb}_tag_${CODEATTACK_TAG}.json"
}

# ========= 单作业：Generic =========
run_generic_job() {
    local gpu_id="$1" stage_name="$2" judge_model="$3"
    local dataset_name="$4" base_model="$5" log_file="$6"
    local response_file; response_file="$(build_response_file "${dataset_name}" "${base_model}")"

    {
        echo ">> [GPU ${gpu_id}] ${stage_name}/${dataset_name} | $(date '+%F %T')"

        CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate/run_inference.py \
            --model "${base_model}" \
            --datasets "${dataset_name}" \
            --save_path "${SAVE_PATH}" \
            --tag "${TAG}" \
            --n_generation "${N_GENERATION}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --seed "${SEED}" \
            --tensor_parallel_size "${TP_SIZE}" \
            --gpu_memory_utilization "${GPU_MEM_UTIL}"

        sleep 3

        if [[ -f "${response_file}" ]]; then
            CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate/moderation_as_judge_v4.py \
                --response_file "${response_file}" \
                --moderation "${judge_model}" \
                --aggregate "${AGGREGATE}" \
                --save_path "${SAVE_PATH}"
        else
            echo "[ERROR] 推理输出不存在: ${response_file}"
        fi

        echo "<< [GPU ${gpu_id}] done | $(date '+%F %T')"
    } &> "${log_file}"
}

# ========= 单作业：CodeAttack =========
run_codeattack_job() {
    local gpu_id="$1" stage_name="$2" judge_model="$3"
    local exp_name="$4" base_model="$5" log_file="$6"
    local response_file; response_file="$(build_codeattack_response_file "${exp_name}" "${base_model}")"

    {
        echo ">> [GPU ${gpu_id}] ${stage_name}/CodeAttack/${exp_name} | $(date '+%F %T')"

        cd "${CODEATTACK_DIR}" || { echo "[ERROR] cannot cd ${CODEATTACK_DIR}"; exit 1; }
        CUDA_VISIBLE_DEVICES=${gpu_id} python main_test.py \
            --exp_name "${exp_name}" \
            --data_key "${CODEATTACK_DATA_KEY}" \
            --prompt-type "${CODEATTACK_PROMPT_TYPE}" \
            --target-model "${base_model}" \
            --target-max-n-tokens "${CODEATTACK_MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --tag "${CODEATTACK_TAG}" \
            --tensor_parallel_size "${TP_SIZE}" \
            --gpu_memory_utilization "${GPU_MEM_UTIL}" \
            --seed "${SEED}" \
            --num_sample 1 \
            --start_idx 0 \
            --end_idx -1

        sleep 3

        cd "${WORK_ROOT}" || { echo "[ERROR] cannot cd ${WORK_ROOT}"; exit 1; }
        if [[ -f "${response_file}" ]]; then
            CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate/moderation_as_judge_v4.py \
                --response_file "${response_file}" \
                --moderation "${judge_model}" \
                --aggregate "${AGGREGATE}" \
                --save_path "${SAVE_PATH}"
        else
            echo "[ERROR] CodeAttack 推理输出不存在: ${response_file}"
        fi

        echo "<< [GPU ${gpu_id}] done | $(date '+%F %T')"
    } &> "${log_file}"
}

# ================= GPU 池调度器 =================
GPU_POOL=""
RUNNING_PIDS=()

init_gpu_pool() {
    GPU_POOL="${LOG_DIR}/.gpu_pool.fifo"
    rm -f "${GPU_POOL}"
    mkfifo "${GPU_POOL}"
    exec 9<>"${GPU_POOL}"
    for g in "${gpus[@]}"; do
        echo "${g}" >&9
    done
}

cleanup_on_exit() {
    for pid in "${RUNNING_PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py|CodeAttack/main_test\.py' 2>/dev/null || true
    exec 9>&- 2>/dev/null || true
    exec 9<&- 2>/dev/null || true
    [[ -n "${GPU_POOL}" && -e "${GPU_POOL}" ]] && rm -f "${GPU_POOL}"
}
trap cleanup_on_exit EXIT INT TERM

acquire_gpu() {
    local g
    read -r g <&9
    echo "${g}"
}
release_gpu() {
    echo "$1" >&9
}

dispatch_job() {
    local job_kind="$1" stage_name="$2" judge_model="$3"
    local task_name="$4" base_model="$5" safe_model="$6"

    local gpu_id; gpu_id="$(acquire_gpu)"
    local log_file
    if [[ "${job_kind}" == "codeattack" ]]; then
        log_file="${LOG_DIR}/${safe_model}_${stage_name// /_}_CodeAttack_${task_name}_gpu${gpu_id}.log"
    else
        log_file="${LOG_DIR}/${safe_model}_${stage_name// /_}_${task_name}_gpu${gpu_id}.log"
    fi

    (
        if [[ "${job_kind}" == "codeattack" ]]; then
            run_codeattack_job "${gpu_id}" "${stage_name}" "${judge_model}" \
                "${task_name}" "${base_model}" "${log_file}"
        else
            run_generic_job "${gpu_id}" "${stage_name}" "${judge_model}" \
                "${task_name}" "${base_model}" "${log_file}"
        fi
        release_gpu "${gpu_id}"
        echo "<< [GPU ${gpu_id}] done: ${stage_name}/${task_name} (log: ${log_file})"
    ) &

    local pid=$!
    RUNNING_PIDS+=("${pid}")
    echo ">> [GPU ${gpu_id}] dispatch: ${stage_name}/${task_name} (pid=${pid})"
}

# ================= 主流程 =================
cd "${WORK_ROOT}" || { echo "[FATAL] cannot cd to ${WORK_ROOT}"; exit 1; }
mkdir -p "${LOG_DIR}"
echo "[cwd] $(pwd)"
echo "[LOG_DIR] ${LOG_DIR}"

pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py|CodeAttack/main_test\.py' 2>/dev/null || true
sleep 2

if ! grep -q 'CUDA_VISIBLE_DEVICES (from shell)' "${WORK_ROOT}/evaluate/run_inference.py" 2>/dev/null; then
    echo "[CRITICAL] run_inference.py 未修复，8 个进程会全挤到 GPU 0！"
    exit 1
fi

[[ "${stage1_1}" == "true" ]] && ensure_over_safety_symlinks

init_gpu_pool
echo "[GPU pool] initialized with ${num_gpus} slots: ${gpus[*]}"

for model_entry in "${model_list[@]}"; do
    model_name="${model_entry%%|*}"
    base_model="${model_entry#*|}"
    safe_model="$(basename "${base_model}")"

    echo "=========================="
    echo "🚀 评测模型: ${model_name}"
    echo "   权重: ${base_model}"
    echo "   TAG=${TAG}  gpu_mem_util=${GPU_MEM_UTIL}"
    echo "=========================="

    if [[ ! -e "${base_model}" ]]; then
        echo "[WARN] 权重不存在，跳过: ${base_model}"
        continue
    fi

    declare -a jobs=()

    if [[ "${stage1_2}" == "true" && "${stage_codeattack}" == "true" ]]; then
        for exp_name in "${codeattack_exp_name_list[@]}"; do
            jobs+=("codeattack|Jailbreak|MD-Judge|${exp_name}")
        done
    fi

    if [[ "${stage1_2}" == "true" ]]; then
        for d in "${jailbreak_dataset_name_list[@]}"; do
            jobs+=("generic|Jailbreak|MD-Judge|${d}")
        done
    fi

    if [[ "${stage1}" == "true" ]]; then
        for d in "${safety_dataset_name_list[@]}"; do
            jobs+=("generic|Safety|MD-Judge|${d}")
        done
    fi

    if [[ "${stage1_1}" == "true" ]]; then
        for d in "${over_safety_dataset_name_list[@]}"; do
            jobs+=("generic|Over_Safety|wildguard|${d}")
        done
    fi

    echo "[queue] ${#jobs[@]} jobs for model ${model_name}"

    for job in "${jobs[@]}"; do
        IFS='|' read -r job_kind stage_name judge_model task_name <<< "${job}"
        dispatch_job "${job_kind}" "${stage_name}" "${judge_model}" \
                     "${task_name}" "${base_model}" "${safe_model}"
    done

    echo "[sync] waiting all jobs of model ${model_name} to finish..."
    wait
    RUNNING_PIDS=()

    exec 9>&- 2>/dev/null || true
    exec 9<&- 2>/dev/null || true
    rm -f "${GPU_POOL}"
    init_gpu_pool
    echo "[model done] ${model_name}"
done

echo "🎉 完成！日志: ${LOG_DIR}"
# Export LOG_DIR path so workflow layer can reference it (best effort via stdout).
echo "[auto_run_evaluation_pool] LOG_DIR=${LOG_DIR}"
