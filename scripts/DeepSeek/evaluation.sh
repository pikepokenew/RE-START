#!/bin/bash
# 8 卡并行评测：Safety / Over-Safety / Jailbreak
# 用法: bash scripts/DeepSeek/evaluation.sh
#       GPU_MEM_UTIL=0.95 bash scripts/DeepSeek/evaluation.sh
set -u
conda activate /apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env

# ================= 配置 =================
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}

stage1="true"    # Safety      -> MD-Judge
stage1_1="true"  # Over Safety -> wildguard
stage1_2="true"  # Jailbreak   -> MD-Judge

WORK_ROOT="/apdcephfs_jn3/share_535475/common/dellwu/RE-START/"
OVER_SAFETY_DIR="${WORK_ROOT}/evaluate/over_safety"
LOG_DIR="${WORK_ROOT}/logs/eval_$(date +%Y%m%d_%H%M%S)"

NEED_SYSTEM_PROMPT=0
N_GENERATION=1
MAX_NEW_TOKENS=4096
TEMPERATURE=0.0
TOP_P=1.0
SEED=42
TP_SIZE=1
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

SAVE_PATH="evaluate/results"
AGGREGATE="best-of-n"

model_list=(
    # "DeepSeek-R1-Distill-Qwen-14B|/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
    # "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3|/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
    "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3|/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
)

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
        echo "${SAVE_PATH}/${dataset_name}/${mb}_${N_GENERATION}_sys_${NEED_SYSTEM_PROMPT}.json"
    else
        echo "${SAVE_PATH}/${dataset_name}/${mb}_sys_${NEED_SYSTEM_PROMPT}.json"
    fi
}

# ctrl+c 时回收所有后台子 shell + python 残留
RUNNING_PIDS=()
cleanup_on_exit() {
    for pid in "${RUNNING_PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py' 2>/dev/null || true
}
trap cleanup_on_exit EXIT INT TERM


# ================= 并行调度 =================
run_parallel() {
    local stage_name="$1" judge_model="$2"
    shift 2
    local datasets=("$@")

    echo "-------- ${stage_name} | Judge=${judge_model} | ${num_gpus} GPUs --------"

    local gpu_idx=0
    local pids=()

    for dataset_name in "${datasets[@]}"; do
        local gpu_id=${gpus[$gpu_idx]}
        local response_file; response_file="$(build_response_file "${dataset_name}" "${base_model}")"
        local safe_model;    safe_model="$(basename "${base_model}")"
        local log_file="${LOG_DIR}/${safe_model}_${stage_name// /_}_${dataset_name}_gpu${gpu_id}.log"

        (
            {
                echo ">> [GPU ${gpu_id}] ${stage_name}/${dataset_name} | $(date '+%F %T')"

                CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate/run_inference.py \
                    --model "${base_model}" \
                    --datasets "${dataset_name}" \
                    --save_path "${SAVE_PATH}" \
                    --need_system_prompt "${NEED_SYSTEM_PROMPT}" \
                    --n_generation "${N_GENERATION}" \
                    --max_new_tokens "${MAX_NEW_TOKENS}" \
                    --temperature "${TEMPERATURE}" \
                    --top_p "${TOP_P}" \
                    --seed "${SEED}" \
                    --tensor_parallel_size "${TP_SIZE}" \
                    --gpu_memory_utilization "${GPU_MEM_UTIL}"

                sleep 5

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
            echo "<< [GPU ${gpu_id}] done: ${stage_name}/${dataset_name} (log: ${log_file})"
        ) &

        local pid=$!
        pids+=("${pid}")
        RUNNING_PIDS+=("${pid}")
        echo ">> [GPU ${gpu_id}] dispatch: ${stage_name}/${dataset_name} (pid=${pid})"

        gpu_idx=$(( (gpu_idx + 1) % num_gpus ))
        if [[ ${gpu_idx} -eq 0 ]]; then
            wait "${pids[@]}"
            pids=(); RUNNING_PIDS=()
            sleep 3
        fi
    done

    if [[ ${#pids[@]} -gt 0 ]]; then
        wait "${pids[@]}"
        pids=(); RUNNING_PIDS=()
    fi
    echo "-------- ${stage_name} 阶段完成 --------"
}


# ================= 主流程 =================
cd "${WORK_ROOT}" || { echo "[FATAL] cannot cd to ${WORK_ROOT}"; exit 1; }
mkdir -p "${LOG_DIR}"
echo "[cwd] $(pwd)"
echo "[LOG_DIR] ${LOG_DIR}"

# 清理本用户残留的推理/裁判进程
pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py' 2>/dev/null || true
sleep 2

# 自检 run_inference.py 是否尊重外部 CUDA_VISIBLE_DEVICES
if ! grep -q 'CUDA_VISIBLE_DEVICES (from shell)' "${WORK_ROOT}/evaluate/run_inference.py" 2>/dev/null; then
    echo "[CRITICAL] run_inference.py 未修复，8 个进程会全挤到 GPU 0！"
    exit 1
fi

[[ "$stage1_1" == "true" ]] && ensure_over_safety_symlinks

for model_entry in "${model_list[@]}"; do
    model_name="${model_entry%%|*}"
    base_model="${model_entry#*|}"

    echo "=========================="
    echo "🚀 评测模型: ${model_name}"
    echo "   权重: ${base_model}"
    echo "   gpu_mem_util=${GPU_MEM_UTIL}"
    echo "=========================="

    if [[ ! -e "${base_model}" ]]; then
        echo "[WARN] 权重不存在，跳过: ${base_model}"
        continue
    fi

    [[ "$stage1"   == "true" ]] && run_parallel "Safety"      "MD-Judge"  "${safety_dataset_name_list[@]}"
    [[ "$stage1_1" == "true" ]] && run_parallel "Over_Safety" "wildguard" "${over_safety_dataset_name_list[@]}"
    [[ "$stage1_2" == "true" ]] && run_parallel "Jailbreak"   "MD-Judge"  "${jailbreak_dataset_name_list[@]}"
done

echo "🎉 完成！日志: ${LOG_DIR}"
