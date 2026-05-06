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

TAG="${TAG:-none}"       # 统一用 tag 控制 prompt 策略（见 run_inference.py 的 TAG_REGISTRY）
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
    "DeepSeek-R1-Distill-Qwen-14B|/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
    "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3|/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
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

# ---- CodeAttack (Jailbreak stage, judged by MD-Judge) ----
# 每个 exp_name 对应 CodeAttack/data/data_${exp_name}.json
codeattack_exp_name_list=("python_string_full" "python_list_full" "python_stack_full")
CODEATTACK_DIR="${WORK_ROOT}/CodeAttack"
CODEATTACK_DATA_KEY="code_wrapped_plain_attack"
CODEATTACK_MAX_NEW_TOKENS=16000
CODEATTACK_TAG="${CODEATTACK_TAG:-${TAG}}"    # 默认跟随全局 TAG；需独立指定时再覆盖
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

# ctrl+c 时回收所有后台子 shell + python 残留
RUNNING_PIDS=()
cleanup_on_exit() {
    for pid in "${RUNNING_PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py|CodeAttack/main_test\.py' 2>/dev/null || true
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
                    --tag "${TAG}" \
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


# ================= CodeAttack 并行调度 =================
# 与 run_parallel 不同：
#   - 推理走 CodeAttack/main_test.py（必须 cd CodeAttack，才能读到 ./data/data_{exp}.json）
#   - 裁判走 evaluate/moderation_as_judge_v4.py（在 WORK_ROOT 下执行）
#   - 每个 exp_name 一个子任务
# 产物路径 (main_test.py 默认)：
#   evaluate/results/CodeAttack_{exp_name}_${CODEATTACK_DATA_KEY}/{model_basename}_tag_${CODEATTACK_TAG}.json
run_codeattack_parallel() {
    local stage_name="$1" judge_model="$2"
    shift 2
    local exp_name_list=("$@")

    echo "-------- ${stage_name} (CodeAttack) | Judge=${judge_model} | tag=${CODEATTACK_TAG} | ${num_gpus} GPUs --------"

    local gpu_idx=0
    local pids=()
    local safe_model; safe_model="$(basename "${base_model}")"

    for exp_name in "${exp_name_list[@]}"; do
        local gpu_id=${gpus[$gpu_idx]}
        local response_dir="${SAVE_PATH}/CodeAttack_${exp_name}_${CODEATTACK_DATA_KEY}"
        local response_file="${response_dir}/${safe_model}_tag_${CODEATTACK_TAG}.json"
        local log_file="${LOG_DIR}/${safe_model}_${stage_name// /_}_CodeAttack_${exp_name}_gpu${gpu_id}.log"

        (
            {
                echo ">> [GPU ${gpu_id}] ${stage_name}/CodeAttack/${exp_name} | $(date '+%F %T')"

                # ---- 1) 生成 ----
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

                sleep 5

                # ---- 2) 裁判 (MD-Judge) ----
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
            echo "<< [GPU ${gpu_id}] done: ${stage_name}/CodeAttack/${exp_name} (log: ${log_file})"
        ) &

        local pid=$!
        pids+=("${pid}")
        RUNNING_PIDS+=("${pid}")
        echo ">> [GPU ${gpu_id}] dispatch: ${stage_name}/CodeAttack/${exp_name} (pid=${pid})"

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
    echo "-------- ${stage_name} (CodeAttack) 阶段完成 --------"
}


# ================= 主流程 =================
cd "${WORK_ROOT}" || { echo "[FATAL] cannot cd to ${WORK_ROOT}"; exit 1; }
mkdir -p "${LOG_DIR}"
echo "[cwd] $(pwd)"
echo "[LOG_DIR] ${LOG_DIR}"

# 清理本用户残留的推理/裁判进程
pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py|CodeAttack/main_test\.py' 2>/dev/null || true
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
    [[ "$stage1_2" == "true" ]] && run_codeattack_parallel "Jailbreak" "MD-Judge" "${codeattack_exp_name_list[@]}"
done

echo "🎉 完成！日志: ${LOG_DIR}"
