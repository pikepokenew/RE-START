#!/bin/bash
# =============================================================================
# 8 卡 GPU-pool 并行评测（推理 + 裁判一体化作业）
# -----------------------------------------------------------------------------
# 任务逻辑与 run_evaluation.sh 完全一致：Safety / Over-Safety / Jailbreak / CodeAttack
# 区别只在调度：
#   - 不再按 stage 顺序执行；所有 (dataset + codeattack_exp) 打平成统一作业队列
#   - 固定 8 个 GPU 槽位做生产者-消费者调度：任一 GPU 一空闲就抓下一个作业
#   - CodeAttack 作业放队列最前面，优先启动（max_new_tokens=16000，最耗时）
#   - 单作业 = 单 GPU 上的 "推理 -> 裁判" 闭环，彼此独立、互不等待
#   - 产物路径、命名、判官选择与 run_evaluation.sh 完全一致
#
# 用法：
#   bash scripts/DeepSeek/run_evaluation_pool.sh
#   TAG=self_align_v2 GPU_MEM_UTIL=0.95 bash scripts/DeepSeek/run_evaluation_pool.sh
#   STAGE_CODEATTACK=false bash scripts/DeepSeek/run_evaluation_pool.sh
# =============================================================================
set -u
conda activate /apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env

# ================= 配置 =================
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}

# 允许从环境变量关掉某类任务（与旧脚本语义保持一致）
stage1="${STAGE_SAFETY:-true}"         # Safety      -> MD-Judge
stage1_1="${STAGE_OVER_SAFETY:-true}"  # Over Safety -> wildguard
stage1_2="${STAGE_JAILBREAK:-true}"    # Jailbreak   -> MD-Judge (含 CodeAttack)
stage_codeattack="${STAGE_CODEATTACK:-true}"

WORK_ROOT="/apdcephfs_jn3/share_535475/common/dellwu/RE-START/"
OVER_SAFETY_DIR="${WORK_ROOT}/evaluate/over_safety"
LOG_DIR="${WORK_ROOT}/logs/eval_pool_$(date +%Y%m%d_%H%M%S)"

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

# 模型使用 name|path 的管道格式
model_list=(
    # "DeepSeek-R1-Distill-Qwen-14B|/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
    # "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3|/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
    # "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3|/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
    # "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-3_with_hint_r64_64_5e-5_cosine_bs_4_ep_3|/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-3_with_hint_r64_64_5e-5_cosine_bs_4_ep_3"
    "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_constant_bs_4_ep_3|/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_ultrafeedback_DS-R1-14B_restart-2_with_hint_r64_64_5e-5_constant_bs_4_ep_3"
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
    # 与 run_inference.py / moderation_as_judge_v4.py 的产物命名保持一致
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

# ========= 单作业：Generic (harmful_questions 下的一个数据集) =========
# 参数：gpu_id  stage_name  judge_model  dataset_name  base_model  log_file
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
# 参数：gpu_id  stage_name  judge_model  exp_name  base_model  log_file
run_codeattack_job() {
    local gpu_id="$1" stage_name="$2" judge_model="$3"
    local exp_name="$4" base_model="$5" log_file="$6"
    local response_file; response_file="$(build_codeattack_response_file "${exp_name}" "${base_model}")"

    {
        echo ">> [GPU ${gpu_id}] ${stage_name}/CodeAttack/${exp_name} | $(date '+%F %T')"

        # 推理：必须 cd 到 CodeAttack 目录（它读 ./data/data_{exp}.json）
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

        # 裁判：回到 WORK_ROOT
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
# 设计：
#   - 用 FIFO named pipe 做"空闲 GPU 令牌池"，初始注入 num_gpus 个令牌
#   - 每开始一个作业：从 pipe 读一个 gpu_id（阻塞等待）
#   - 子 shell 执行完后把同一个 gpu_id 写回 pipe，归还令牌
#   - 这样调度粒度 = 单作业，而不是旧脚本里的"一整轮"，GPU 利用率最大化

GPU_POOL=""
RUNNING_PIDS=()

init_gpu_pool() {
    GPU_POOL="${LOG_DIR}/.gpu_pool.fifo"
    rm -f "${GPU_POOL}"
    mkfifo "${GPU_POOL}"
    # 为 pipe 保留一个读端 fd（避免所有读者退出导致写端阻塞）
    exec 9<>"${GPU_POOL}"
    for g in "${gpus[@]}"; do
        echo "${g}" >&9
    done
}

cleanup_on_exit() {
    # 杀掉所有在途子 shell 和 python 残留
    for pid in "${RUNNING_PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py|CodeAttack/main_test\.py' 2>/dev/null || true
    # 关闭并清理 fifo
    exec 9>&- 2>/dev/null || true
    exec 9<&- 2>/dev/null || true
    [[ -n "${GPU_POOL}" && -e "${GPU_POOL}" ]] && rm -f "${GPU_POOL}"
}
trap cleanup_on_exit EXIT INT TERM

# 从池中阻塞读取一个 gpu_id
acquire_gpu() {
    local g
    read -r g <&9
    echo "${g}"
}
# 归还一个 gpu_id 到池
release_gpu() {
    echo "$1" >&9
}

# ========= 通用作业派发 =========
# 参数：job_kind  stage_name  judge_model  task_name  base_model  safe_model
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

# 清理本用户残留的推理/裁判进程
pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py|CodeAttack/main_test\.py' 2>/dev/null || true
sleep 2

# 自检
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

    # ---- 构建当前模型的统一作业队列 ----
    # 顺序策略：CodeAttack 最耗时 -> 排最前，其它按 stage 类别顺次追加。
    # 这样调度器先把 3 个长作业抛给前 3 张卡，其余 5 张卡立刻开始消化短作业。
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

    # ---- 逐个派发，acquire_gpu 会阻塞直到有空闲 GPU ----
    for job in "${jobs[@]}"; do
        IFS='|' read -r job_kind stage_name judge_model task_name <<< "${job}"
        dispatch_job "${job_kind}" "${stage_name}" "${judge_model}" \
                     "${task_name}" "${base_model}" "${safe_model}"
    done

    # 等待当前模型的全部作业完成，再切到下一个模型（避免不同模型在同一卡上争显存）
    echo "[sync] waiting all jobs of model ${model_name} to finish..."
    wait
    RUNNING_PIDS=()

    # 重置 GPU 令牌池（保险起见，防止有作业因异常未归还令牌）
    exec 9>&- 2>/dev/null || true
    exec 9<&- 2>/dev/null || true
    rm -f "${GPU_POOL}"
    init_gpu_pool
    echo "[model done] ${model_name}"
done

echo "🎉 完成！日志: ${LOG_DIR}"
