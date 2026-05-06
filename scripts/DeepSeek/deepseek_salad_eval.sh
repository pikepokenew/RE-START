#!/bin/bash
# =============================================================================
# DeepSeek-R1-Distill-Qwen-14B 8卡并行评估脚本
# 支持多数据集并行处理，使用 MD-Judge 评估
# =============================================================================
set -u
conda activate /apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env

# ================= 配置 =================
WORK_ROOT="/apdcephfs_jn3/share_535475/common/dellwu/RE-START/"
LOG_DIR="${WORK_ROOT}/logs/eval_8gpu_$(date +%Y%m%d_%H%M%S)"

# GPU配置
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}

# 模型配置（使用管道符分隔格式）
MODELS=(
    "DeepSeek-R1-Distill-Qwen-14B|/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
)

# 数据集配置（使用管道符分隔格式）
DATASETS=(
    "Salad-attack_enhanced_set_sub_v1|${WORK_ROOT}/evaluate/harmful_questions/Salad-attack_enhanced_set_sub_v1.json"
    # 可以添加更多数据集，格式："数据集名称|数据集路径"
    # "advbench|${WORK_ROOT}/evaluate/harmful_questions/advbench.json"
    # "HEx-PHI|${WORK_ROOT}/evaluate/harmful_questions/HEx-PHI.json"
)

# 生成参数
TAG="temp06_n64"
N_GENERATION=64
MAX_NEW_TOKENS=4096
TEMPERATURE=0.6
TOP_P=0.95
SEED=42
TP_SIZE=1
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# 评估配置
SAVE_PATH="evaluate/results"
AGGREGATE="best-of-n"
JUDGE_MODEL="MD-Judge"

# ================= 工具函数 =================

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
    pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py' 2>/dev/null || true
}
trap cleanup_on_exit EXIT INT TERM

# ================= 数据拆分函数 =================
# 获取数据集总行数并计算每个GPU处理的样本数
get_dataset_size() {
    local dataset_path="$1"
    python3 -c "import json; print(len(json.load(open('${dataset_path}'))))" 2>/dev/null || echo "0"
}

# 拆分数据集为8个部分
split_dataset_for_gpus() {
    local dataset_path="$1"
    local output_dir="$2"
    local dataset_name="$3"
    
    local total_samples; total_samples="$(get_dataset_size "${dataset_path}")"
    
    # 对于500条数据，优化分配策略
    local base_samples=$((total_samples / num_gpus))
    local remainder=$((total_samples % num_gpus))
    
    echo "[INFO] 数据集 ${dataset_name} 总样本数: ${total_samples}"
    echo "[INFO] 优化分配: ${num_gpus}个GPU，基础 ${base_samples}条/GPU，余数 ${remainder}条"
    
    # 为每个GPU创建数据子集
    local current_start=0
    for ((i=0; i<num_gpus; i++)); do
        local gpu_samples=$base_samples
        if [[ $i -lt $remainder ]]; then
            gpu_samples=$((gpu_samples + 1))
        fi
        
        local end=$((current_start + gpu_samples))
        local subset_file="${output_dir}/${dataset_name}_subset_gpu${i}.json"
        
        # 使用Python进行精确的数据切片
        python3 -c "
import json
data = json.load(open('${dataset_path}'))
subset = data[${current_start}:${end}]
with open('${subset_file}', 'w') as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)
"
        
        # 同时复制到logs/data_subsets目录，方便run_inference.py查找
        cp "${subset_file}" "logs/data_subsets/" 2>/dev/null || true
        
        echo "✅ GPU ${i} 数据子集: ${subset_file} (样本 ${current_start}-$((end-1))，共${gpu_samples}条)"
        current_start=$end
    done
}

# ================= 并行处理函数 =================
run_gpu_parallel() {
    local model_name="$1" base_model="$2"
    local dataset_name="$3" dataset_path="$4"
    local gpu_id="$5" subset_index="$6"
    
    local subset_file="${LOG_DIR}/data_subsets/${dataset_name}_subset_gpu${subset_index}.json"
    local response_file="${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu${subset_index}_tag_${TAG}.json"
    local log_file="${LOG_DIR}/DeepSeek-R1-Distill-Qwen-14B_${dataset_name}_gpu${gpu_id}.log"

    (
        {
            echo ">> [GPU ${gpu_id}] 开始处理数据子集 ${subset_index} | $(date '+%F %T')"
            echo "   模型: ${model_name}"
            echo "   数据集子集: ${subset_file}"
            echo "   参数: temperature=${TEMPERATURE}, n=${N_GENERATION}"
            echo "   输出文件: ${response_file}"

            # ---- 1) 推理生成 ----
            echo "[GPU ${gpu_id}] 开始推理生成..."
            CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate/run_inference.py \
                --model "${base_model}" \
                --datasets "${dataset_name}_subset_gpu${subset_index}" \
                --save_path "${SAVE_PATH}" \
                --tag "gpu${subset_index}_${TAG}" \
                --n_generation "${N_GENERATION}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --seed "${SEED}" \
                --tensor_parallel_size "${TP_SIZE}" \
                --gpu_memory_utilization "${GPU_MEM_UTIL}"

            sleep 5

            # ---- 2) MD-Judge 评估 ----
            echo "[GPU ${gpu_id}] 开始MD-Judge评估..."
            if [[ -f "${response_file}" ]]; then
                CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate/moderation_as_judge_v4.py \
                    --response_file "${response_file}" \
                    --moderation "${JUDGE_MODEL}" \
                    --aggregate "${AGGREGATE}" \
                    --save_path "${SAVE_PATH}"
                echo "[GPU ${gpu_id}] MD-Judge评估完成"
            else
                echo "[ERROR] 推理输出不存在: ${response_file}"
            fi

            echo "<< [GPU ${gpu_id}] 子集 ${subset_index} 完成 | $(date '+%F %T')"
        } &> "${log_file}"
        echo "<< [GPU ${gpu_id}] 完成子集 ${subset_index}: ${dataset_name} (日志: ${log_file})"
    ) &

    local pid=$!
    RUNNING_PIDS+=("${pid}")
    echo ">> [GPU ${gpu_id}] 派发子集 ${subset_index} (PID=${pid})"
}

# ================= 主流程 =================
cd "${WORK_ROOT}" || { echo "[FATAL] 无法切换到工作目录: ${WORK_ROOT}"; exit 1; }
mkdir -p "${LOG_DIR}"
mkdir -p "${LOG_DIR}/data_subsets"

# 清理本用户残留的推理/裁判进程
pkill -9 -u "$(whoami)" -f 'run_inference\.py|moderation_as_judge_v4\.py' 2>/dev/null || true
sleep 2

# 自检 run_inference.py 是否尊重外部 CUDA_VISIBLE_DEVICES
if ! grep -q 'CUDA_VISIBLE_DEVICES (from shell)' "${WORK_ROOT}/evaluate/run_inference.py" 2>/dev/null; then
    echo "[CRITICAL] run_inference.py 未修复，8 个进程会全挤到 GPU 0！"
    exit 1
fi

echo "=========================================="
echo "🚀 DeepSeek-R1-Distill-Qwen-14B 8卡数据并行评估"
echo "=========================================="
echo "GPU数量: ${num_gpus}"
echo "温度: ${TEMPERATURE}"
echo "生成次数: ${N_GENERATION}"
echo "评估器: ${JUDGE_MODEL}"
echo "日志目录: ${LOG_DIR}"
echo "=========================================="

# 检查模型和数据集
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name base_model <<< "$model_entry"
    
    if [[ ! -e "${base_model}" ]]; then
        echo "[ERROR] 模型权重不存在: ${base_model}"
        exit 1
    fi
    echo "✅ 模型检查通过: ${model_name}"
done

for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name dataset_path <<< "$dataset_entry"
    
    if [[ ! -f "${dataset_path}" ]]; then
        echo "[ERROR] 数据集文件不存在: ${dataset_path}"
        exit 1
    fi
    echo "✅ 数据集检查通过: ${dataset_name}"
done

# 数据拆分阶段
echo ""
echo "[INFO] 开始数据拆分..."
for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name dataset_path <<< "$dataset_entry"
    split_dataset_for_gpus "${dataset_path}" "${LOG_DIR}/data_subsets" "${dataset_name}"
done

# 并行调度执行
echo ""
echo "[INFO] 开始8卡数据并行调度..."

RUNNING_PIDS=()

for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name dataset_path <<< "$dataset_entry"
    
    for model_entry in "${MODELS[@]}"; do
        IFS='|' read -r model_name base_model <<< "$model_entry"
        
        # 为每个GPU启动并行任务
        for ((i=0; i<num_gpus; i++)); do
            gpu_id=${gpus[$i]}
            run_gpu_parallel "$model_name" "$base_model" "$dataset_name" "$dataset_path" "$gpu_id" "$i"
        done
        
        # 等待当前数据集的所有GPU任务完成
        echo "[INFO] 等待数据集 ${dataset_name} 的所有GPU任务完成..."
        wait
        RUNNING_PIDS=()
        sleep 3
    done
done

# 结果合并阶段
echo ""
echo "[INFO] 开始合并8卡生成结果..."
for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name dataset_path <<< "$dataset_entry"
    
    final_response_file="${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_tag_${TAG}.json"
    
    # 合并所有GPU子集的结果
    jq -s 'add' \
        "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu0_tag_${TAG}.json" \
        "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu1_tag_${TAG}.json" \
        "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu2_tag_${TAG}.json" \
        "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu3_tag_${TAG}.json" \
        "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu4_tag_${TAG}.json" \
        "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu5_tag_${TAG}.json" \
        "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu6_tag_${TAG}.json" \
        "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu7_tag_${TAG}.json" \
        > "${final_response_file}" 2>/dev/null
    
    if [[ -f "${final_response_file}" ]]; then
        echo "✅ 合并完成: ${final_response_file}"
        echo "   总样本数: $(jq length "${final_response_file}" 2>/dev/null || echo "无法解析")"
        
        # 清理临时子集文件
        for ((i=0; i<num_gpus; i++)); do
            rm -f "${SAVE_PATH}/${dataset_name}/DeepSeek-R1-Distill-Qwen-14B_gpu${i}_tag_${TAG}.json"
        done
    else
        echo "❌ 合并失败: ${final_response_file}"
    fi
done

echo "🎉 8卡数据并行任务完成！"
echo "日志目录: ${LOG_DIR}"
echo "数据子集目录: ${LOG_DIR}/data_subsets"
echo ""

echo "=========================================="
echo "8卡数据并行脚本执行完毕"
echo "=========================================="