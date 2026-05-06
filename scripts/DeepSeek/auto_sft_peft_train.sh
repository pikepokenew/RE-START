#!/bin/bash
# =============================================================================
# [AUTO] SFT LoRA PEFT training + merge-export (auto_ prefix version)
#
# 本脚本是 sft_peft_train.sh 的自动化副本，业务行为与原脚本完全一致
# （lora_rank/alpha=64, lr=5e-5, bs=4, epoch=3, cutoff_len=4096, fp16,
#  template=deepseek3, DEVICES=0-7），仅将以下三处硬编码改为可通过环境变量覆盖：
#
#   DATASET_NAME        - 训练数据集名（必填，缺失即非零码退出）
#   BASE_MODEL_NAME     - 基础模型的 tag 名（默认 DeepSeek-R1-Distill-Qwen-14B）
#   MODEL_NAME_OR_PATH  - 基础模型权重路径（默认指向 base 14B 权重）
#   LR_SCHEDULER_TYPE   - 学习率调度器（默认 cosine）
#
# Usage:
#   DATASET_NAME=wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint \
#       bash auto_sft_peft_train.sh
# =============================================================================

set -uo pipefail

# -----------------------------------------------------------------------------
# [AUTO] 强制校验必填环境变量
# -----------------------------------------------------------------------------
if [[ -z "${DATASET_NAME:-}" ]]; then
    echo "[FATAL][auto_sft_peft_train] DATASET_NAME env var is required."
    exit 2
fi

nvidia-smi

conda activate /apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env

ENV_BIN="/apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env/bin/"
cd /apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory

# -----------------------------------------------------------------------------
# Hyper-params (identical to original sft_peft_train.sh)
# -----------------------------------------------------------------------------
lr_list=("5e-5")
epoch=3
batch_size_list=("4")
template="deepseek3"
DEVICES="0,1,2,3,4,5,6,7"
lora_rank=64
lora_alpha=64
tag=""
port=29500

# -----------------------------------------------------------------------------
# [AUTO] Configurable via env (falls back to sensible defaults)
# -----------------------------------------------------------------------------
dataset_name="${DATASET_NAME}"
base_model_name="${BASE_MODEL_NAME:-DeepSeek-R1-Distill-Qwen-14B}"
model_name_or_path="${MODEL_NAME_OR_PATH:-/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B}"
lr_scheduler_type="${LR_SCHEDULER_TYPE:-constant}"

echo "========================================"
echo "  [auto] DATASET_NAME       = ${dataset_name}"
echo "  [auto] BASE_MODEL_NAME    = ${base_model_name}"
echo "  [auto] MODEL_NAME_OR_PATH = ${model_name_or_path}"
echo "  [auto] LR_SCHEDULER_TYPE  = ${lr_scheduler_type}"
echo "========================================"

for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do

    lora_name=peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}_adapter
    target_model_name=${base_model_name}_peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}

    # -------------------------------------------------------------------------
    # Stage A: llamafactory-cli train
    # -------------------------------------------------------------------------
    CUDA_VISIBLE_DEVICES=$DEVICES ${ENV_BIN}llamafactory-cli train \
        --stage sft \
        --do_train \
        --model_name_or_path ${model_name_or_path} \
        --dataset ${dataset_name} \
        --template $template \
        --finetuning_type lora \
        --lora_target all \
        --output_dir saves/${base_model_name}/lora/${lora_name} \
        --overwrite_cache \
        --overwrite_output_dir \
        --per_device_train_batch_size $batch_size \
        --lr_scheduler_type $lr_scheduler_type \
        --logging_steps 10 \
        --save_strategy epoch \
        --learning_rate ${lr} \
        --lora_alpha $lora_alpha \
        --lora_rank $lora_rank \
        --num_train_epochs $epoch \
        --cutoff_len 4096 \
        --plot_loss \
        --fp16

    train_rc=$?
    if [[ "${train_rc}" -ne 0 ]]; then
        echo "[FATAL][auto_sft_peft_train] llamafactory-cli train failed with rc=${train_rc}"
        exit "${train_rc}"
    fi

    # -------------------------------------------------------------------------
    # Stage B: llamafactory-cli export (merge LoRA)
    # -------------------------------------------------------------------------
    ${ENV_BIN}llamafactory-cli export \
        --model_name_or_path ${model_name_or_path} \
        --adapter_name_or_path saves/${base_model_name}/lora/${lora_name} \
        --finetuning_type lora \
        --template $template \
        --export_dir models/${target_model_name} \
        --export_size 7 \
        --export_legacy_format False

    export_rc=$?
    if [[ "${export_rc}" -ne 0 ]]; then
        echo "[FATAL][auto_sft_peft_train] llamafactory-cli export failed with rc=${export_rc}"
        exit "${export_rc}"
    fi

    # -------------------------------------------------------------------------
    # [AUTO] Verify merged model artifacts exist
    # -------------------------------------------------------------------------
    merged_dir="/apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory/models/${target_model_name}"
    if [[ ! -f "${merged_dir}/config.json" ]]; then
        echo "[FATAL][auto_sft_peft_train] missing config.json in ${merged_dir}"
        exit 3
    fi
    # at least one weight shard: *.safetensors or pytorch_model*.bin
    if ! ls "${merged_dir}"/*.safetensors >/dev/null 2>&1 \
         && ! ls "${merged_dir}"/pytorch_model*.bin >/dev/null 2>&1; then
        echo "[FATAL][auto_sft_peft_train] no weight file (*.safetensors or pytorch_model*.bin) in ${merged_dir}"
        exit 3
    fi
    echo "[auto_sft_peft_train] OK: merged model at ${merged_dir}"

    done

done
