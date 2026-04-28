#!/bin/bash
#SBATCH -J DS-R1-Safe-SFT
#SBATCH -o logs_and_outputs/train/DS-R1-Safe-SFT-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -w gpu18

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

nvidia-smi

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991
export WANDB_DISABLED="false"

port=$(shuf -i25000-30000 -n1)

# GSM8K SFT
cd ~/LLaMA-Factory

lr_list=("5e-5")
epoch=3
batch_size_list=("4")
template="glmz1"
DEVICES="0"
lora_rank=64
lora_alpha=64
# dataset_name="wildjailbreak_train_Qwen3-14B_DA_sft_Qwen3-14B_prefix_1"
# dataset_name="wildjailbreak_train_Qwen3-14B_self_align_sft_rnd_mask_with_hint_iter_2"
dataset_name="wildjailbreak_train_GLM-Z1-9B_self_align_v10_rnd_mask_hint10_ufv4_sft_iter_1"
tag=""
# dataset_name="wildjailbreak_train_R1_Qwen_DA_sft_v2"

for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
    
    # lr_scheduler_type=constant
    lr_scheduler_type=cosine
    model_folder="/home/dwu/local_models"
    # base_model_name=Qwen3-8B
    base_model_name=GLM-Z1-9B-0414

    lora_name=${base_model_name}_peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}_adapter
    target_model_name=${base_model_name}_peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}

    CUDA_VISIBLE_DEVICES=$DEVICES DS_SKIP_CUDA_CHECK=1 deepspeed --master_port $port src/train.py \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --stage sft \
        --do_train \
        --model_name_or_path ${model_folder}/${base_model_name} \
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

    # Merge LoRA

    python src/export_model.py \
        --model_name_or_path ${model_folder}/${base_model_name} \
        --adapter_name_or_path saves/${base_model_name}/lora/${lora_name} \
        --finetuning_type lora \
        --template $template \
        --export_dir models/${target_model_name} \
        --export_size 7 \
        --export_legacy_format False
    done

done
