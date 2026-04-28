#!/bin/bash
#SBATCH -J DS-R1-Safe-SFT
#SBATCH -o logs_and_outputs/train/DS-R1-Safe-SFT-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -w gpu17

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

cd ~/LLaMA-Factory

# model_folder=/home/share/models
model_folder="/home/dwu/LLaMA-Factory/models"
# base_model_name=DeepSeek-R1-Distill-Qwen-14B
base_model_name=DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen-14B_DA_dpo_2k_1e-6_cosine_ep_1
template=deepseek3
model_name=$base_model_name
# base_model_name=DeepSeek-R1-Distill-Llama-8B
# template=deepseek3
tag="_1k"
lr_list=("1e-6")
batch_size_list=("1")
lora_rank=64
lora_alpha=64

for lr in "${lr_list[@]}"; do

    for batch_size in "${batch_size_list[@]}"; do
        # dataset_name="PKU-SafeRLHF_kto"
        dataset_name="wildjailbreak_train_R1_Qwen-14B_DA_dpo_iter_1"
        epoch=1
        lr_scheduler_type=cosine
        lora_name=peft_${dataset_name}${tag}_lr_${lr}_${lr_scheduler_type}_ep_${epoch}_adapter
        # target_model_name=${model_name}_peft_${dataset_name}${tag}_${lr}_${lr_scheduler_type}_ep_${epoch}
        target_model_name=${model_name}_iterative_dpo_tier_1
        # target_model_name=${base_model_name}${tag}
        CUDA_VISIBLE_DEVICES="0" DS_SKIP_CUDA_CHECK=1 deepspeed --master_port $port src/train.py \
            --stage dpo \
            --deepspeed examples/deepspeed/ds_z3_config.json \
            --do_train \
            --model_name_or_path ${model_folder}/${base_model_name} \
            --dataset ${dataset_name} \
            --template $template \
            --pref_beta 0.1 \
            --finetuning_type lora \
            --lora_target all \
            --output_dir saves/${base_model_name}/lora/${lora_name} \
            --overwrite_cache \
            --overwrite_output_dir \
            --per_device_train_batch_size $batch_size \
            --lr_scheduler_type $lr_scheduler_type \
            --save_strategy epoch \
            --logging_steps 10 \
            --learning_rate ${lr} \
            --lora_alpha $lora_alpha \
            --lora_rank $lora_rank \
            --num_train_epochs $epoch \
            --plot_loss \
            --cutoff_len 4096 \
            --max_samples 1000 \
            --ddp_timeout 720000000 \
            --fp16

        python src/export_model.py \
            --model_name_or_path ${model_folder}/${base_model_name} \
            --adapter_name_or_path saves/${base_model_name}/lora/${lora_name} \
            --finetuning_type lora \
            --template $template \
            --export_dir models/${target_model_name}\
            --export_size 7 \
            --export_legacy_format False
    done
done
