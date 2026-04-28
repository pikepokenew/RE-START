#!/bin/bash
#SBATCH -J DS-R1-DPO
#SBATCH -o logs_and_outputs/train/DS-R1-DPO-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 6:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH -w gpu06

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)
DEVICES="0"
cd ~/LLaMA-Factory

model_folder="/home/dwu/LLaMA-Factory/models"
# model_folder=/home/dwu/local_models
# model_folder="/home/share/models"
base_model_name=DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_R1_llama3_DA_adversarial_sft_r64_64_1e-5_cosine_bs_4_ep_3
lr=5e-6

dataset_name="wildjailbreak_train_R1_llama3_DA_dpo"
lora_rank=64
batch_size=4
epoch=1
lora_name=peft_${dataset_name}_lr_${lr}_r${lora_rank}_bs_${batch_size}_ep_${epoch}_adapter
target_model_name=${base_model_name}_peft_${dataset_name}_${lr}_r${lora_rank}_bs_${batch_size}_ep_${epoch}
template=deepseek3
# # DPO noraml template
CUDA_VISIBLE_DEVICES=$DEVICES DS_SKIP_CUDA_CHECK=1 deepspeed --master_port $port src/train.py \
    --stage dpo \
    --deepspeed examples/deepspeed/ds_z3_config.json \
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
    --lr_scheduler_type cosine \
    --lora_rank $lora_rank \
    --lora_alpha 64 \
    --save_steps 1000 \
    --logging_steps 10 \
    --learning_rate ${lr} \
    --num_train_epochs $epoch \
    --plot_loss \
    --ddp_timeout 720000000 \
    --fp16

# LoRA Merge
echo "----------------- LoRA Merge -----------------"
python src/export_model.py \
    --model_name_or_path ${model_folder}/${base_model_name} \
    --adapter_name_or_path saves/${base_model_name}/lora/${lora_name} \
    --finetuning_type lora \
    --template $template \
    --export_dir models/${target_model_name} \
    --export_size 7 \
    --export_legacy_format False
