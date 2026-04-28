#!/bin/bash
#SBATCH -J Llama3-DPO
#SBATCH -o logs_and_outputs/train/Llama3-DPO-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:4
#SBATCH -w gpu18

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

cd ~/LLaMA-Factory

model_folder="/home/dwu/LLaMA-Factory/models"
# model_folder=/home/dwu/local_models
base_model_name=DeepSeek-R1-Distill-Llama-8B_fft_PKU-SafeRLHF_R1_llama3_DA_sft_1k_1e-5_cosine_bs_4_ep_3_1k
lr=1e-5
# dataset_name="PKU-SafeRLHF-Prefix-5k"
dataset_name="PKU-SafeRLHF_R1_llama3_qwen_mix_dpo"
# lora_name=peft_${dataset_name}_lr_${lr}_adapter
epoch=1

target_model_name=${base_model_name}_fft_${dataset_name}_${lr}_ep_${epoch}
# # DPO noraml template
CUDA_VISIBLE_DEVICES="0,1,2,3" DS_SKIP_CUDA_CHECK=1 deepspeed --master_port $port src/train.py \
    --stage dpo \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --do_train \
    --model_name_or_path ${model_folder}/${base_model_name} \
    --dataset ${dataset_name} \
    --template deepseek3 \
    --finetuning_type full \
    --lora_target q_proj,v_proj \
    --output_dir models/${target_model_name} \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --lr_scheduler_type cosine \
    --save_steps 1000 \
    --logging_steps 10 \
    --learning_rate ${lr} \
    --num_train_epochs $epoch \
    --plot_loss \
    --ddp_timeout 720000000 \
    --fp16
