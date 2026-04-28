#!/bin/bash
#SBATCH -J Tulu2-DPO
#SBATCH -o logs_and_outputs/train/Tulu2-DPO-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_h100:2
#SBATCH -w gpu02

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

cd ~/LLaMA-Factory

lr_list=("5e-6" "1e-5" "5e-5" "1e-4")
batch_size_list=("4" "8" "16")

for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
        CUDA_VISIBLE_DEVICES="0,1" DS_SKIP_CUDA_CHECK=1 deepspeed --master_port $port src/train.py \
            --stage dpo \
            --deepspeed examples/deepspeed/ds_z3_config.json \
            --do_train \
            --model_name_or_path ~/local_models/tulu-2-7b \
            --dataset BeaverTails_Refusal_SafeRLHF \
            --template tulu2 \
            --finetuning_type lora \
            --lora_target q_proj,v_proj \
            --output_dir saves/tulu-2-7b/lora/peft_BeaverTails_Refusal_SafeRLHF_lr_${lr}_batch_${batch_size}_adapter \
            --overwrite_cache \
            --overwrite_output_dir \
            --per_device_train_batch_size ${batch_size} \
            --lr_scheduler_type constant \
            --save_strategy epoch \
            --logging_steps 10 \
            --learning_rate ${lr} \
            --num_train_epochs 3.0 \
            --plot_loss \
            --ddp_timeout 720000000 \
            --max_samples 5000 \
            --fp16 \
            --lora_dropout 0.5
    done
done