#!/bin/bash
#SBATCH -J Llama3-KTO
#SBATCH -o logs_and_outputs/train/Llama3-KTO-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:a100-sxm4-80gb:2
#SBATCH -w gpu06

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

cd ~/LLaMA-Factory

model_folder=/home/share/models
base_model_name=DeepSeek-R1-Distill-Llama-8B
template=deepseek3

# model_folder="/home/dwu/LLaMA-Factory/models"
# base_model_name=Meta-Llama-3-8B-Instruct_peft_PKU-SafeRLHF_R1_SFT_1e-5_cosine_batch_8_epoch_10
# template=deepseek3

DEVICES="0,1"
lr_list=("1e-5")
batch_size_list=("1")

for lr in "${lr_list[@]}"; do

    for batch_size in "${batch_size_list[@]}"; do
        dataset_name="PKU-SafeRLHF_R1_llama3_DA_kto"
        epoch=3
        lr_scheduler_type=cosine
        target_model_name=${base_model_name}_fft_${dataset_name}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}_1k
        CUDA_VISIBLE_DEVICES=$DEVICES DS_SKIP_CUDA_CHECK=1 deepspeed --master_port $port src/train.py \
            --stage kto \
            --deepspeed examples/deepspeed/ds_z3_config.json \
            --do_train \
            --model_name_or_path ${model_folder}/${base_model_name} \
            --dataset ${dataset_name} \
            --template $template \
            --pref_beta 0.1 \
            --finetuning_type full \
            --lora_target q_proj,v_proj \
            --output_dir models/${target_model_name} \
            --overwrite_cache \
            --overwrite_output_dir \
            --per_device_train_batch_size $batch_size \
            --lr_scheduler_type $lr_scheduler_type \
            --save_strategy epoch \
            --logging_steps 10 \
            --learning_rate ${lr} \
            --num_train_epochs $epoch \
            --plot_loss \
            --max_samples 1000 \
            --ddp_timeout 720000000 \
            --bf16

    done
done

