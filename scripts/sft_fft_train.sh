#!/bin/bash
#SBATCH -J Llama3-Safe-SFT
#SBATCH -o logs_and_outputs/train/Llama3-Safe-SFT-slurm-%j.out                             
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2
#SBATCH -w gpu18


source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune
screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

cd ~/LLaMA-Factory

# nvidia-smi

lr_list=("5e-6")
epoch=3
batch_size_list=("1")
template="deepseek3"
DEVICES="0,1"
# dataset_name="hh-rlhf_sft"
dataset_name="wildjailbreak_train_R1_llama3_DA_harmful_qwen_sft"
seed=42
# dataset_name="PKU-SafeRLHF_R1_sft_v2"
for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
        lr_scheduler_type=cosine
        model_folder="/home/share/models"
        # model_folder="/home/dwu/LLaMA-Factory/models"
        base_model_name=DeepSeek-R1-Distill-Llama-8B
        target_model_name=${base_model_name}_fft_${dataset_name}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}_1k
        CUDA_VISIBLE_DEVICES=$DEVICES DS_SKIP_CUDA_CHECK=1 deepspeed --master_port $port src/train.py \
            --deepspeed examples/deepspeed/ds_z3_config.json \
            --stage sft \
            --do_train \
            --model_name_or_path ${model_folder}/${base_model_name} \
            --dataset ${dataset_name} \
            --template $template \
            --finetuning_type full \
            --lora_target q_proj,v_proj \
            --output_dir models/${target_model_name} \
            --overwrite_cache \
            --overwrite_output_dir \
            --per_device_train_batch_size $batch_size \
            --lr_scheduler_type $lr_scheduler_type \
            --logging_steps 10 \
            --save_strategy epoch \
            --cutoff_len 4096 \
            --learning_rate $lr \
            --seed $seed \
            --max_samples 1000 \
            --num_train_epochs $epoch \
            --plot_loss \
            --fp16 
    done
done

