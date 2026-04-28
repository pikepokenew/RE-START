#!/bin/bash
#SBATCH -J Llama3-Safe-SFT
#SBATCH -o logs_and_outputs/train/Llama3-Safe-SFT-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 16:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH -w gpu09

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

# GSM8K SFT
cd ~/LLaMA-Factory

lr_list=("5e-5")
epoch=5
batch_size_list=("8")
template="deepseek3"
# dataset_name="hh-rlhf_sft"
# dataset_name="hh-rlhf_backtracking_sft"
dataset_name="wildjailbreak_train_R1_Qwen_DA_sft_v2_1"
for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
    
    lr_scheduler_type=cosine
    model_folder="/home/dwu/local_models"
    # model_folder="/home/dwu/LLaMA-Factory/models"
    base_model_name=DeepSeek-R1-Distill-Qwen-14B
    lora_name=${base_model_name}_peft_${dataset_name}_${lr}_${lr_scheduler_type}_bs_${batch_size}_epoch_${epoch}_adapter
    target_model_name=${base_model_name}_peft_${dataset_name}_${lr}_${lr_scheduler_type}_bs_${batch_size}_epoch_${epoch}

    python src/export_model.py \
        --model_name_or_path ${model_folder}/${base_model_name} \
        --adapter_name_or_path /home/dwu/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-14B/lora/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_1_r64_64_1e-5_constant_bs_4_ep_3_adapter \
        --finetuning_type lora \
        --template $template \
        --export_dir models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_1_r64_64_1e-5_constant_bs_4_ep_3 \
        --export_size 7 \
        --export_legacy_format False
    done

done
