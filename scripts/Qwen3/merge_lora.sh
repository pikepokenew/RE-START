#!/bin/bash
#SBATCH -J Llama3-Safe-SFT
#SBATCH -o logs_and_outputs/train/Llama3-Safe-SFT-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 16:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH -w gpu07

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
template="qwen3"
# dataset_name="hh-rlhf_sft"
# dataset_name="hh-rlhf_backtracking_sft"
dataset_name="wildjailbreak_train_R1_Qwen_DA_sft_v2_1"
for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
    
        model_folder="/home/share/models"
        base_model_name=Qwen3-14B
        # base_model_name=Qwen3-8B

        python src/export_model.py \
            --model_name_or_path ${model_folder}/${base_model_name} \
            --adapter_name_or_path /home/dwu/LLaMA-Factory/saves/Qwen3-14B/lora/Qwen3-14B_peft_OpenMathInstruct-2_10k_STAR-S_filtered_r64_64_5e-5_cosine_bs_4_ep_3_adapter/checkpoint-829 \
            --finetuning_type lora \
            --template $template \
            --export_dir models/Qwen3-14B_peft_OpenMathInstruct-2_10k_STAR-S_filtered_r64_64_5e-5_cosine_bs_4_ep_1 \
            --export_size 7 \
            --export_legacy_format False


        # model_folder="/home/share/models"
        # base_model_name=Qwen2.5-Math-1.5B-Instruct

        # python src/export_model.py \
        #     --model_name_or_path ${model_folder}/${base_model_name} \
        #     --adapter_name_or_path /home/dwu/S2R/saved_models/Qwen2.5-Math-1.5B-Instruct_0710 \
        #     --finetuning_type lora \
        #     --template qwen \
        #     --export_dir models/Qwen2.5-Math-1.5B-Instruct_0710_GRPO \
        #     --export_size 7 \
        #     --export_legacy_format False
    done

done
