#!/bin/bash
#SBATCH -J Llama3-Safe-SFT
#SBATCH -o logs_and_outputs/train/Llama3-Safe-SFT-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 2:00:00     
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
template="deepseek3"
# dataset_name="hh-rlhf_sft"
# dataset_name="hh-rlhf_backtracking_sft"
dataset_name="wildjailbreak_train_R1_Qwen_DA_sft_v2_1"
for lr in "${lr_list[@]}"; do

    for batch_size in "${batch_size_list[@]}"; do
        
        # #################### DeepSeek-R1-Distill-Qwen-14B ####################
        model_folder="/home/share/models"
        # model_folder="/home/dwu/local_models"
        # model_folder="/home/dwu/LLaMA-Factory/models"
        # base_model_name=DeepSeek-R1-Distill-Qwen-14B
        # model_folder="/home/dwu/local_models"
        # base_model_name=DeepSeek-R1-0528-Qwen3-8B
        # base_model_name=Qwen3-8B
        base_model_name=DeepSeek-R1-Distill-Qwen-14B

        python src/export_model.py \
            --model_name_or_path ${model_folder}/${base_model_name} \
            --adapter_name_or_path /home/dwu/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-14B/lora/peft_wildjailbreak_train_R1_Qwen_RealSafe_sft_5k_r64_64_5e-5_cosine_bs_4_ep_3_adapter \
            --finetuning_type lora \
            --template $template \
            --export_dir models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_RealSafe_sft_5k_v2_r64_64_5e-5_cosine_bs_4_ep_3 \
            --export_size 7 \
            --export_legacy_format False

        # #################### DeepSeek-R1-Distill-Llama-8B ####################
        # # model_folder="/home/share/models"
        # model_folder="/home/share/models"
        # base_model_name=DeepSeek-R1-Distill-Llama-8B
        # # base_model_name=Qwen3-8B
        # python src/export_model.py \
        #     --model_name_or_path ${model_folder}/${base_model_name} \
        #     --adapter_name_or_path /home/dwu/LLaMA-Factory/saves/DeepSeek-R1-Distill-Llama-8B/lora/DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_R1_Qwen3_DA_sft_5k_r64_64_5e-5_cosine_bs_8_ep_3_adapter \
        #     --finetuning_type lora \
        #     --template $template \
        #     --export_dir models/DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_R1_Qwen3_DA_sft_5k_r64_64_5e-5_cosine_bs_8_ep_3 \
        #     --export_size 7 \
        #     --export_legacy_format False

        # ################################ DPO ################################
        # python src/export_model.py \
        #     --model_name_or_path /home/share/models/DeepSeek-R1-Distill-Llama-8B \
        #     --adapter_name_or_path /home/dwu/LLaMA-Factory/saves/DeepSeek-R1-Distill-Llama-8B/lora/DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_Qwen3-14B_DA_sft_mix_iter_2_r64_64_5e-5_cosine_bs_8_ep_3_adapter \
        #     --finetuning_type lora \
        #     --template $template \
        #     --export_dir models/DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_Qwen3-14B_DA_sft_mix_iter_2_r64_64_5e-5_cosine_bs_8_ep_3 \
        #     --export_size 7 \
        #     --export_legacy_format False
        # ################################ KTO ################################
        # python src/export_model.py \
        #     --model_name_or_path /home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3 \
        #     --adapter_name_or_path /home/dwu/LLaMA-Factory/saves/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3/lora/peft_wildjailbreak_train_R1_Qwen_DA_kto_v2_16_lr_1e-5_cosine_ep_1_adapter \
        #     --finetuning_type lora \
        #     --template $template \
        #     --export_dir models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v2_16_r64_64_5e-5_constant_bs_4_ep_3_peft_wildjailbreak_train_R1_Qwen_DA_kto_v2_16_lr_1e-5_cosine_ep_1 \
        #     --export_size 7 \
        #     --export_legacy_format False

    done

done
