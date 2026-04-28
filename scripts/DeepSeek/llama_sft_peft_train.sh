#!/bin/bash
#SBATCH -J DS-R1-Safe-SFT
#SBATCH -o logs_and_outputs/train/DS-R1-Safe-SFT-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 6:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -w gpu18

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

# GSM8K SFT
cd ~/LLaMA-Factory

lr_list=("5e-5")
epoch=3
batch_size_list=("8")
template="deepseek3"
DEVICES="0"
lora_rank=64
lora_alpha=64
tag=""
# dataset_name="wildjailbreak_train_R1_Qwen_DA_sft_gcg_iter_2_random_v1_1"
dataset_name="wildjailbreak_train_Phi-4_SA_sft_iter_2_v2"

for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
    
    lr_scheduler_type=cosine
    model_folder="/home/share/models"
    # model_folder="~/local_models"
    # model_folder="/home/dwu/LLaMA-Factory/models"
    base_model_name=DeepSeek-R1-Distill-Llama-8B
    # base_model_name=DeepSeek-R1-Distill-Qwen-14B
    # base_model_name=DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_R1_Llama_DA_sft_random_r64_64_5e-5_constant_bs_4_ep_3_iter_2
    # base_model_name=Llama-3.1-Nemotron-Nano-8B-v1

    # base_model_name=DeepSeek-R1-Distill-Qwen-7B
    lora_name=${base_model_name}_peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}_adapter
    target_model_name=${base_model_name}_peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}
    # target_model_name=DeepSeek-R1-Distill-Llama-8B_peft_wildjailbreak_train_Qwen3-14B_SA_sft_5k_r64_64_5e-5_cosine_bs_8_ep_3_iter_1

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
