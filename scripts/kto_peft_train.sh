#!/bin/bash
#SBATCH -J Llama3-KTO
#SBATCH -o logs_and_outputs/train/Llama3-KTO-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH -w gpu07

source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune

screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

cd ~/LLaMA-Factory

# model_folder=/home/dwu/local_models
# base_model_name=DeepSeek-R1-Distill-Llama-8B
# template=deepseek3

model_folder="/home/dwu/LLaMA-Factory/models"
base_model_name=DeepSeek-R1-Distill-Llama-8B_peft_PKU-SafeRLHF_R1_llama3_DA_spec_rules_sft_r64_64_2e-4_constant_bs_4_ep_6
template=deepseek3

lr_list=("1e-5")
batch_size_list=("4")
lora_rank=64
for lr in "${lr_list[@]}"; do

    for batch_size in "${batch_size_list[@]}"; do
        # dataset_name="PKU-SafeRLHF_kto"
        dataset_name="PKU-SafeRLHF_R1_llama3_DA_spec_rules_kto"
        epoch=3
        lr_scheduler_type=constant
        lora_name=peft_${dataset_name}_lr_${lr}_bs_${batch_size}_r_${lora_rank}_bs_${batch_size}_${lr_scheduler_type}_ep_${epoch}_adapter
        target_model_name=${base_model_name}_peft_${dataset_name}_lr_${lr}_bs_${batch_size}_r${lora_rank}_${lr_scheduler_type}_ep_${epoch}
        CUDA_VISIBLE_DEVICES="0,1,2,3" DS_SKIP_CUDA_CHECK=1 deepspeed --master_port $port src/train.py \
            --stage kto \
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
            --logging_steps 10 \
            --learning_rate ${lr} \
            --save_strategy epoch \
            --lora_rank $lora_rank \
            --lora_alpha 64 \
            --num_train_epochs $epoch \
            --plot_loss \
            --ddp_timeout 720000000 \
            --bf16

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

