nvidia-smi

conda activate /apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env

ENV_BIN="/apdcephfs_jn3/share_535475/common/dellwu/envs/llamafactory_env/bin/"
cd /apdcephfs_jn3/share_535475/common/dellwu/LlamaFactory
lr_list=("5e-5")
epoch=3
batch_size_list=("4")
template="deepseek3"
DEVICES="0,1,2,3,4,5,6,7"
# DEVICES="0"
lora_rank=64
lora_alpha=64
tag=""
port=29500  # 添加port变量定义
# dataset_name="wildjailbreak_train_R1_Qwen_DA_scg_iter_2_random_v1_1"
# dataset_name="wildjailbreak_train_DS-V3.2-Exp-thinking_self_align_v2_sft_v3-0_UF"
# dataset_name="wildjailbreak_train_DS-V3.2-Exp-thinking_self_align_v2_sft_v3-0_UF"
# dataset_name=wildjailbreak_train_R1_Qwen_self_align_v2_1_sft_rnd_mask_hint16_helpful_v3-1_UF
# dataset_name="wildjailbreak_ultrafeedback_DS-R1-14B_restart-1_with_hint"
dataset_name="wildjailbreak_ultrafeedback_DS-R1-14B_RealSafe-R1"
# dataset_name=OpenMathInstruct-2_10k_STAR-S_filtered_Hard

for lr in "${lr_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
    
    # lr_scheduler_type=constant
    lr_scheduler_type=cosine
    # model_folder="/home/share/models"
    # model_folder="/home/dwu/local_models"
    # model_folder="/home/dwu/LLaMA-Factory/models"
    # base_model_name=DeepSeek-R1-Distill-Llama-8B
    base_model_name=DeepSeek-R1-Distill-Qwen-14B
    model_name_or_path="/apdcephfs_nj4/share_300616873/hunyuan/external/DeepSeek-R1-Distill-Qwen-14B"
    # base_model_name=DeepSeek-R1-Distill-Qwen-7B
    # lora_name=${base_model_name}_peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}_adapter
    lora_name=peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}_adapter
    target_model_name=${base_model_name}_peft_${dataset_name}${tag}_r${lora_rank}_${lora_alpha}_${lr}_${lr_scheduler_type}_bs_${batch_size}_ep_${epoch}
    
    # 使用llamafactory-cli train命令进行训练
    CUDA_VISIBLE_DEVICES=$DEVICES ${ENV_BIN}llamafactory-cli train \
        --stage sft \
        --do_train \
        --model_name_or_path ${model_name_or_path} \
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

    # 使用llamafactory-cli export命令合并LoRA
    ${ENV_BIN}llamafactory-cli export \
        --model_name_or_path ${model_name_or_path} \
        --adapter_name_or_path saves/${base_model_name}/lora/${lora_name} \
        --finetuning_type lora \
        --template $template \
        --export_dir models/${target_model_name} \
        --export_size 7 \
        --export_legacy_format False
    done

done
