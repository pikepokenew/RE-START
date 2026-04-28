#!/bin/bash
#SBATCH -J Over-Refusal
#SBATCH -o logs_and_outputs/eval/Over-Refusal-test-slurm-%j.out                           
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 12:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -w gpu11
source /home/dwu/miniconda3/etc/profile.d/conda.sh
run_env=immune
conda activate $run_env
# 定义字符串列表
over_safety_benchmark_list=("OKTest" "XSTest")
# over_safety_benchmark_list=("or-bench_1k")
# model_list=("Qwen3-14B_sys_14" "DeepSeek-R1-Distill-Qwen-14B_sys_14" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_RealSafe_sft_5k_fa2_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_safechain_5k_sft_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_STAR-1_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_safechain_5k_sft_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_STAR-1_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_RealSafe_sft_5k_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_sys_4" "DeepSeek-R1-Distill-Qwen-14B_sys_4" "Qwen3-14B_sys_0" "DeepSeek-R1-Distill-Qwen-14B_sys_0")
# model_list=("Qwen3-14B_sys_14" "DeepSeek-R1-Distill-Qwen-14B_sys_14" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0")
# model_list=("Qwen3-14B_sys_14" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0")
# model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_1_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_1_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_RealSafe_sft_5k_v2_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_RealSafe_sft_5k_v2_r64_64_5e-5_cosine_bs_4_ep_3_sys_0")
# model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-2_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "Qwen3-14B_peft_wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-2_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0")
model_list=("DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_0_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_25_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_50_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0" "DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_self_align_v2_sft_75_mask_hint16_helpful_v3-0_UF_r64_64_5e-5_cosine_bs_4_ep_3_sys_0")

# 方法1: 直接遍历列表
# echo "=== 方法1: 直接遍历 ==="
pool_size=50
# model_name="gpt-4.1"
model_name="gpt-4.1-mini"
for model in "${model_list[@]}"; do
    for benchmark in "${over_safety_benchmark_list[@]}"; do
        echo "处理模型: $model"
        python evaluate/gpt4_as_judge_v2.py --response_file /home/dwu/Immunization/evaluate/results/${benchmark}/${model}.json --pool_size $pool_size --model_name $model_name
    done
done
# echo

# # 方法2: 使用索引遍历
# echo "=== 方法2: 使用索引遍历 ==="
# for i in "${!model_list[@]}"; do
#     echo "索引 $i: ${model_list[i]}"
# done