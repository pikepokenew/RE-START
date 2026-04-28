#!/bin/bash
#SBATCH -J Gen-Data
#SBATCH -o logs_and_outputs/Gen-Data-Script-slurm-%j.out                             
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH -t 24:00:00     
#SBATCH --mem=64gb
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -w gpu18


source /home/dwu/miniconda3/etc/profile.d/conda.sh

conda activate immune
screen -dmS clash /home/dwu/clash/clash -f /home/dwu/clash/config.yaml
export http_proxy=http://127.0.0.1:8991 && export https_proxy=http://127.0.0.1:8991 && export all_proxy=http://127.0.0.1:8991

port=$(shuf -i25000-30000 -n1)

# ###################################################################
cd ~/Immunization

python evaluate/generate_responses_v4.py --model ${model_name_or_path} --dataset evaluate/harmful_questions/wildjailbreak_train.json --n_generation 1 --max_new_tokens 4096 --temperature 1.0 --need_system_prompt $system_prompt --num_samples $num_samples

python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/wildjailbreak_train/DeepSeek-R1-Distill-Llama-8B_temp_1.0_n_8_sys_8.json --moderation wildguard

# 生成数据
# Qwen-8B/14B

# # 用DS-Distill-Qwen-14B生成续写数据
# python evaluate/generate_responses_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --prefix 1 --max_new_tokens 4096 --dataset evaluate/results/${dataset_name}/Qwen3-8B_sys_0.json --tag deliberative_alignment_v2

# python evaluate/generate_responses_with_prefix.py --model /home/share/models/DeepSeek-R1-Distill-Qwen-14B --prefix 1 --max_new_tokens 4096 --dataset evaluate/results/${dataset_name}/Qwen3-14B_sys_0.json --tag deliberative_alignment_v2

# model_name="DeepSeek-R1-Distill-Qwen-14B"
# model_name="DeepSeek-R1-Distill-Qwen-14B"
# model_name="DeepSeek-R1-Distill-Qwen-14B"
# gen_data_model="DeepSeek-R1-Distill-Qwen-14B"
# dataset_name="wildjailbreak_train"
# python evaluate/generate_responses_v3.py --model /home/share/models/${model_name} --dataset evaluate/harmful_questions/${dataset_name}.json --max_new_tokens 4096 --num_samples 10000 --temperature 0.0

# python evaluate/generate_responses_v3.py --model /home/share/models/${model_name} --dataset evaluate/harmful_questions/${dataset_name}.json --max_new_tokens 4096 --num_samples 10000 --temperature 0.0 --need_system_prompt 8
# python evaluate/moderation_as_judge_v3.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_8.json --moderation wildguard


# 批量生成的续写数据，其中从第2步-第10步前缀开始续写
# steps_list=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
# steps_list=("1")
# n_generation=3
# temperature=1.0

# python evaluate/generate_responses_with_prefix_v2.py --model /home/dwu/LLaMA-Factory/models/DeepSeek-R1-Distill-Qwen-14B_peft_wildjailbreak_train_R1_Qwen_DA_sft_v4_r64_64_5e-5_constant_bs_4_ep_3   --num_samples -1 --dataset /home/dwu/Immunization/evaluate/harmful_questions/CodeChameleon_target_llama-2_tree.json  --prefix 0 --max_new_tokens 8192 --n_generation 10 --temperature 0.6
# for step in "${steps_list[@]}"; do
#     # python evaluate/generate_responses_with_prefix_v2.py --model /home/share/models/${gen_data_model}   --num_samples 5000 --dataset evaluate/results/${dataset_name}/${model_name}_sys_0.json  --prefix ${step} --max_new_tokens 4096 --tag deliberative_alignment_v2 --n_generation 3 --temperature 1.00



#     # python evaluate/moderation_as_judge_v4.py --response_file evaluate/results/${dataset_name}/${model_name}_sys_0_${gen_data_model}_sys_0_deliberative_alignment_v2_temp_${temperature}_n_${n_generation}_prefix_${step}_v2.json --moderation wildguard

# done
